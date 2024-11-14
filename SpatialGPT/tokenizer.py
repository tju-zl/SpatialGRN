from typing import Optional

import numpy as np

import torch
from torch import Tensor
from torch.nn import Parameter, Module
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value

from .utils import random_walk_path


## * from MPNN based GCN
@torch.jit._overload
def gcn_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, improved, add_self_loops, flow,
        dtype):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> OptPairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, improved, add_self_loops, flow,
        dtype):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(  # noqa: F811
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    improved: bool = False,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None
    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


## * Multi-scale gene expression embedding: Node2vec + MPNN-GCN
# Similar to graphSage, while the linear transformation is removed. 
class GeneRep(MessagePassing):
    def __init__(self, args, add_self_loops=True, normalize=True, improved = False, cached=False):
        super().__init__(flow=args.flow, aggr='add')
        self.normalize = normalize
        self.n_gcns = args.n_gcn
        self.n_layers = 1 + args.n_gcn + args.n_randwalk
        self.n_walks = args.n_randwalk
        self.add_self_loops = add_self_loops
        self.improved = improved
        self.cached = cached
        self.args = args
        
        self._cached_edge_index = None
        self._cached_adj_t = None
        
    def forward(self, x, edge_index, edge_weight: OptTensor = None):
        
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(0), self.improved,
                        self.add_self_loops, self.args.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]
        
        # embed x
        emb = []
        emb.append(x)
        
        # embed k-hop neighbors
        for i in range(self.n_gcns):
            emb.append(self.propagate(edge_index, x=emb[i], edge_weight=edge_weight))
        
        # embed k-times random walks
        edge_index_list = random_walk_path(self.args, edge_index)
        if self.n_walks != len(edge_index_list):
            raise ValueError('hypara of random walk error')
        for edge in edge_index_list:
            edge_index1, edge_weight1 = gcn_norm(edge, edge_weight=None, num_nodes=x.size(0), flow=self.args.flow, dtype=x.dtype)
            emb.append(self.propagate(edge_index1, x=x, edge_weight=edge_weight1))

        return torch.cat(emb).view(self.n_layers,x.size(0),x.size(1)).permute(1,2,0)
    
    def message(self, x_j, edge_weight: OptTensor):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return spmm(adj_t, x, reduce=self.aggr)


## * Gene ID vocab: from gene name to index.
from torchtext.vocab import Vocab
import torchtext.vocab as torch_vocab
from typing import Dict, Iterable, List, Optional, Tuple, Union
from typing_extensions import Self
from collections import Counter, OrderedDict
from pathlib import Path
import json


class GeneID(Vocab):
    def __init__(self, gene_vocab: Union[List[str], Vocab], 
                 specials: Optional[List[str]]=None, 
                 special_first: bool=True, 
                #  default_token: Optional[str]='<pad>'
                 ):
        """
        Vocabulary of genes, from scGPT.
        
        Args:
            gene_vocab (Vocab): list or vocab: list of genes or a vocab object.
            specials (List[str]): list of special tokens (cls, pad).
            special_first (bool, optional): Add special token to the beginning. Defaults to True.
            default_token (str, optional): Defaults to '<pad>'.
        """
        if isinstance(gene_vocab, Vocab):
            _vocab = gene_vocab
            if specials is not None:
                raise ValueError(
                    "receive non-empty specials when init from a Vocab object."
                )
        elif isinstance(gene_vocab, list):
            _vocab = self._build_vocab_from_iterator(
                gene_vocab,
                specials=specials,
                special_first=special_first,
            )
        else:
            raise ValueError(
                "gene_vocab must be a list of gene names or a Vocab object."
            )
        super().__init__(_vocab.vocab)
        # if default_token is not None and default_token in self:
        #     self.set_default_token(default_token)
    
    def _build_vocab_from_iterator(
        self,
        iterator: Iterable,
        min_freq: int = 1,
        specials: Optional[List[str]] = None,
        special_first: bool = True,
    ) -> Vocab:
        """
        init the [] vocab.
        """

        counter = Counter()
        counter.update(iterator)

        if specials is not None:
            for tok in specials:
                del counter[tok]

        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[0])
        sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)

        if specials is not None:
            if special_first:
                specials = specials[::-1]
            for symbol in specials:
                ordered_dict.update({symbol: min_freq})
                ordered_dict.move_to_end(symbol, last=not special_first)

        word_vocab = torch_vocab.vocab(ordered_dict, min_freq=min_freq)
        return word_vocab
    
    @classmethod
    def from_file(cls, path: Union[Path, str]):
        """
        load vocab from a file, default format is json of str to index mapping.

        Args:
            path (Union[Path, str]): path of vocab
        """
        if isinstance(path, str):
            path = Path(path)
        try:
            with path.open('r') as f:
                token2idx = json.load(f)
        except:
            raise ValueError(f'{path} not found.')
        return cls.from_dict(token2idx)
    
    @classmethod
    def from_dict(cls, token2idx: Dict[str, int], 
                #   default_token: Optional[str]='<pad>'
                  ):
        """
        load vocabulary from a dict.

        Args:
            token2idx (Dict[str, int]): dict mapping gene name to indics.
            default_token (Optional[str], optional): Defaults to '<pad>'.
        """
        _vocab = cls([])
        
        for t,i in sorted(token2idx.items(), key=lambda x: x[1]):
            _vocab.insert_token(t, i)
        
        # if default_token is not None and default_token in _vocab:
        #     _vocab.set_default_token(default_token)
        
        # special_tokens = ['<pad>', '<eoc>']
        # for s in special_tokens:
        #     if s not in _vocab:
        #         _vocab.append_token(s)
        
        return _vocab

    # def set_default_token(self, defualt_token):
    #     if defualt_token not in self:
    #         raise ValueError(f'{defualt_token} is not in the vocabulary.')
    #     self.set_default_index(self[defualt_token])
        

# # remove the genes not in vocab, gene name to idx
# def gene2idx(vocab, adata):
#     adata.var['gene_names'] = adata.var.index.tolist()
#     adata.var['id_in_vocab'] = [1 if gene in vocab else -1 for gene in adata.var["gene_names"]]
#     gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
#     print(f'match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes'
#           f'in vocabulary of size {len(vocab)}')
    
#     adata = adata[:, adata.var["id_in_vocab"] >= 0]
#     gene_name = adata.var["gene_names"].tolist()
#     gene_idx = np.array(vocab(gene_name), dtype=int)
#     return adata, gene_idx

# remove the genes not in vocab, gene name to idx
def gene_in_voc(vocab, adata):
    adata.var['gene_names'] = adata.var.index.tolist()
    adata.var['id_in_vocab'] = [1 if gene in vocab else -1 for gene in adata.var["gene_names"]]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    print(f'match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes '
          f'in vocabulary of size {len(vocab)}')
    
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    return adata


def gene2idx(vocab, adata):
    gene_name = adata.var["gene_names"].tolist()
    gene_idx = np.array(vocab(gene_name), dtype=int)
    return gene_idx