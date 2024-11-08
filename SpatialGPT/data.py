import scanpy as sc
import numpy as np
import torch
import time
from datasets import load_dataset
from torch_geometric.nn.pool import radius_graph
from torch_geometric.utils import to_undirected
import re
import anndata as ad
import pandas as pd


# 0. download dataset (in preparation, coming soon).
def get_srt(args, dataset=None, data_path=None, make_adata=True):
    """ All SRT datasets are stored in Hugging Face Hub.
    Args:
        args (NameSpace): global configeration.
        dataset (Str, optional): the name of dataset. Defaults to 'all'.
        data_path (Str, optional): dataset dictionary. Defaults to args.dataset_path.
        make_adata (bool, optional): generate h5ad adata file. Defaults to True.
    """
    star_t = time.time()
    assert (dataset), 'If you don\'t provide the specific dataset, all available datasets will download.'
    if data_path is None:
        data_path = args.dataset_path
    
    if dataset == '12-sclice_DLPFC':
        # well known benchmark dataset for spatial clustering methods.
        url = 'SpatialOmics/12-slices_DLPFC'
        slices = ['151507', '151508', '151509', '151510', '151670', '151671', '151672', '151673', '151374', '151675', '151676']
        dataset = load_dataset(url, cache_dir=data_path)
        print(f'dataset: {dataset} has downloaded to {dataset.cache_files}')
        if make_adata:
            pass
    
    elif dataset == None:
        pass


# 1. remove repeat gene (mean), before the prepare step.
def rr_gene(adata):
    raw_dim = len(adata.var_names)
    adata.var_names = [ re.sub(r'\.\d+$', '', gene) for gene in adata.var_names]
    df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, columns=adata.var_names, index=adata.obs_names)
    df_mean = df.groupby(df.columns, axis=1).mean().round().astype(int)
    adata_new = ad.AnnData(df_mean.values, obs=adata.obs, var=pd.DataFrame(index=df_mean.columns))
    remove_dim = len(adata_new.var_names)
    print(f'removed {remove_dim-raw_dim} repreat genes.')
    return adata_new


# 2. preprocess dataset (filter genes and cells, hvg, norm if mlp)
def prepare_dataset(args, adata):
    adata.var_names_make_unique()
    # filter genes and cells (quality control)
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.filter_cells(adata, min_genes=200)
    
    # hvgs and norm
    if args.hvgs < adata.X.shape[0]:
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=args.hvgs)
        if args.decoder == 'MLP':
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.scale(adata)
        adata_hvg = adata[:, adata.var['highly_variable']]    
        return adata_hvg
    else:
        if args.decoder == 'MLP':
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.scale(adata, zero_center=True, max_valu=10)
        return adata


# 3. compute edges of spots using coordinates.
def compute_edge(args, adata):
    coordinate = torch.FloatTensor(adata.obsm['spatial']).to(args.device)
    edge_index = radius_graph(coordinate, args.srt_resolution, max_num_neighbors=args.max_neighbors, flow=args.flow)
    edge_index = to_undirected(edge_index)
    print('Average spatial edge:', edge_index.size()[1] / adata.n_obs)
    return edge_index


# 4. dataloader
def dataset_loader(args, data):
    pass
