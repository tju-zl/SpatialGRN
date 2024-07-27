import torch
from torch.nn import Module, ModuleList, Transformer, Softmax
import torch_geometric
from torch_geometric.nn import MessagePassing, SAGEConv
from torch_geometric.nn.aggr import Aggregation
# random walk


class SGRNModel(Module):
    def __init__(self, args, in_channels) -> None:
        super().__init__()
        self.latent_dim = args.latent_dim
        self.args = args
        self.inchannels = in_channels
        n_layers = args.embed_dim
        self.gene_embedding = GeneRep(args, in_channels, n_layers)
        
    def forward(self):
        pass
    

# Similar to graphSage, while the linear transformation is removed. 
class GeneRep(MessagePassing):
    def __init__(self, args, in_channels, n_layers):
        super().__init__()
        self.args = args
        
    def forward(self, x, edge_list):
        pass


class BipolarAttention(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class ComputeLosses(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def loss(self):
        pass

