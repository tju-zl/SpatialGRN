import torch
from torch.nn import Module, ModuleList, Transformer, Softmax
from torch_geometric.utils import add_remaining_self_loops, spmm


class SGRNModel(Module):
    def __init__(self, args, in_channels) -> None:
        super().__init__()
        self.latent_dim = args.latent_dim
        self.args = args
        self.inchannels = in_channels
        n_layers = args.embed_dim
        
    def forward(self, emb):
        pass
    



class BipolarAttention(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class ComputeLosses(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def loss(self):
        pass


