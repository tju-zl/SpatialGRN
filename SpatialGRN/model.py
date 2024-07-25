import torch
from torch.nn import Module, ModuleList, Transformer, Softmax
import torch_geometric
from torch_geometric.nn import MessagePassing


class SGRN(Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        
    def forward(self):
        pass
    

class ComputeLoss(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def loss(self):
        pass

