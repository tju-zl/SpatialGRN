import torch
from torch.nn import Module, Linear, ReLU, Sequential
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .module import classicAttention, BipolarAttention


class SGRNModel(Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.latent_dim = args.latent_dim
        self.args = args
        self.embed_dim = args.embed_dim
        # self.bp_attention = BipolarAttention(args, self.embed_dim)
        self.bp_attention = classicAttention(args, self.embed_dim)
        
        # layers = []
        # hidden_dim = [2*args.latent_dim, 4*args.latent_dim]
        # in_dim = args.latent_dim
        # for dim in hidden_dim:
        #     layers.append(Linear(in_dim, dim))
        #     layers.append(ReLU())
        #     in_dim = dim
        # layers.append(Linear(in_dim, args.hvgs))
        # self.decoder = Sequential(*layers)
        self.decoder = Linear(args.hvgs, args.hvgs)
        
    def forward(self, emb):
        att_weights, z= self.bp_attention(emb)

        z = torch.mean(z, dim=0)
        
        if self.args.eval:
            return att_weights, z
        
        output = self.decoder(z)
        
        return att_weights, z, output


class ComputeLosses(Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        
    def loss(self, x, y):
        lo = F.mse_loss(x, y)
        return lo
