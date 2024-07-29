import torch
from torch.nn import Module, Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SGRNModel(Module):
    def __init__(self, args, embed_dim) -> None:
        super().__init__()
        self.latent_dim = args.latent_dim
        self.args = args
        self.embed_dim = embed_dim
        self.bp_attention = BipolarAttention(args, self.embed_dim)
        self.compute_loss = ComputeLosses(args)
        self.decoder = GCNConv(args.latent_dim, args.hvgs, add_self_loops=True)
        
    def forward(self, emb, edge_index):
        att_weights, z= self.bp_attention(emb)

        z = torch.mean(z, dim=1)
        
        output = self.decoder(z, edge_index)
        
        return att_weights, z, output


class BipolarAttention(Module):
    def __init__(self, args, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.query = Linear(embed_dim, args.latent_dim)
        self.key = Linear(embed_dim, args.latent_dim)
        self.value = Linear(embed_dim, args.latent_dim)
        
        self.out = Linear(args.latent_dim, args.latent_dim)
    
    def forward(self, x):
        n_embed = x.size(-1)
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        att_scores = torch.matmul(Q, K.transpose(-2, -1)) / (n_embed ** 0.5)
        att_weights = torch.tanh(att_scores)
        
        att_output = torch.matmul(att_weights, V)
        output = self.out(att_output)
            
        return att_weights, output


class classicAttention(Module):
        def __init__(self, args, embed_dim):
            super().__init__()
            self.embed_dim = embed_dim
            
            self.query = Linear(embed_dim, args.latent_dim)
            self.key = Linear(embed_dim, args.latent_dim)
            self.value = Linear(embed_dim, args.latent_dim)
            
            self.out = Linear(args.latent_dim, args.latent_dim)
        
        def forward(self, x):
            n_embed = x.size(-1)
            
            Q = self.query(x)
            K = self.key(x)
            V = self.value(x)
            
            att_scores = torch.matmul(Q, K.transpose(-2, -1)) / (n_embed ** 0.5)
            att_weights = F.softmax(att_scores, dim=-1)
            
            att_output = torch.matmul(att_weights, V)
            output = self.out(att_output)
            
            return att_weights, output


class ComputeLosses(Module):
    def __init__(self, args) -> None:
        super().__init__(args)
        
    def loss(self):
        pass


