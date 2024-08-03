import torch
from torch.nn import Module, Linear, ReLU, Sequential
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


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
        self.decoder = Linear(args.embed_dim, args.hvgs)
        
    def forward(self, emb):
        att_weights, z= self.bp_attention(emb)

        z = torch.mean(z, dim=1)
        
        if self.args.eval:
            return att_weights, z
        
        output = self.decoder(z)
        
        return att_weights, z, output


class BipolarAttention(Module):
    def __init__(self, args, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.query = Linear(embed_dim, args.latent_dim)
        self.key = Linear(embed_dim, args.latent_dim)
        self.value = Linear(embed_dim, args.latent_dim)
        
        self.out = Linear(args.latent_dim, args.embed_dim)
    
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
            
            self.out = Linear(args.latent_dim, args.embed_dim)
        
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
        super().__init__()
        self.args = args
        
    def loss(self, x, y):
        lo = F.mse_loss(x, y)
        return lo
