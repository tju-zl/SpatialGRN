import torch
from torch.nn import Linear, Module, Softmax, Embedding, LayerNorm
import torch.nn.functional as F


# tanh attention
class BiAttention(Module):
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


# valina attention
class Attention(Module):
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

class TransformerModel(Module):
    def __init__(self, ) -> None:
         super().__init__()
         pass
     
    def forward(self, args, ):
        pass


# get the embedding of gene id.
class GeneIDEmb(Module):
    def __init__(self, args):
        super().__init__()
        self.emb_dim = args.gid_dim  # default 50.
        self.n_token = args.n_token
        self.embedding = Embedding(self.n_token, self.emb_dim)
        self.norm = LayerNorm(self.emb_dim)
    
    def forward(self, idx):
        emb = self.embedding(idx)
        emb = self.norm(emb)
        return emb
