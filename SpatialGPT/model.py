import torch
from torch.nn import Module, Linear, ReLU, Sequential
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .module import *
from .utils import get_library


## * SpatialGPT overall model
class SGModel(Module):
    def __init__(self, args, ):
        super().__init__()
        self.d_model = args.d_model
        self.cls_token = nn.Parameter(torch.zeros(1,1,self.d_model))
        self.gene_id = GeneIDEmb(args)
    
        self.encoder = TransformerModel(args)
        self.decoder = TaskDecoder(args)
        self.total_loss = ComputeLosses(args)
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x, stoken, idx):
        # token_dim: [batch x genes x emb]
        gene_id_emb = self.gene_id(idx).expand(stoken.shape[0], -1, -1)
        # print(self.gene_id(idx).shape, gene_id_emb.shape, stoken.shape)
        stoken = torch.cat((gene_id_emb, stoken), dim=-1)
        cls_token = self.cls_token.expand(stoken.shape[0], -1, -1)
        spatial_token = torch.cat((cls_token, stoken), dim=-2)
        emb, att = self.encoder(spatial_token)
        l = get_library(x)
        x_rate, theta, cell_exp = self.decoder(l, emb)
        loss = self.total_loss(x, x_rate, theta, cell_exp)
        return emb, att, loss



## * Overall loss computation
class ComputeLosses(Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.dropsim = DropoutSim(args)
        
    def forward(self, x, x_rate, theta, cell_exp):
        x_ = torch.log(1+x)
        re_gene = F.mse_loss(x_, cell_exp)
        
        negb = NegBinom(x_rate, torch.exp(theta))
        re_cls = -negb.log_prob(x).sum(-1).mean()
        
        do_sim = self.dropsim(negb.sample(), cell_exp)
        
        return re_gene + re_cls + 0.1 * do_sim


