import torch
from torch.nn import (Linear, Module, Softmax, 
                      Embedding, LayerNorm, 
                      Sequential, LeakyReLU,
                      TransformerEncoder, 
                      TransformerEncoderLayer)
from torch.distributions import (constraints, 
                                 Distribution, 
                                 Gamma, Poisson)
import torch.nn.functional as F
from SpatialGPT.utils import get_library


class TransformerModel(Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.n_layer = args.n_layer
        self.n_head = args.n_head
        self.fast = args.fast
        self.f_dim = args.f_dim
        self.dropout = args.dropout
        
        transformer_layer = TransformerEncoderLayer(self.d_model, self.n_head, 
                                                    self.f_dim, self.dropout,
                                                    batch_first=True, norm_first=True)
        self.transformer_encoder = TransformerEncoder(transformer_layer, self.n_layer)
    
    def forward(self, tokens):
        emb = self.transformer_encoder(tokens)
        return emb


class TaskDecoder(Module):
    def __init__(self, args):
        super().__init__()
        self.cls_decoder = ClsDecoder(args)
        self.exp_decoder = ExpDecoder(args)
        self.dropout_sim = DropoutSim(args)
        
    def forward(self, x):
        x_rate, theta = self.cls_decoder(x[:,0,:])
        cell_exp = self.exp_decoder(x[:,1:,:])
        return x_rate, theta, cell_exp
    


class ClsDecoder(Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.out_dim = args.hvgs
        self.recover = Sequential(Linear(self.d_model, self.d_model),
                                  LeakyReLU(),
                                  Linear(self.d_model, self.d_model),
                                  LeakyReLU(),
                                  Linear(self.d_model, self.out_dim))
        self.esp = 1e-5
        self.log_theta = torch.nn.Parameter(torch.randn(self.out_dim), requires_grad=True)
        
    def forward(self, x):
        l = get_library(x)
        x_rate = l * self.recover(x)
        theta = F.softplus(self.log_theta) + self.esp
        return x_rate, theta


class ExpDecoder(Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.recover = Sequential(Linear(self.d_model, self.d_model),
                                  LeakyReLU(),
                                  Linear(self.d_model, self.d_model),
                                  LeakyReLU(),
                                  Linear(self.d_model, 1))
        
    def forward(self, x):
        pred_exp = self.recover(x).squeeze(-1)
        return pred_exp
        
    
    
class DropoutSim(Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.infonce = InfoNCE(args)
        
    def forward(self, cls, exp):
        exp_mask = mask_feature(exp, self.dropout)
        ctr = self.infonce(cls, exp_mask)
        return ctr
        



class NegBinom(Distribution):
    """
    Gamma-Poisson mixture approximation of Negative Binomial(mean, dispersion)

    lambda ~ Gamma(mu, theta)
    x ~ Poisson(lambda)
    """
    arg_constraints = {
        'mu': constraints.greater_than_eq(0),
        'theta': constraints.greater_than_eq(0),
    }
    support = constraints.nonnegative_integer

    def __init__(self, mu, theta, eps=1e-10):
        """
        Parameters
        ----------
        mu : torch.Tensor
            mean of NegBinom. distribution
            shape - [# genes,]

        theta : torch.Tensor
            dispersion of NegBinom. distribution
            shape - [# genes,]
        """
        self.mu = mu
        self.theta = theta
        self.eps = eps
        super(NegBinom, self).__init__(validate_args=True)

    def sample(self):
        lambdas = Gamma(
            concentration=self.theta + self.eps,
            rate=(self.theta + self.eps) / (self.mu + self.eps),
        ).rsample()

        x = Poisson(lambdas).sample()

        return x

    def log_prob(self, x):
        """log-likelihood"""
        ll = torch.lgamma(x + self.theta) - \
             torch.lgamma(x + 1) - \
             torch.lgamma(self.theta) + \
             self.theta * (torch.log(self.theta + self.eps) - torch.log(self.theta + self.mu + self.eps)) + \
             x * (torch.log(self.mu + self.eps) - torch.log(self.theta + self.mu + self.eps))

        return ll



class InfoNCE(Module):
    def __init__(self, args, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.args = args
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.temperature = args.tau

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=1., reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)
    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def mask_feature(x, mask_prob):
    """
    Args:
        x (torch.Tensor): shape (num_nodes, num_features)
        mask_prob (float): ratio
    Returns:
        torch.Tensor
    """
    mask = torch.bernoulli(torch.full(x.shape, 1 - mask_prob)).to(x.device)
    x_masked = x * mask
    return x_masked

# ! unused code
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


class SGModels(Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.latent_dim = args.latent_dim
        self.args = args
        self.embed_dim = args.embed_dim
        # self.bp_attention = BipolarAttention(args, self.embed_dim)
        self.bp_attention = Attention(args, self.embed_dim)

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