import os
from tqdm.notebook import trange
import matplotlib.pyplot as plt
from scipy.sparse import issparse
import torch

from .utils import set_random_seed, get_device, get_log_dir, get_output_dir
from .model import SGRNModel, ComputeLosses


class SpatailGRN:
    def __init__(self, args, adata):
        set_random_seed(args.seed)
        args.device = get_device(args)
        args.log_dir = get_log_dir(args)
        args.output_dir = get_output_dir(args)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        self.args = args
        
        # adata preprocess
        
        
        
        # model initial
        self.model = SGRNModel(args).to(args.device)
        self.compute_losses = ComputeLosses(args).to(args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wegiht_decay)
        
    def fit(self):
        self.model.train()
        losses = []
        
        for ep in trange(self.args.max_epoch):
            self.optimizer.zero_grad()
            z = self.model()
            loss = self.compute_losses.loss(z)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            
            if ep % (self.args.max_epoch/self.args.log_steps) == 0 and self.args.visualize:
                    print(f'EP[%4d]: loss=%.4f.' % (ep, loss.item()))
            
        if self.args.visualize:
            x = range(1, len(losses)+1)
            plt.plot(x, losses)
            plt.show()
            
    def analysis(self):
        self.model.eval()
        with torch.no_grad():
            z = self.model(self.adata_x, self.edge_index)[0]
        
        # todo some functions of downstream analysis
        self.adata.obsm['latent'] = z.to('cpu').detach().numpy()
        return self.adata
