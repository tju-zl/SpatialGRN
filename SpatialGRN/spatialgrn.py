import os
from tqdm.notebook import trange
import matplotlib.pyplot as plt
from scipy.sparse import issparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from .utils import get_device, get_log_dir, get_output_dir
from .model import SGRNModel, ComputeLosses
from .data import prepare_dataset, compute_edge


class SpatailGRN:
    def __init__(self, args, adata):

        args.device = get_device(args)
        args.log_dir = get_log_dir(args)
        args.output_dir = get_output_dir(args)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        
        # adata preprocess
        self.adata = prepare_dataset(args, adata)
        self.x = torch.FloatTensor(self.adata.X.toarray())
        self.edge_index = compute_edge(args, self.adata).to('cpu')
        args.n_spots = self.x.size(0)
        
        # model initial
        self.model = SGRNModel(args).to(args.device)
        self.compute_losses = ComputeLosses(args).to(args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wegiht_decay)
        
        self.args = args

        
    def fit(self, emb):
        if emb.requires_grad:
            print(emb.requires_grad)
            emb = emb.detach()
        dataset = TensorDataset(emb, self.x)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        self.model.train()
        losses = []
        
        for ep in trange(self.args.max_epoch):
            for batch in dataloader:
                embedding, xx = batch
                self.optimizer.zero_grad()
                output = self.model(embedding.to(self.args.device))[-1]
                loss = self.compute_losses.loss(xx.to(self.args.device), output)
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
            
    def eval(self, emb):
        if emb.requires_grad:
            print(emb.requires_grad)
            emb = emb.detach()
        dataset = TensorDataset(emb)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
        z_save_path = os.path.join(self.args.output_dir, os.path.basename(self.args.dataset_path)+'_latent.npy')
        att_save_path = os.path.join(self.args.output_dir, os.path.basename(self.args.dataset_path)+'_att.npy')
        self.model.eval()
        self.args.eval=True
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                att, z = self.model(batch[0].to(self.args.device))
                for i in range (att.shape[0]):
                    top_att = np.sort(att[i].flatten())[-10:]
                if i == 0:
                    np.save(att_save_path, att.cpu().numpy())
                    np.save(z_save_path, z.cpu().numpy())
                else:
                    with open(att_save_path, 'ab') as f:
                        np.save(f, att.cpu().numpy())
                    with open(z_save_path, 'ab') as f:
                        np.save(f, z.cpu().numpy())
        return att_save_path, z_save_path
        # todo some functions of downstream analysis
        # self.adata.obsm['grn_mat'] = attention.numpy()
        # return self.model(emb.to(self.args.device))[0].detach().cpu()
        # return self.adata
