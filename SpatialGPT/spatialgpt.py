import os
from tqdm.notebook import trange
import matplotlib.pyplot as plt
from scipy.sparse import issparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import wandb
from pathlib import Path
import json

from SpatialGPT.utils import get_device, get_log_dir, get_output_dir, EarlyStopping, get_library
from SpatialGPT.model import SGModel
from SpatialGPT.data import prepare_dataset, compute_edge
from SpatialGPT.visualization import *
from SpatialGPT.tokenizer import GeneRep, GeneID, gene_in_voc, gene2idx
from torch.optim.lr_scheduler import StepLR


class SpatailGPT:
    def __init__(self, args):
        # system environment
        args.device = get_device(args)
        args.log_dir = get_log_dir(args)
        args.output_dir = get_output_dir(args)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        self.log_dir = args.log_dir
        self.output_dir = args.output_dir
        
        # # setting wandb
        # wandb.init(project='SpatialGPT', config=self.args)  # following args must consistent with config
        
        self.vocab = GeneID.from_file(Path(args.vocab_path))
        # model initial
        self.model = SGModel(args).to(args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, 
                                          weight_decay=args.wegiht_decay, eps=1e-4 if args.amp else 1e-8)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.9)
        # self.early_stopping = EarlyStopping(patience=10, delta=0.001, path=self.output_dir+'/best_model.pt')
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        self.args = args
        

    def adata2tensor(self, adata):
        adata = gene_in_voc(self.vocab, adata)
        self.adata = prepare_dataset(self.args, adata)
        self.gene_idx = gene2idx(self.vocab, self.adata)
        self.x = torch.FloatTensor(self.adata.X.toarray())
        self.edge_index = compute_edge(self.args, self.adata).to('cpu')
        self.args.n_spots = self.x.size(0)
        self.exp_token = GeneRep(self.args)
    
        
    def fit(self, adata):
        self.adata2tensor(adata)
        emb = self.exp_token(self.x, self.edge_index)
        # print(emb, emb.requires_grad)

        dataset = TensorDataset(emb, self.x)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        
        self.model.train()
        self.args.eval=False
        losses = []
        
        for ep in trange(self.args.max_epoch):
            running_loss = 0.0
            for batch in dataloader:
                with torch.cuda.amp.autocast(enabled=self.args.amp):
                    embedding, xx = batch
                    latent, att, loss = self.model(xx.to(self.args.device), 
                                        embedding.to(self.args.device),
                                        torch.tensor(self.gene_idx).to(self.args.device))
                    
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                running_loss += loss.item()
            losses.append(running_loss)
            self.scheduler.step()
            
            # # earlystopping function to add.
            # self.early_stopping(running_loss, self.model)
            # if self.early_stopping.early_stop:
            #     print('Early Stopping.')
            #     break
            
            # print loss info
            if ep % (self.args.max_epoch/self.args.log_steps) == 0 and self.args.visual:
                    print(f'EP[%2d]: loss=%.4f.' % (ep+1, running_loss))

        # plotting loss
        plot_loss_curve(self.args, losses)
        torch.save(self.model.state_dict(), self.output_dir+'/best_model.pt')
        # save model
        # if not self.early_stopping.early_stop:
        #     torch.save(self.model.state_dict(), self.output_dir+'/best_model.pt')

    def batch_fit(self, adata_list):
        # get the stoken before training
        emb_list = []
        for adata in adata_list:
            self.adata2tensor(adata)
            emb_list.append(self.exp_token(self.x, self.edge_index).detach())
        
        self.model.train()
        self.args.eval=False
        losses = []
        for ep in trange(self.args.max_epoch):
            running_loss = 0.0
            for i, emb in enumerate(emb_list):
                self.adata2tensor(adata_list[i])
                dataset = TensorDataset(emb, self.x)
                dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
                for batch in dataloader:
                    with torch.cuda.amp.autocast(enabled=self.args.amp):
                        embedding, xx = batch
                        latent, att, loss = self.model(xx.to(self.args.device), 
                                            embedding.to(self.args.device),
                                            self.gene_idx.to(self.args.device))
                        
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    running_loss += loss.item()
            losses.append(running_loss)
            self.scheduler.step()

    # pretained downsteam task
    def eval_emb(self, emb):
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
    def eval_adata(self, adata):
        self.adata2tensor(adata)
        emb = self.exp_token(self.x, self.edge_index)
        if emb.requires_grad:
            print(emb.requires_grad)
            emb = emb.detach()
        dataset = TensorDataset(emb, self.x)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        
        self.model.eval()
        losses = []
        
        for ep in trange(self.args.max_epoch):
            running_loss = 0.0
            for batch in dataloader:
                with torch.cuda.amp.autocast(enabled=self.args.amp):
                    embedding, xx = batch
                    emb, att, loss = self.model(xx.to(self.args.device), 
                                        embedding.to(self.args.device),
                                        self.gene_idx.to(self.args.device))
                    
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                running_loss += loss.item()
            losses.append(running_loss)
            self.scheduler.step()
            
            # print loss info
            if ep % (self.args.max_epoch/self.args.log_steps) == 0 and self.args.visualize:
                    print(f'EP[%4d]: loss=%.4f.' % (ep, loss.item()))
                    
                    
    # TODO Finetune task
    def finetune_adata(self, args_new, adata):
        pass
    
    def finetune_emb(self, args_new, emb): # finetune emb
        pass