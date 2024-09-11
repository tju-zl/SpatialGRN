import os
import random
import torch
import numpy as np
from sklearn.cluster import KMeans
import numpy as np
from node2vec import Node2Vec
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def get_device(args):
    if args.gpu != -1:
        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    args.device = device
    print('using device {}'.format(device))
    return device


# Training info
def get_log_dir(args):
    if not os.path.exists('../Log/'):
        os.makedirs('../Log/')
    log_dir = '../Log/' + '_'.join([os.path.basename(args.dataset_path).split('.')[0], args.version])
    return log_dir


# Results info
def get_output_dir(args):
    if not os.path.exists('../Output/'):
        os.makedirs('../Output/')
    output_dir = '../Output/' + '_'.join([os.path.basename(args.dataset_path), args.version])
    return output_dir


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

# random walk path generation
def random_walk_path(args, edge_index):
    edge_list = []
    data = Data(edge_index=edge_index, num_nodes=args.n_spots)
    G = to_networkx(data)
    for length in args.n_randomwalk:
        edge_list.append(Node2Vec(G, p=1, q=args.q_randomwalk[0], walk_length=length, num_walks=1, workers=1, quiet=True).walks)
        edge_list.append(Node2Vec(G, p=1, q=args.q_randomwalk[1], walk_length=length, num_walks=1, workers=1, quiet=True).walks)
    
    walks_int = []
    for edge in edge_list:
        walks_int.append([[int(node) for node in walk] for walk in edge])
    
    edge_index_list = []
    n_walk_list = [item for item in args.n_randomwalk for _ in range(2)]
    for k in range(len(n_walk_list)):
        edges = [[],[]]
        for i in range(args.n_spots):
            for j in range(n_walk_list[k]):
                edges[0].append(walks_int[k][i][j])
                edges[1].append(i)
        edge_index_list.append(torch.tensor(np.array(edges), dtype=torch.long))
    return edge_index_list


# find neighbors of a node
def find_neighbors(node, edge_index):
    row, col = edge_index
    neighbors = col[row == node]
    return neighbors


# early stopping method
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, path='best_model.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_check_point(val_loss, model)
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_check_point(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            
    def save_check_point(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        print(f'Validation loss ({val_loss:.6f}) decreased. Saving model to {self.path}')


# gene model clustering
def gene_cluster(emb):
    pass


# load models
def load_model(model, path):
    try:
        model.load_state_dict(torch.load(path))
    except:
        raise(f'{path} not find!')

