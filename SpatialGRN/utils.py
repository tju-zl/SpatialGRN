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


def get_log_dir(args):
    log_dir = '../Log/' + '_'.join([os.path.basename(args.dataset_path).split('.')[0], args.version])
    return log_dir


def get_output_dir(args):
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
    
    
def random_walk_path(args, edge_index):
    edge_list = []
    data = Data(edge_index=edge_index, num_nodes=args.n_spots)
    G = to_networkx(data)
    for length in args.n_randomwalk:
        edge_list.append(Node2Vec(G, p=1, q=args.q[0], walk_length=length, num_walks=1, workers=4).walks)
        edge_list.append(Node2Vec(G, p=1, q=args.q[1], walk_length=length, num_walks=1, workers=4).walks)
    
    walks_int = []
    for edge in edge_list:
        walks_int.append([int(node) for node in walk] for walk in edge)
    
    edge_index_list = []
    for k, edge in enumerate(walks_int):
        edges = [[],[]]
        for i in range(len(edges)):
            for j in range(args.n_randomwalk[k]):
                edges[0].append(edges[i][j])
                edges[1].append(i)
        edge_index_list.append(torch.tensor(np.array(edge), dtype=torch.long))
    return edge_index_list