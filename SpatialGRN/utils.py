import os
import random
import torch
import numpy as np
from sklearn.cluster import KMeans
# import ot

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
    
    
