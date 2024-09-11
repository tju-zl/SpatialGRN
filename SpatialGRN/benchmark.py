import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


# network dense statistic
def network_stat(args, g):
    pass


# criteria computing
def evaluation(y_true, y_pred):
    y_p = y_pred[:,-1].cpu().detach.numpy().flatten()
    y_t = y_true.cpu().numpy().flatten().astype(int)
    
    AUC = roc_auc_score(y_true=y_t, y_pred=y_p)
    APR = average_precision_score(y_true=y_t, y_pred=y_p)
    NAPR = APR/np.mean(y_t)

    return AUC, APR, NAPR


# get attention scores
def get_att(args, model, data):
    att = model(data.to(args.device))[0]
    return att


# get cell embedding
def get_cemb(args, model, data):
    z = model(data.to(args.device))[-1]
    return z
