import scanpy as sc
import numpy as np
import torch
from torch_geometric.nn.pool import radius_graph


def prepare_dataset(args, adata):
    adata.var_names_make_unique()
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_valu=10)
    
    return adata


def compute_edge(args, adata):
    coordinate = torch.FloatTensor(adata.obsm['spatial'])
    edge_index = radius_graph(coordinate, args.srt_resolution, flow=args.flow)
    
    return edge_index
