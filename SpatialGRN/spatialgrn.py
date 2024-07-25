from tqdm.notebook import trange
import matplotlib.pyplot as plt
from scipy.sparse import issparse


class SpatailGRN:
    def __init__(self, args, adata):
        self.args = args
        