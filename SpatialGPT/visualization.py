import seaborn
import matplotlib.pyplot as plt
import warnings
import networkx as nx


# Loss curve
def plot_loss_curve(args, losses):
    if args.visual:
        x = range(1, len(losses)+1)
        plt.plot(x, losses)
        plt.xticks(range(1, len(losses) + 1, 1))
        plt.xlabel('Epoch') 
        plt.ylabel('Running Loss') 
        # plt.title('Loss Curve') 
        plt.show()


# gene module clustering map
def plot_gene_module(args, emb):
    pass
