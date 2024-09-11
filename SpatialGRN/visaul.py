import seaborn
import matplotlib.pyplot as plt

# Loss curve
def plot_loss_curve(args, losses):
    if args.visualize:
        x = range(1, len(losses)+1)
        plt.plot(x, losses)
        plt.show()


# gene module clustering map
def plot_gene_module(args, emb):
    pass
