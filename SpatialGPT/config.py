import argparse

def config():
    parser = argparse.ArgumentParser('Configuration File of SpatialGRN')
    
    # system configuration
    parser.add_argument('--project_name', type=str, default='SpatialGRN')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--version', type=str, default='v_1.0')

    # data preparation
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--srt_resolution', default=2, type=int)
    parser.add_argument('--clusters', default=0, type=int)
    parser.add_argument('--max_neighbors', default=6, type=int)
    parser.add_argument('--hvgs', default=2000, type=int)
    parser.add_argument('--n_hops', default=3, type=int)
    parser.add_argument('--n_randomwalk', default=[5, 10, 15], type=list, help='sample path distance of random walk')
    parser.add_argument('--q_randomwalk', default=[1, 1.5], type=list, help='in-out of random walk')

    # model parameters
    parser.add_argument('--decoder', type=str, default='NB', help='ZINB, NB or MLP')
    parser.add_argument('--token_dim', type=int, default=64, help='dims of gene token')
    parser.add_argument('--latent_dim', type=int, default=64, help='dim of QKV')
    parser.add_argument('--flow', type=str, default='source_to_target')

    # Transformer Module parameters
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=8)


    # training control
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--early_stop', default=True, action='store_true')
    parser.add_argument('--wegiht_decay', type=float, default=1e-6)
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--eval', default=False, action='store_true')

    # output configuration
    parser.add_argument('--log_file', type=str, default='../Log')
    parser.add_argument('--out_file', type=str, default='../Output')

    # analysis configuration
    parser.add_argument('--visual', default=True, action='store_true')

    return parser
