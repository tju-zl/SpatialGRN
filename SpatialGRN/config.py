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
    parser.add_argument('--srt_resolution', default=150, type=int)
    parser.add_argument('--clusters', default=0, type=int)

    # model parameters
    parser.add_argument('--infer_mode', type=str, default='simplex', help='resnet or simplex')
    parser.add_argument('--hops', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=10, help='must >= 8')
    parser.add_argument('--latent_dim', type=int, default=50)
    parser.add_argument('--spread', type=float, default=3e-3)
    parser.add_argument('--flow', type=str, default='source_to_target')

    # training control
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--wegiht_decay', type=float, default=1e-6)
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--tau', type=float, default=1)

    # output configuration
    parser.add_argument('--log_file', type=str, default='./Log')
    parser.add_argument('--out_file', type=str, default='./Output')

    # analysis configuration
    parser.add_argument('--visualize', default=True, action='store_true')

    return parser
