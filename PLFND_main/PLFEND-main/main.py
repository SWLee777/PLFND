import os
import argparse
import torch
import numpy as np
import random
from run import Run


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='plfend')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--max_len', type=int, default=340)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--early_stop', type=int, default=3)
    parser.add_argument('--vocab_file', default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt')
    parser.add_argument('--root_path', default='./data/weibo21/')
    parser.add_argument('--bert', default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--emb_dim', type=int, default=768)
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--save_param_dir', default='./param_model')
    return parser.parse_args()


def set_random_seed(seed):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def prepare_environment(gpu):
    """
    Prepare the environment by setting the CUDA visible devices.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def create_config(args):
    """
    Create a configuration dictionary from the parsed arguments.
    """
    return {
        'use_cuda': True,
        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'early_stop': args.early_stop,
        'num_workers': args.num_workers,
        'vocab_file': args.vocab_file,
        'bert': args.bert,
        'root_path': args.root_path,
        'weight_decay': 5e-5,
        'model': {
            'mlp': {'dims': [384], 'dropout': 0.2}
        },
        'emb_dim': args.emb_dim,
        'lr': args.lr,
        'epoch': args.epoch,
        'model_name': args.model_name,
        'seed': args.seed,
        'save_param_dir': args.save_param_dir
    }


def main():
    """
    Main function to run the model.
    """
    args = parse_arguments()
    prepare_environment(args.gpu)
    set_random_seed(args.seed)
    config = create_config(args)

    run_instance = Run(config=config)
    results = run_instance.main()
    return results


if __name__ == '__main__':
    main()
