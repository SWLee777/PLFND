import os
import argparse
import torch
import numpy as np
import random
from run import Run
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='FastPLFND')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--max_len', type=int, default=170)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--early_stop', type=int, default=3)
parser.add_argument('--vocab_file', default='pretrained_model/minibrt-h288/vocab.txt')
parser.add_argument('--root_path', default='./data/weibo21/')
parser.add_argument('--bert', default='pretrained_model/minibrt-h288')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--seed', type=int, default=3047)
parser.add_argument('--gpu', default='0')
parser.add_argument('--emb_dim', type=int, default=288)
parser.add_argument('--lr', type=float, default=0.0004)
parser.add_argument('--save_param_dir', default='./param_model')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

config = {
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

if __name__ == '__main__':
    start_time = time.time()

    run_instance = Run(config=config)
    results = run_instance.main()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Model run time: {elapsed_time:.2f} seconds")
