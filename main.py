import torch
from algorithm import RLOptim
from env import NasEnv
import argparse

parser = argparse.ArgumentParser()
'''Environment'''
parser.add_argument('-m', "--model", type=str, default='preactresnet18')
parser.add_argument('-w', "--width", type=float, default=1.0)
parser.add_argument('-d', "--depth", type=float, default=1.0)
parser.add_argument('-r', "--resolution", type=int, default=32)
parser.add_argument('-e', "--epoch", type=int, default=30)
args = parser.parse_args()

toy_env = NasEnv(modelName='preactresnet18',
                 platform='desktop',
                 device='cuda',
                 batch_size=64,
                 lr=1e-2,
                 n_epoch=args.epoch)
rl_optim = RLOptim(toy_env, latency_th=1.0, mem_th=10.0, expl_interval=(0.1, 0.1, 1))

obs_init = torch.tensor([args.width, args.depth, args.resolution], device='cuda')
rl_optim.search(obs_init, expl_step=10, update_step=100)
