import torch
from algorithm import RLOptim
from env import NasEnv
import argparse
import os

parser = argparse.ArgumentParser()
'''Environment'''
parser.add_argument('-m', "--model", type=str, default='preactresnet18')
parser.add_argument('-p', "--platform", type=str, default='desktop')
parser.add_argument('-v', "--device", type=str, default='cuda')
parser.add_argument('-w', "--width", type=float, default=1.0)
parser.add_argument('-d', "--depth", type=float, default=1.0)
parser.add_argument('-r', "--resolution", type=int, default=32)
#
parser.add_argument('-e', "--epoch", type=int, default=30)
parser.add_argument("--latency", type=float, default=5, help='latency bound (s)')
parser.add_argument("--mem", type=float, default=100, help='memory bound (MB)')
parser.add_argument("--id", type=int, default=0)
args = parser.parse_args()

if torch.cuda.is_available():
    run_on = "cuda"
else:
    run_on = "cpu"

env = NasEnv(modelName=args.model, platform=args.platform, device=args.device, n_epoch=args.epoch)

path = os.path.join('./buffer', f'{args.platform}_{args.device}_l{args.latency}_m{args.mem}')
os.makedirs(path, exist_ok=True)

rl_optim = RLOptim(env,
                   latency_th=args.latency,
                   mem_th=args.mem,
                   expl_interval=(0.1, 0.1, 1),
                   path=path,
                   idx=args.id)
obs_init = torch.tensor([args.width, args.depth, args.resolution], device=run_on)
rl_optim.search(obs_init, expl_step=10, update_step=100)
