import torch
from algorithm import RLOptim
import os
import argparse


class ToyEnv():
    def __init__(self):
        pass

    def eval_arg(self, inp_arg, eval_acc=True):
        mem = self.eval_mem(inp_arg)
        latency = self.eval_latency(inp_arg)
        if eval_acc:
            acc = self.eval_acc(inp_arg)
            return mem, acc, latency
        else:
            return mem, latency

    def eval_mem(self, inp_arg):
        mem = torch.max(inp_arg[:2])
        return mem.item()

    def eval_latency(self, inp_arg):
        # latency = torch.max(inp_arg[:2])
        latency = inp_arg[0]**2 + inp_arg[1]**2
        return latency.item()

    def eval_acc(self, inp_arg):
        # acc = torch.sum(inp_arg[:2])
        # acc = 2 * inp_arg[0] + inp_arg[1]
        dist = torch.sqrt((inp_arg[0] - 0.5)**2 + (inp_arg[1] - 0.3)**2)
        dist += (inp_arg[0] - 0.5).abs() + (inp_arg[1] - 0.3).abs()

        acc = 1 - dist / 2
        return acc.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''Environment'''
    parser.add_argument("--id", type=int, default=0)
    args = parser.parse_args()

    if torch.cuda.is_available():
        run_on = "cuda"
    else:
        run_on = "cpu"

    toy_env = ToyEnv()
    path = os.path.join('./buffer', 'test')
    os.makedirs(path, exist_ok=True)

    rl_optim = RLOptim(toy_env,
                       latency_th=1.0,
                       mem_th=10.0,
                       expl_interval=(0.01, 0.01, 0.),
                       path=path,
                       idx=args.id,
                       device=run_on)
    obs_init = torch.tensor([2.0, 2.0, 0.0], device=run_on)
    rl_optim.search(obs_init, expl_step=100, update_step=1000, reward_scale=10., test=True)
