import torch
from algorithm import search


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
        return mem

    def eval_latency(self, inp_arg):
        latency = torch.max(inp_arg[:2])
        return latency

    def eval_acc(self, inp_arg):
        acc = torch.sum(inp_arg[:2])
        return acc


if __name__ == '__main__':
    expl_step = 20
    update_step = 1000
    toy_env = ToyEnv()

    obs_init = torch.tensor([2.0, 2.0, 1.0], device='cuda')
    search(toy_env, obs_init, expl_step, update_step)
