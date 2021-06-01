import torch
from algorithm import RLOptim


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
        # latency = torch.max(inp_arg[:2])
        latency = inp_arg[0]**2 + inp_arg[1]**2
        return latency

    def eval_acc(self, inp_arg):
        # acc = torch.sum(inp_arg[:2])
        # acc = 2 * inp_arg[0] + inp_arg[1]
        acc = 2 - (inp_arg[0] - 0.5).abs() - (inp_arg[1] - 0.3).abs()
        return acc


if __name__ == '__main__':
    expl_step = 100
    update_step = 1000
    toy_env = ToyEnv()

    obs_init = torch.tensor([2.0, 2.0, 1.0], device='cuda')
    rl_optim = RLOptim(toy_env, latency_th=1.0, mem_th=10.0)

    rl_optim.search(obs_init, expl_step, update_step)
