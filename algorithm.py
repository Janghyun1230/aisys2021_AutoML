import torch
import numpy as np
import torch.optim as optim
from torch.distributions.categorical import Categorical
np.set_printoptions(precision=3)


class Buffer():
    # obs (inp_arg) action (up or down) reward (acc diff + indicator)
    def __init__(self, state_dim, action_dim, size=1000, device='cuda'):
        self.obs = torch.zeros([size, state_dim], device=device)
        self.actions = torch.zeros([size, action_dim], device=device, dtype=torch.long)
        self.rewards = torch.zeros([size], device=device)

        self._size = size
        self._top = 0
        self._cur = 0

    def add_sample(self, obs, action, reward):
        self.obs[self._cur] = obs
        self.actions[self._cur] = action
        self.rewards[self._cur] = reward

        self._top = min(self._top + 1, self._size)
        self._cur = (self._cur + 1) % self._size

    def sample(self, batch_size):
        indices = np.random.randint(0, self._top, batch_size)
        obs = self.obs[indices]
        action = self.actions[indices]
        reward = self.rewards[indices]

        return obs, action, reward


class Policy(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim * 3)

        self.action_dim = action_dim

    def forward(self, obs):
        """
        Returns: per feature action probability (decrease, stay, increase)
        """
        h = self.fc1(obs)
        h = self.relu(h)
        h = self.fc2(h)
        output = h.reshape(-1, self.action_dim, 3)
        prob = torch.softmax(output, -1)  # batch x feature x 3
        return prob


def search(env, obs_init, expl_step, update_step, epoch=100, batch_size=16, lr=1e-2, device='cuda'):
    print("Obs init: ", obs_init.cpu().numpy())
    intervals = [0.01, 0.01, 0.]
    latency_th = 5.0
    mem_th = 10

    policy = Policy(3, 256, 3).to(device)
    replay_buffer = Buffer(3, 3, device=device)
    optimizer = optim.SGD(policy.parameters(), lr=lr, momentum=0.9)
    contraction = torch.tensor([-0.1, -0.1, 0.], device=device)

    # Reduce towards a valid solution
    obs_cur = obs_init
    mem, latency = env.eval_arg(obs_cur, eval_acc=False)
    while (mem > mem_th) or (latency > latency_th):
        obs_cur += contraction
        mem, latency = env.eval_arg(obs_cur, eval_acc=False)

    _, acc_cur, _ = env.eval_arg(obs_cur, eval_acc=True)
    print("After contraction current obs: ", obs_cur.cpu().numpy(), f"(acc: {acc_cur})")

    acc_best = acc_cur
    best_config = torch.clone(obs_cur)
    for i in range(epoch):
        # Explore
        with torch.no_grad():
            for _ in range(expl_step):
                # Move
                prob = policy(obs_cur.unsqueeze(0))[0]
                obs_next = torch.clone(obs_cur)
                action = torch.zeros([3], device=device, dtype=torch.long)

                for k, p in enumerate(prob):
                    action_feature = Categorical(p).sample()
                    action[k] = action_feature
                    if action_feature == 0:
                        obs_next[k] -= intervals[k]
                    elif action_feature == 1:
                        pass
                    else:
                        obs_next[k] += intervals[k]
                # Eval
                mem, acc, latency = env.eval_arg(obs_next, eval_acc=True)
                if (mem > mem_th) or (latency > latency_th):
                    reward = torch.tensor(-100., device=device)
                    obs_next = obs_cur
                    acc = acc_cur
                else:
                    reward = (acc - acc_cur)
                    if acc > acc_best:
                        acc_best = torch.max(acc, acc_best)
                        best_config = torch.clone(obs_cur)

                replay_buffer.add_sample(obs_cur, action, reward)
                obs_cur = obs_next
                acc_cur = acc

                print(obs_cur)

        # Update policy
        for _ in range(update_step):
            # action: {0,1,2} ^ (batch x feature)
            obs, actions, rewards = replay_buffer.sample(batch_size)
            prob = policy(obs)

            prob_gather = torch.gather(prob, -1, actions.unsqueeze(-1)).squeeze()  # batch x feature
            log_prob_gather = torch.log(prob_gather + 1e-6).sum(-1)  # batch

            loss = -(rewards * log_prob_gather).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Epoch {i}] best acc: {acc_best:.2f} with {best_config.cpu().numpy()}")