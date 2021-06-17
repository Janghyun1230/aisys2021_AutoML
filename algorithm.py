import torch
import numpy as np
import os
import glob
import torch.optim as optim
from torch.distributions.categorical import Categorical
np.set_printoptions(precision=3)


class Buffer():
    def __init__(self, state_dim, size=10000, device='cuda'):
        """
        obs ([state_dim]): inp_arg
        action ([state_dim]): action for each arg (down 0, stay 1, up 2)
        reward ([1]): acc difference + indicator (latency, memory)
        """
        self.obs = torch.zeros([size, state_dim], device=device)
        self.actions = torch.zeros([size, state_dim], device=device, dtype=torch.long)
        self.rewards = torch.zeros([size], device=device)

        self._size = size
        self._top = 0
        self._cur = 0
        self.total_added = 0

    def check_exist(self, obs, action):
        """Check obs action pair is existing 
        """
        duplicate = False
        obs_diff = (self.obs[:self._top] - obs.reshape(1, -1)).abs().sum(-1)
        action_diff = (self.actions[:self._top] - action.reshape(1, -1)).abs().sum(-1)

        dup = ((obs_diff + action_diff) == 0).float().sum()
        if dup > 0:
            duplicate = True

        return duplicate

    def add_sample(self, obs, action, reward):
        self.obs[self._cur] = obs
        self.actions[self._cur] = action
        self.rewards[self._cur] = reward

        self._top = min(self._top + 1, self._size)
        self._cur = (self._cur + 1) % self._size

        self.total_added += 1

    def sample(self, batch_size):
        batch_size = min(batch_size, self._top)
        indices = np.random.randint(0, self._top, batch_size)
        obs = self.obs[indices]
        action = self.actions[indices]
        reward = self.rewards[indices]

        return obs, action, reward

    def save(self, path, id):
        data_dict = {'obs': self.obs, 'actions': self.actions, 'rewards': self.rewards}
        torch.save(data_dict, os.path.join(path, f'buffer{id}.pt'))
        print("Buffer saved! ", os.path.join(path, f'buffer{id}.pt'))

    def load(self, path):
        file_list = glob.glob(os.path.join(path, 'buffer*.pt'))
        for file in file_list:
            data_dict = torch.load(file)
            n = data_dict['obs'].shape[0]
            for i in range(n):
                dup = self.check_exist(data_dict['obs'][i], data_dict['actions'][i])
                if not dup:
                    self.add_sample(data_dict['obs'][i], data_dict['actions'][i],
                                    data_dict['rewards'][i])
        print("Buffer is loaded from ", file_list)


class Policy(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, input_dim * 3)
        self.act = torch.nn.LeakyReLU(negative_slope=0.2)

        self.input_dim = input_dim

    def forward(self, obs, expl=False, eps=0.2):
        """
        Return ([batch x feature x 3]): action probability (decrease, stay, increase) per feature 
        """

        h = self.act(self.fc1(obs))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        output = h.reshape(obs.shape[0], self.input_dim, 3)
        prob = torch.softmax(output, -1)  # batch x feature x 3

        if expl:
            prob = prob + eps
            prob = prob / prob.sum(-1)

        return prob


class RLOptim():
    def __init__(self,
                 env,
                 latency_th,
                 mem_th,
                 expl_interval,
                 path='./buffer',
                 idx=0,
                 device='cuda'):
        self.env = env
        self.device = device
        self.latency_th = latency_th
        self.mem_th = mem_th
        self.expl_intervals = expl_interval

        self.path = path
        self.idx = idx

    def obs_contraction(self, obs_init):
        """Reduce obs_init towards a valid solution
        """
        contraction = torch.tensor([-0.1, -0.1, 0.], device=self.device)

        obs_cur = obs_init
        mem, latency = self.env.eval_arg(obs_cur, eval_acc=False)
        while (mem > self.mem_th) or (latency > self.latency_th):
            obs_cur += contraction
            mem, latency = self.env.eval_arg(obs_cur, eval_acc=False)

            if (obs_cur[0] < 0.1):
                obs_cur[0] = 0.1
                contraction[0] = 0.
            if (obs_cur[1] < 0.1):
                obs_cur[1] = 0.1
                contraction[1] = 0.
            if (obs_cur[0] == 0.1) and (obs_cur[1] == 0.1):
                contraction[2] = -1
            if (obs_cur[2] < 4):
                print("\nThere are no possible architecture satisfying constraints!")
                return obs_cur, 0

        _, acc_cur, _ = self.env.eval_arg(obs_cur, eval_acc=True)
        print('\n', "=" * 50)
        print("After contraction current obs: ", obs_cur.cpu().numpy(), f"(acc: {acc_cur:.2f})")

        return obs_cur, acc_cur

    def search(self,
               obs_init,
               expl_step,
               update_step,
               reward_scale=1.,
               expl_eps=1.0,
               expl_ths=1000,
               epoch=500,
               batch_size=256,
               lr=1e-3,
               test=False):

        policy = Policy(3, 128).to(self.device)
        # optimizer = optim.SGD(policy.parameters(), lr=lr)
        optimizer = optim.Adam(policy.parameters(), lr=lr)
        replay_buffer = Buffer(3, device=self.device)

        print("Obs init: ", obs_init.cpu().numpy())
        obs_cur, acc_cur = self.obs_contraction(obs_init)
        if obs_cur[2] < 4:
            return

        best_val = acc_cur
        best_config = torch.clone(obs_cur)
        best_acc_list = []
        cur_obs_list = []
        for i in range(epoch):
            print('\n', "=" * 50)
            # Explore
            for _ in range(expl_step):
                with torch.no_grad():
                    best_acc_list.append(best_val)
                    cur_obs_list.append(obs_cur)

                # Move on next obs
                with torch.no_grad():
                    prob = policy(obs_cur.unsqueeze(0), expl=True, eps=expl_eps)[0]
                obs_next = torch.clone(obs_cur)
                action = torch.zeros([3], device=self.device, dtype=torch.long)

                for k, p in enumerate(prob):
                    action_feature = Categorical(p).sample()
                    action[k] = action_feature
                    if action_feature == 0:
                        obs_next[k] -= self.expl_intervals[k]
                    elif action_feature == 1:
                        pass
                    else:
                        obs_next[k] += self.expl_intervals[k]

                # Chect whether the next obs is a valid argument
                if test:
                    invalid_flag = 0.
                else:
                    invalid_flag = (obs_next[:2] < 0.1).float().sum()
                    invalid_flag += (obs_next[-1] < 4).float().sum()

                # Evaluate and add to buffer
                if invalid_flag > 0:
                    reward = torch.tensor(-0.5, device=self.device)
                    obs_next = obs_cur
                    acc = acc_cur
                    if not replay_buffer.check_exist(obs_cur, action):
                        replay_buffer.add_sample(obs_cur, action, reward)

                elif not replay_buffer.check_exist(obs_cur, action):
                    mem, acc, latency = self.env.eval_arg(obs_next, eval_acc=True)
                    if (mem > self.mem_th) or (latency > self.latency_th):
                        reward = torch.tensor(-0.5, device=self.device)
                        obs_next = obs_cur
                        acc = acc_cur
                    else:
                        reward = torch.tensor((acc - acc_cur) * reward_scale, device=self.device)

                    replay_buffer.add_sample(obs_cur, action, reward)

                else:
                    mem, latency = self.env.eval_arg(obs_next, eval_acc=False)
                    if (mem > self.mem_th) or (latency > self.latency_th):
                        obs_next = obs_cur
                        acc = acc_cur

                obs_cur = obs_next
                acc_cur = acc
                if acc > best_val:
                    best_val = max(acc, best_val)
                    best_config = torch.clone(obs_cur)

            # Save buffer and synchronize
            replay_buffer.save(self.path, self.idx)
            replay_buffer.load(self.path)

            # Update policy
            for _ in range(update_step):
                # action: {0,1,2} ^ (batch x feature)
                obs, actions, rewards = replay_buffer.sample(batch_size)
                prob = policy(obs)
                prob_gather = torch.gather(prob, -1,
                                           actions.unsqueeze(-1)).squeeze()  # batch x feature
                log_prob_gather = torch.log(prob_gather + 1e-6).sum(-1)  # batch
                loss = -(rewards * log_prob_gather).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Adjust expl eps
            if replay_buffer._top > expl_ths:
                expl_eps /= 2
                expl_ths += 500
                print("Expl eps is decayed! ", expl_eps)

            print(f"[Search Epoch {i+1}] best acc: {best_val:.2f} with {best_config.cpu().numpy()}",
                  end='')
            print(f", Buffer total: {replay_buffer.total_added}, current: {obs_cur.cpu().numpy()}")

            torch.save({
                'acc': best_acc_list,
                'obs': cur_obs_list
            }, os.path.join(self.path, f'data{self.idx}.pt'))
