import torch
import numpy as np
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

    def sample(self, batch_size):
        batch_size = min(batch_size, self._top)
        indices = np.random.randint(0, self._top, batch_size)
        obs = self.obs[indices]
        action = self.actions[indices]
        reward = self.rewards[indices]

        return obs, action, reward


class Policy(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, input_dim * 3)

        self.input_dim = input_dim

    def forward(self, obs, expl=False, eps=0.2):
        """
        Return ([batch x feature x 3]): action probability (decrease, stay, increase) per feature 
        """
        h = torch.relu(self.fc1(obs))
        h = torch.relu(self.fc2(h))
        h = self.fc3(h)
        output = h.reshape(obs.shape[0], self.input_dim, 3)
        prob = torch.softmax(output, -1)  # batch x feature x 3

        if expl:
            prob = prob + eps
            prob = prob / prob.sum(-1)

        return prob


class RLOptim():
    def __init__(self, env, latency_th, mem_th, expl_interval, device='cuda'):
        self.env = env
        self.device = device
        self.latency_th = latency_th
        self.mem_th = mem_th
        self.expl_intervals = expl_interval

    def obs_contraction(self, obs_init):
        """Reduce obs_init towards a valid solution
        """
        contraction = torch.tensor([-0.1, -0.1, 0.], device=self.device)

        obs_cur = obs_init
        mem, latency = self.env.eval_arg(obs_cur, eval_acc=False)
        while (mem > self.mem_th) or (latency > self.latency_th):
            obs_cur += contraction
            mem, latency = self.env.eval_arg(obs_cur, eval_acc=False)

        _, acc_cur, _ = self.env.eval_arg(obs_cur, eval_acc=True)
        print('\n', "=" * 50)
        print("After contraction current obs: ", obs_cur.cpu().numpy(), f"(acc: {acc_cur:.2f})")

        return obs_cur, acc_cur

    def search(self,
               obs_init,
               expl_step,
               update_step,
               reward_scale=1.,
               expl_eps=0.3,
               expl_ths=500,
               epoch=500,
               batch_size=256,
               lr=1e-3):

        policy = Policy(3, 128).to(self.device)
        optimizer = optim.SGD(policy.parameters(), lr=lr)
        # optimizer = optim.Adam(policy.parameters(), lr=lr)
        replay_buffer = Buffer(3, device=self.device)

        print("Obs init: ", obs_init.cpu().numpy())
        obs_cur, acc_cur = self.obs_contraction(obs_init)

        best_val = acc_cur
        best_config = torch.clone(obs_cur)
        for i in range(epoch):
            # Explore
            for _ in range(expl_step):
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

                invalid_flag = (obs_next < 0.1).float().sum()

                # Evaluate and add to buffer
                if not replay_buffer.check_exist(obs_cur, action):
                    mem, acc, latency = self.env.eval_arg(obs_next, eval_acc=True)
                    if (mem > self.mem_th) or (latency > self.latency_th) or (invalid_flag > 0):
                        reward = torch.tensor(-1., device=self.device)
                        obs_next = obs_cur
                        acc = acc_cur
                    else:
                        reward = torch.tensor((acc - acc_cur) * reward_scale, device=self.device)
                        if acc > best_val:
                            best_val = max(acc, best_val)
                            best_config = torch.clone(obs_cur)

                        replay_buffer.add_sample(obs_cur, action, reward)
                else:
                    mem, latency = self.env.eval_arg(obs_next, eval_acc=False)
                    if (mem > self.mem_th) or (latency > self.latency_th):
                        obs_next = obs_cur
                        acc = acc_cur

                obs_cur = obs_next
                acc_cur = acc

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

            print('\n', "=" * 50)
            print(f"[Search Epoch {i+1}] best acc: {best_val:.2f} with {best_config.cpu().numpy()}",
                  end='')
            print(f", Buffer size: {replay_buffer._top}, current: {obs_cur.cpu().numpy()}")