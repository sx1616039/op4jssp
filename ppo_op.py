import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from simple_op_jssp import JobEnv


class Actor(nn.Module):
    def __init__(self, num_input, num_output, unit_num=100):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_input, unit_num)
        self.action_head = nn.Linear(unit_num, num_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = self.action_head(x)
        return action_prob


class Critic(nn.Module):
    def __init__(self, num_input, unit_num=100, num_output=1):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_input, unit_num)
        self.state_value = nn.Linear(unit_num, num_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO:
    def __init__(self, j_env, unit_num=100, memory_size=5, batch_size=32, clip_ep=0.2):
        super(PPO, self).__init__()
        self.env = j_env
        self.memory_size = memory_size
        self.batch_size = batch_size  # update batch size
        self.epsilon = clip_ep

        self.state_dim = self.env.state_num
        self.action_dim = self.env.action_num
        self.case_name = self.env.case_name
        self.gamma = 1  # reward discount
        self.A_LR = 0.0001  # learning rate for actor
        self.C_LR = 0.0002  # learning rate for critic
        self.A_UPDATE_STEPS = 10  # actor update steps
        self.C_UPDATE_STEPS = 10  # critic update steps
        self.max_grad_norm = 0.5
        self.training_step = 0

        self.actor_net = Actor(self.state_dim, self.action_dim, unit_num)
        self.critic_net = Critic(self.state_dim, unit_num)
        self.actor_optimizer = optimizer.Adam(self.actor_net.parameters(), self.A_LR)
        self.critic_net_optimizer = optimizer.Adam(self.critic_net.parameters(), self.C_LR)
        if not os.path.exists('param'):
            os.makedirs('param/net_param')

    def select_action(self, state, action_mask):
        state = torch.from_numpy(state).float().unsqueeze(0)
        mask = torch.from_numpy(action_mask).float().unsqueeze(0)
        with torch.no_grad():
            prob = self.actor_net(state)
            prob += mask
            action_prob = F.softmax(prob, dim=1)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_params(self):
        torch.save(self.actor_net.state_dict(), 'param/net_param/' + self.env.case_name + 'actor_net.model')
        torch.save(self.critic_net.state_dict(), 'param/net_param/' + self.env.case_name + 'critic_net.model')

    def load_params(self):
        self.critic_net.load_state_dict(torch.load('param/net_param/' + self.env.case_name + 'critic_net.model'))
        self.actor_net.load_state_dict(torch.load('param/net_param/' + self.env.case_name + 'actor_net.model'))

    def update(self, bs, ba, br, bp, bm):
        # get old actor log prob
        old_action_log_prob = torch.tensor(bp, dtype=torch.float).view(-1, 1)
        state = torch.tensor(np.array(bs), dtype=torch.float)
        action = torch.tensor(ba, dtype=torch.long).view(-1, 1)
        d_reward = torch.tensor(br, dtype=torch.float)
        mask = torch.tensor(np.array(bm), dtype=torch.float)

        for i in range(self.A_UPDATE_STEPS):
            for index in BatchSampler(SubsetRandomSampler(range(len(ba))), self.batch_size, False):
                #  compute the advantage
                d_reward_index = d_reward[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = d_reward_index - V
                advantage = delta.detach()

                # epoch iteration, PPO core!
                prob = self.actor_net(state[index])  # new policy
                prob += mask[index]
                action_prob = F.softmax(prob, dim=1).gather(1, action[index])
                ratio = (action_prob / old_action_log_prob[index])
                surrogate = ratio * advantage
                clip_loss = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
                action_loss = -torch.min(surrogate, clip_loss).mean()

                # update actor network
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(d_reward_index, V)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

    def train(self):
        column = ["episode", "make_span", "reward", "no_op_cnt"]
        results = pd.DataFrame(columns=column, dtype=float)
        index = 0
        converged = 0
        converged_value = []
        t0 = time.time()
        for i_epoch in range(4000):
            if time.time() - t0 > 3600:  # 3600 seconds
                break
            bs, ba, br, bp, bm = [], [], [], [], []
            for m in range(self.memory_size):  # memory size is the number of complete episode
                buffer_s, buffer_a, buffer_r, buffer_p, buffer_m = [], [], [], [], []
                state = self.env.reset()
                episode_reward = 0
                while True:
                    mask = np.repeat(-1e+8, self.action_dim)
                    for k in range(self.action_dim):
                        if state[k] == 1:
                            mask[k] = 0
                    buffer_m.append(mask)
                    action, action_prob = self.select_action(state, mask)
                    next_state, reward, done = self.env.step(action)
                    buffer_s.append(state)
                    buffer_a.append(action)
                    buffer_r.append(reward)
                    buffer_p.append(action_prob)

                    state = next_state
                    episode_reward += reward
                    if done:
                        v_s_ = 0
                        discounted_r = []
                        for r in buffer_r[::-1]:
                            v_s_ = r + self.gamma * v_s_
                            discounted_r.append(v_s_)
                        discounted_r.reverse()

                        bs[len(bs):len(bs)] = buffer_s
                        ba[len(ba):len(ba)] = buffer_a
                        br[len(br):len(br)] = discounted_r
                        bp[len(bp):len(bp)] = buffer_p
                        bm[len(bm):len(bm)] = buffer_m
                        # Episode: make_span: Episode reward: no-op count
                        print('{}    {}    {:.2f}  {}'.format(i_epoch, self.env.current_time,
                                                              episode_reward, self.env.no_op_cnt))
                        index = i_epoch * self.memory_size + m
                        results.loc[index] = [i_epoch, self.env.current_time, episode_reward, self.env.no_op_cnt]
                        converged_value.append(self.env.current_time)
                        if len(converged_value) >= 31:
                            converged_value.pop(0)
                        break
            self.update(bs, ba, br, bp, bm)
            converged = index
            if min(converged_value) == max(converged_value) and len(converged_value) >= 30:
                break
        if not os.path.exists('results'):
            os.makedirs('results')
        results.to_csv("results/" + str(self.env.case_name) + "_data.csv")
        self.save_params()
        return min(converged_value), converged, time.time()-t0


if __name__ == '__main__':
    name = "op_13102"
    param = [name, "converged_iterations", "total_time"]
    path = "data_set_sizes/"
    simple_results = pd.DataFrame(columns=param, dtype=int)
    for file_name in os.listdir(path):
        title = file_name.split('.')[0]
        env = JobEnv(title, path)
        scale = env.job_num * env.machine_num
        model = PPO(env, unit_num=env.state_num, memory_size=3, batch_size=scale, clip_ep=0.2)
        simple_results.loc[title] = model.train()
    simple_results.to_csv(name + ".csv")

    # env = JobEnv("ta51")
    # scale = env.job_num * env.machine_num
    # model = PPO(env, unit_num=env.state_num, memory_size=3, batch_size=scale, clip_ep=0.2)
    # model.train()
