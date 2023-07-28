import numpy as np
import math

import gymnasium as gym

import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import itertools
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class DQN(nn.Module):

    def __init__(self, env, LAYER_SIZE):
        super().__init__()

        inputs = np.prod(env.observation_space.shape)
        self.flatten = nn.Flatten()
        self.network_stack = nn.Sequential(
            nn.Linear(inputs, LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(LAYER_SIZE, LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(LAYER_SIZE, env.action_space.n),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x = self.flatten(x)
        x = x.to(torch.float32)
        actions = self.network_stack(x)
        return actions

class Agent:

    def __init__(self,
                 env: gym.Env,
                 layer_size: int = 64,
                 min_buffer_size:int=200,
                 N:int=100000,
                 M:int=1000,
                 batch_size:int=100,
                 epsilon:int=1,
                 epsilon_decay:float=1 - 1e-4,
                 min_epsilon:float=.01,
                 alpha:float=5e-4,
                 C:int=5,
                 gamma:float=.99):

        self.env = env
        self.test_env = env

        self.online_net = DQN(env, layer_size)
        self.target_net = DQN(env, layer_size)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.layer_size = layer_size

        self.min_buffer_size = min_buffer_size
        # Max Buffer Size
        self.N = N

        # Max number of episodes
        self.M = M

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.batch_size = batch_size

        # Set Learning Rate
        self.alpha = alpha

        # Set the target network frequency update
        self.C = C

        self.gamma = gamma

        # Replay Buffer D
        # Rewards for last 50 episodes to track progress
        # Rewards for all episodes
        self.D = self._initialize_replay_buffer()
        self.r_scores = deque(maxlen=50)
        self.epsiode_rewards = []
        self.testing_rewards = []


        # Huber loss
        self.criterion = nn.functional.smooth_l1_loss

        # Adam Optimizer
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=alpha)

        self.model = None

    def _initialize_replay_buffer(self):

        s, _ = self.env.reset()
        D = deque(maxlen=self.N)

        for _ in range(self.min_buffer_size):

            a = self.env.action_space.sample()
            s_prime, r, done, _, _ =self.env.step(a)
            #             s_prime, r, done, _, _ = self.env.step(a)
            e = (s, a, r, s_prime, done)
            D.append(e)
            s = s_prime

            if done:
                s = self.env.reset()
        return D

    def get_best_action(self, s):
        if self.model is None:
            self.model = self.online_net
        return torch.argmax(self.model(torch.tensor(s).unsqueeze(0))).item()

    def train(self):

        step = 0

        for ep in range(self.M):

            episode_r = 0
            s, _ = self.env.reset()

            while True:

                if np.random.uniform(0, 1) < self.epsilon:
                    a = self.env.action_space.sample()
                else:
                    a = self.get_best_action(s)

                # Changed for gymnasium implementation
                s_prime, r, done, _, _ = self.env.step(a)
                # s_prime, r, done, _ = self.env.step(a)
                episode_r += r
                e = [s, a, r, s_prime, done]
                self.D.append(e)
                s = s_prime

                if done:
                    break

                batch = random.sample(self.D, self.batch_size)

                s_tensor = torch.tensor(np.array([b[0] for b in batch]))
                a_tensor = torch.tensor(np.array([b[1] for b in batch]))
                r_tensor = torch.tensor(np.array([b[2] for b in batch]))
                s_prime_tensor = torch.tensor(np.array([b[3] for b in batch]))
                done_tensor = torch.tensor(np.array([b[4] for b in batch]),
                                           dtype=torch.int64)  # Cast as integer for piecewise target calculation

                t = self.target_net(s_prime_tensor)
                t_tensor = torch.max(t, axis=1).values.unsqueeze(-1)

                targets = r_tensor.unsqueeze(-1) + self.gamma * (1 - done_tensor.unsqueeze(-1)) * t_tensor

                Q = self.online_net(s_tensor)
                Q_actions = torch.gather(Q, axis=1, index=a_tensor.unsqueeze(-1))

                loss = self.criterion(Q_actions, targets)

                # Run the model update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Decay Epsilon
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                step += 1

                if step % self.C == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

            self.r_scores.append(episode_r)
            self.epsiode_rewards.append(episode_r)

            if ep % 10 == 0:
                print("-" * 50)
                print()
                print(f"Episode: {ep}")
                print(f"Reward: {np.mean(self.r_scores)}")
                print(f"Epsilon: {self.epsilon}")
                print("-" * 50)

            # if np.mean(self.r_scores) >= 220:

        self.model = self.online_net
        try:
            torch.save(self.online_net.state_dict(), './trained_models/DQN.pt')
            print("model state dict saved sucessfully")
        except:
            print("problem saving model")

    def test(self):
        print("testing model...")
        if self.model:
            with torch.no_grad():
                self.model.eval()
                s, _ = self.test_env.reset()
                episode_r = 0

                while True:
                    a = self.get_best_action(s)
                    s_prime, r, done, _, _ = self.test_env.step(a)
                    #                 s_prime, r, done, _,_ = self.test_env.step(a)
                    # self.test_env.render()
                    episode_r += r
                    s = s_prime

                    if done:
                        print(episode_r)
                        self.testing_rewards.append(episode_r)
                        break

        else:

            model_state_dict= torch.load('./trained_models/DQN.pt')
            self.model = DQN(self.env, self.layer_size)
            self.model.load_state_dict(model_state_dict)

            with torch.no_grad():
                self.model.eval()
                s, _ = self.test_env.reset()
                episode_r = 0

                while True:
                    a = self.get_best_action(s)
                    s_prime, r, done, _, _ = self.test_env.step(a)
                    #                 s_prime, r, done, _,_ = self.test_env.step(a)
                    # self.test_env.render()
                    episode_r += r
                    s = s_prime

                    if done:
                        print(episode_r)
                        self.testing_rewards.append(episode_r)
                        break
