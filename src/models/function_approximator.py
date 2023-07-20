import numpy as np

import torch
import torch.nn as nn



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
            nn.Softmax()
        )

    def forward(self, x):
        actions = self.network_stack(x)
        return actions