import torch
import gym
import numpy as np
from IPython.display import clear_output
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
F32Tensor = torch.FloatTensor
I64Tensor = torch.LongTensor
class Model(torch.nn.Module):
    def __init__(self, env, layer_size=256):
        super(Model, self).__init__()
        inputs = np.prod(env.observation_space.shape)
        self.layer_size = layer_size
        self.features = torch.nn.Sequential(
            torch.nn.Linear(inputs, layer_size),
            torch.nn.Sigmoid(),
            torch.nn.Linear(layer_size, layer_size),
            torch.nn.Tanh()
        )
        self.rnn_layer = nn.GRU(layer_size, 32, 2)
        self.hidden_fc_layer = nn.Linear(32, 31)
        self.hidden_state = torch.zeros(2, 1, 32)
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(31, env.action_space.n),
            torch.nn.Softmax(dim=-1)
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(31, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x, self.hidden_state = self.rnn_layer(x.view(1, -1, self.layer_size), self.hidden_state.data)
        x = F.relu(self.hidden_fc_layer(x.squeeze()))
        value = self.critic(x)
        actions = self.actor(x)
        return value, actions

    def compute_critic_value(self, x):
        x = self.features(x)
        x, self.hidden_state = self.rnn_layer(x.view(1, -1, self.layer_size), self.hidden_state.data)
        x = F.relu(self.hidden_fc_layer(x.squeeze()))
        value = self.critic(x)
        return value
    def compute_action_probs(self, state, action):
        action_value, features = self.forward(state)
        distribution = torch.distributions.Categorical(features)
        action_log_probs = distribution.log_prob(action).view(-1, 1)
        action_entropy = distribution.entropy().mean()
        return action_value, action_log_probs, action_entropy

    def choose_action(self, state):
        value, actor_features = self.forward(state)
        dist = torch.distributions.Categorical(actor_features)
        chosen_action = dist.sample()
        return chosen_action.item()


class Memory(object):
    def __init__(self):
        self.states, self.actions, self.true_values = [], [], []

    def push(self, state, action, true_value):
        self.states.append(state)
        self.actions.append(action)
        self.true_values.append(true_value)

    def pop_all(self):
        states = torch.stack(self.states)
        actions = I64Tensor(self.actions)
        true_values = F32Tensor(self.true_values).unsqueeze(1)
        self.states, self.actions, self.true_values = [], [], []
        return states, actions, true_values

def compute_true_values(model, states, rewards, dones, gamma=0.99):
    R = []
    rewards = F32Tensor(rewards)
    dones = F32Tensor(dones)
    states = torch.stack(states)

    if dones[-1] == True:
        next_value = rewards[-1]
    else:
        next_value = model.compute_critic_value(states[-1].unsqueeze(0))

    R.append(next_value)
    for i in reversed(range(0, len(rewards) - 1)):
        if not dones[i]:
            next_value = rewards[i] + next_value * gamma
        else:
            next_value = rewards[i]
        R.append(next_value)

    R.reverse()

    return F32Tensor(R)

def reflect(model, optimizer, memory, entropy_coef=0.01, critic_coef=0.5):
    states, actions, true_values = memory.pop_all()
    values, log_probs, entropy = model.compute_action_probs(states, actions)
    td_error = true_values - values
    critic_loss = (td_error ** 2).mean()
    actor_loss = -(log_probs * td_error.detach()).sum() / len(states)
    total_loss = critic_coef * critic_loss + actor_loss - entropy_coef * entropy
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), 0.5) # using clip_grad_value_ instead of clip_grad_norm_
    optimizer.step()


def plot(data, frame_idx):
    print(data)
    clear_output(True)
    plt.figure(figsize=(20, 5))
    if data['episode_rewards']:
        ax = plt.subplot(121)
        ax = plt.gca()
        average_score = np.mean(data['episode_rewards'][-100:])
        plt.title(f"Frame: {frame_idx} - Average Score: {average_score}")
        plt.grid()
        plt.plot(data['episode_rewards'])
    if data['values']:
        ax = plt.subplot(122)
        average_value = np.mean(data['values'][-1000:])
        plt.title(f"Frame: {frame_idx} - Average Values: {average_value}")
        plt.plot(data['values'])
    plt.savefig("./a2c_plots")


def train_model(env=None, layer_size=156, learning_rate=0.001, gamma=0.80, critic_coef=0.5, entropy_coef=0.01, c=100):
    data = {
        'episode_rewards': [],
        'values': []
    }
    model = Model(env, layer_size=layer_size)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, eps=1e-5)
    memory = Memory()
    frame_count = 0
    state, _ = env.reset()
    state = torch.from_numpy(state).float()
    episode_reward = 0
    episodes = 0
    while True:
        action = model.choose_action(state.unsqueeze(0))
        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward

        true_value = compute_true_values(model, [state], [reward], [done], gamma=gamma).unsqueeze(1)
        memory.push(state, action, true_value[0])
        frame_count += 1
        if done:
            state, _ = env.reset()
            state = torch.from_numpy(state).float()
            data['episode_rewards'].append(episode_reward)
            print(episode_reward)
            episode_reward = 0
            episodes += 1
        else:
            state = F32Tensor(next_state)

        value = reflect(model, optimizer, memory, critic_coef=critic_coef, entropy_coef=entropy_coef)
        if frame_count % 200== 0:
            data['values'].append(value)
            print(frame_count)
        if episodes >= c:
            break

    plot(data, frame_count)
    return model

def test_model(env, state_dict):
    testing_rewards = []
    with torch.no_grad():
        model = Model(env)  # Initialize a new model
        model.load_state_dict(state_dict)  # Load the state dict
        model.eval()  # Set the model to evaluation mode
        state, _ = env.reset()
        total_reward =0
        done = False
        while not done:
            state = torch.from_numpy(state).float().unsqueeze(0)  # Convert state to tensor
            with torch.no_grad():  # Don't compute gradients
                action = model.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)  # Take action in the environment
            total_reward += reward
            print(reward)
            state = next_state
            env.render()  # Render the screen
    print(f"Test episode reward: {total_reward}")
