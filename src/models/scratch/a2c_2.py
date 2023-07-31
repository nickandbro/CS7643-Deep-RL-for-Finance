import torch
import gym
import numpy as np
from IPython.display import clear_output
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt

#max_frames = 5000000
max_frames = 102480
batch_size = 5
learning_rate = 7e-4
gamma = 0.99
entropy_coef = 0.01
critic_coef = 0.5
no_of_workers = 16
if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor

data = {
    'episode_rewards': [],
    'values': []
}


class Model(torch.nn.Module):
    def __init__(self, env):
        super(Model, self).__init__()
        inputs = np.prod(env.observation_space.shape)
        self.features = torch.nn.Sequential(
            torch.nn.Linear(inputs, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU()
        )

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, env.action_space.n),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        value = self.critic(x)
        actions = self.actor(x)
        return value, actions

    def get_critic(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.critic(x)

    def evaluate_action(self, state, action):
        value, actor_features = self.forward(state)
        dist = torch.distributions.Categorical(actor_features)

        log_probs = dist.log_prob(action).view(-1, 1)
        entropy = dist.entropy().mean()

        return value, log_probs, entropy

    def act(self, state):
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
        actions = LongTensor(self.actions)
        true_values = FloatTensor(self.true_values).unsqueeze(1)

        self.states, self.actions, self.true_values = [], [], []

        return states, actions, true_values

def compute_true_values(model, states, rewards, dones):
    R = []
    rewards = FloatTensor(rewards)
    dones = FloatTensor(dones)
    states = torch.stack(states)

    if dones[-1] == True:
        next_value = rewards[-1]
    else:
        next_value = model.get_critic(states[-1].unsqueeze(0))

    R.append(next_value)
    for i in reversed(range(0, len(rewards) - 1)):
        if not dones[i]:
            next_value = rewards[i] + next_value * gamma
        else:
            next_value = rewards[i]
        R.append(next_value)

    R.reverse()

    return FloatTensor(R)


def reflect(model, optimizer, memory):
    states, actions, true_values = memory.pop_all()

    values, log_probs, entropy = model.evaluate_action(states, actions)

    advantages = true_values - values
    critic_loss = advantages.pow(2).mean()

    actor_loss = -(log_probs * advantages.detach()).mean()
    total_loss = (critic_coef * critic_loss) + actor_loss - (entropy_coef * entropy)

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    return values.mean().item()

class Worker(object):
    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.episode_reward = 0
        self.state, _ = self.env.reset()
        self.state = torch.from_numpy(self.state).float()

    def get_batch(self):
        states, actions, rewards, dones = [], [], [], []
        for _ in range(batch_size):
            action = self.model.act(self.state.unsqueeze(0))
            next_state, reward, done, _, _ = self.env.step(action)
            self.episode_reward += reward

            states.append(self.state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            if done:
                self.state, _ = self.env.reset()
                self.state = torch.from_numpy(self.state).float()
                data['episode_rewards'].append(self.episode_reward)
                self.episode_reward = 0
            else:
                self.state = FloatTensor(next_state)

        values = compute_true_values(self.model, states, rewards, dones).unsqueeze(1)
        return states, actions, values


def plot(data, frame_idx):
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
    plt.show()


def train_model(env):
    model = Model(env)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, eps=1e-5)
    memory = Memory()
    workers = []
    for _ in range(no_of_workers):
        workers.append(Worker(env, model))
    frame_idx = 0

    state, _ = env.reset()
    state = torch.from_numpy(state).float()
    episode_reward = 0
    while frame_idx < max_frames:
        for worker in workers:
            states, actions, true_values = worker.get_batch()
            for i, _ in enumerate(states):
                memory.push(
                    states[i],
                    actions[i],
                    true_values[i]
                )
            frame_idx += batch_size

        value = reflect(model, optimizer, memory)
        print(value)
        print("[" + str(frame_idx) + "]")
        if frame_idx % 1000 == 0:
            print(value)
            data['values'].append(value)
            #plot(data, frame_idx)
    #torch.save(model.state_dict(), 'model_weights.pth')
    return model




def test_model(env, state_dict):
    testing_rewards = []
    with torch.no_grad():
        model = Model(env)  # Initialize a new model
        model.load_state_dict(state_dict)  # Load the state dict
        model.eval()  # Set the model to evaluation mode
        #self.model.eval()
        state, _ = env.reset()
        total_reward =0
        done = False

        while not done:
            state = torch.from_numpy(state).float().unsqueeze(0)  # Convert state to tensor
            #print(state)
            with torch.no_grad():  # Don't compute gradients
                action = model.act(state)
            next_state, reward, done, _, _ = env.step(action)  # Take action in the environment
            total_reward += reward
            print(reward)
            state = next_state
            env.render()  # Render the screen
    print(f"Test episode reward: {total_reward}")
