import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

def load_configs(config_path="../configs/configs.yml"):
    import os
    print(os.getcwd())
    with open(config_path) as f:
        configs = yaml.load(f, Loader=yaml.Loader)
    return configs

def maximize_class_probability(model, input_size, class_idx=2, lr=0.01, max_steps=1000000):
    # Define the optimizer for the input data
    input_data = Variable(torch.randn(1, input_size), requires_grad=True)

    optimizer = optim.Adam([input_data], lr=lr)

    s = 0
    while True:
        optimizer.zero_grad()

        # Forward pass through the model
        output = model(input_data)
        prob_class_2 = output[:,2]
        if prob_class_2 >= .999:
            break

        # Use negative log-probability to maximize the probability
        loss = -torch.log(prob_class_2)
        loss.backward()

        optimizer.step()

        if s >= max_steps:
            break
        
        s+=1

    return input_data