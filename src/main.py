import os
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from preprocessing.preprocess import get_preprocessed_data
from rl_environments.env_creation_functions import *
from resources.helper import load_configs, maximize_class_probability
from models.prebuilt.deep_rl_agent import PPOAgent
from models.scratch.dqn import Agent, DQN
from stable_baselines3 import A2C
#from models.scratch.a2c import train_a2c, test_a2c  # import the A2C functions
from models.scratch.a2c import train_model, test_model
configs=load_configs()
def run_preprocessing():
    train, val, test = get_preprocessed_data(symbols=configs["SYMBOLS"])
    train.to_csv("../data/train_large_cap_no_fundamentals.csv")
    val.to_csv("../data/val_large_cap_no_fundamentals.csv")
    test.to_csv("../data/test_large_cap_no_fundamentals.csv")

def train_dqn(env, episodes=50):
    agent = Agent(
        env,
        M=episodes,
        epsilon_decay=1 - 1e-5,
        layer_size=128,
        batch_size=128,
        C=10,
        model_suffix=configs["SYMBOLS"][0]
    )
    agent.train()
    agent.test()

def test_dqn(env):
    agent = Agent(
        env,
        epsilon_decay=1 - 1e-5,
        layer_size=128,
        batch_size=128,
        C=10,
        model_suffix= configs["SYMBOLS"][0]
    )
    agent.test()

def train_ppo(env, timesteps):
    # This is using FinRL implementation of PPOAgent
    train_env, _ = env.get_sb_env()
    agent = PPOAgent(env=train_env)
    model = agent.get_model(model_name="ppo")
    trained_ppo = agent.train_model(model, tb_log_name="ppo", total_timesteps=timesteps)
    return trained_ppo

def test_ppo(model, test_env):
    df_daily_return_ppo, df_actions_ppo = PPOAgent.DRL_prediction(
        model=model,
        environment = test_env
    )
    return df_daily_return_ppo, df_actions_ppo

def main(
        problem="single_stock", 
        needs_preproccess=True,
        needs_training=True,
        rl_algorithm="dqn",
        train_path="../data/train_large_cap_no_fundamentals.csv",
        test_path="../data/test_large_cap_no_fundamentals.csv",
):
    if needs_preproccess:
        run_preprocessing()

    if problem == "stock_trader":
        env = create_stock_env(path=train_path, model_name="PPO", mode="single_stock_SPY")
        test_env = create_stock_env(path=test_path, model_name="PPO", mode="single_stock_SPY")
    elif problem == "portfolio_allocation":
        env = create_stock_portfolio_env(train_path)
        test_env = create_stock_portfolio_env(test_path)
    elif problem == "simple_stock_trader":
        env = create_simple_stock_env(train_path)
        test_env = create_simple_stock_env(test_path)
    else:
        raise ValueError("please use 'single_stock', 'portfolio_allocation', or 'simple_stock_trader' for problem")

    if rl_algorithm == "ppo":
        if needs_training:
            trained_ppo_model = train_ppo(env, 250000)
            try:
                trained_ppo_model.save("./trained_models/ppo_single_stock.zip")
            except Exception as e:
                print(e)
        
        try:
            trained_ppo_model = PPO.load("./trained_models/ppo_single_stock.zip")
        except Exception as e:
            print("problem loading PPO model")

        df_daily_return_ppo, df_actions_ppo = test_ppo(trained_ppo_model, test_env)
        print(df_daily_return_ppo)
        print(df_actions_ppo)
    
    elif rl_algorithm == "dqn":
        
        if needs_training:
            train_dqn(env, episodes=75)
        test_dqn(test_env)
    
    elif rl_algorithm == "a2c":
        if needs_training:
            trained_a2c_model = train_model(env=env, layer_size=156, learning_rate=0.001, gamma=0.80, critic_coef=0.5, entropy_coef=0.01, c=100)

            try:
                torch.save(trained_a2c_model.state_dict(), "../trained_models/a2c_single_stock.pth")
            except Exception as e:
                print(e)
        else:
            try:
                trained_a2c_model = torch.load("../trained_models/a2c_single_stock.pth")  # Load the state dict
            except Exception as e:
                print("problem loading A2C model")
            test_model(test_env, trained_a2c_model)

    else:
        raise ValueError("please use 'ppo', 'dqn', or 'a2c' for rl_algorithm")
    
def get_best_inputs():
    model_state_dict = torch.load('../trained_models/DQN.pt')
    env = create_simple_stock_env("../data/train_large_cap_no_fundamentals.csv")
    inputs = np.prod(env.observation_space.shape)
    layer_size = model_state_dict["network_stack.2.weight"].size(0)
    model = DQN(env=env, LAYER_SIZE=layer_size)
    model.load_state_dict(model_state_dict)
    
    null_inputs = True
    while null_inputs:
        good_inputs = maximize_class_probability(model=model, input_size=inputs, class_idx=2, lr=0.01, max_steps=100000)
        if good_inputs[:,0] != torch.nan:
            null_inputs=False
    print(f"buy score: {model(good_inputs)[:,2]}")
    return good_inputs

if __name__ == "__main__":
    main(rl_algorithm="a2c", problem="simple_stock_trader", needs_preproccess=False)
    #main(rl_algorithm="a2c", needs_preproccess=False, needs_training=False, problem="simple_stock_trader")

# TODO: Build a clustering algorithm that identifies similar states to good inputs and sorts then takes top 5, it can buy the top 5 and sell the bottom 5.
# TODO: Even better: run most recent state for all spy stocks through screener. Buy the top 5 and sell and buy when another takes its place in top 5
