import pandas as pd
import numpy as np

import yfinance as yf
import tushare as ts
from stockstats import StockDataFrame as Sdf
import stockstats

from tqdm import tqdm
import requests
from datetime import datetime, timedelta

import time
import os 
import sys

from stable_baselines3 import PPO

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from preprocessing.preprocess import get_preprocessed_data
from preprocessing.state_generator import get_current_state

from rl_environments.stock_portfolio_env import StockPortfolioEnv
from rl_environments.stock_env import StockTradingEnv
from rl_environments.simple_stock_env import SimpleStockEnv
from rl_environments.env_creation_functions import *
from models.scratch.a2c import train_model, test_model


from resources.helper import load_configs, maximize_class_probability

from models.prebuilt.deep_rl_agent import PPOAgent
from models.scratch.dqn import Agent, DQN

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
        model_suffix=configs["SYMBOLS"][0],
        **configs["DQN_PARAMS"]
    )
    agent.train()
    agent.test()

def test_dqn(env):
    agent = Agent(
        env,
        model_suffix= configs["SYMBOLS"][0],
        **configs["DQN_PARAMS"]
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
        env = create_simple_stock_env(train_path, data_set="train")
        test_env = create_simple_stock_env(test_path, data_set="test")
    else:
        raise ValueError("please use 'single_stock', 'portfolio_allocation', or 'simple_stock_trader' for problem")

    if rl_algorithm == "ppo":
        if needs_training:
            trained_ppo_model = train_ppo(env, 250000)
            try:
                trained_ppo_model.save("../trained_models/ppo_single_stock.zip")
            except Exception as e:
                print(e)
        
        try:
            trained_ppo_model = PPO.load("../trained_models/ppo_single_stock.zip")
        except Exception as e:
            print("problem loading PPO model")

        df_daily_return_ppo, df_actions_ppo = test_ppo(trained_ppo_model, test_env)
        print(df_daily_return_ppo)
        print(df_actions_ppo)
    
    elif rl_algorithm == "dqn":
        
        if needs_training:
            train_dqn(env, episodes=100)
        test_dqn(env)
        test_dqn(test_env)


    elif rl_algorithm == "a2c":
        if needs_training:
            trained_a2c_model = train_model(env=env, layer_size=156, learning_rate=0.001, gamma=0.80, critic_coef=0.5,
                                            entropy_coef=0.01, c=5)

            try:
                torch.save(trained_a2c_model.state_dict(), "../trained_models/a2c_single_stock.pth")
            except Exception as e:
                print(e)
        else:
            try:
                trained_a2c_model = torch.load("../trained_models/a2c_single_stock.pth")  # Load the state dict
            except Exception as e:
                print("problem loading A2C model")
            test_model(test_env,layer_size=156, state_dict=trained_a2c_model)
    else:
        raise ValueError("please use 'ppo', 'dqn', or 'a2c' for rl_algorithm")
    
def get_best_inputs():
    model_state_dict = torch.load('./trained_models/DQN.pt')
    env = create_simple_stock_env("./data/train_large_cap_no_fundamentals.csv")
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

def get_buy_signals(current_state_df, indicators):

    model_state_dict = torch.load('./trained_models/DQN.pt')
    env = create_simple_stock_env("./data/train_large_cap_no_fundamentals.csv")
    inputs = np.prod(env.observation_space.shape)
    layer_size = model_state_dict["network_stack.2.weight"].size(0)
    model = DQN(env=env, LAYER_SIZE=layer_size)
    model.load_state_dict(model_state_dict)

    df = current_state_df[indicators].to_numpy()
    df = np.hstack((df, np.zeros(shape=df.shape[0]).reshape(-1,1)))
    df = np.hstack((df, np.zeros(shape=df.shape[0]).reshape(-1,1)))
    X = torch.tensor(df)
    y = model(X)[:,2]
    current_state_df["buy_signal"] = y.detach().numpy()
    return current_state_df.sort_values(by="buy_signal", ascending=False)

if __name__ == "__main__":
    # current_state_df = get_current_state(needs_preprocessing=False)
    # df = get_buy_signals(current_state_df=current_state_df, indicators=configs["INDICATORS"]["TECHNICAL"])
    # df.to_csv("./data/current_best_states.csv")
    main(
        problem="simple_stock_trader", 
        needs_preproccess=False,
        needs_training=False,
        rl_algorithm="dqn",
        train_path="./data/train_large_cap_no_fundamentals.csv",
        test_path="./data/test_large_cap_no_fundamentals.csv",
    )
