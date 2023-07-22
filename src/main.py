import pandas as pd
import numpy as np

import yfinance as yf
import tushare as ts
from stockstats import StockDataFrame as Sdf
import stockstats

from tqdm import tqdm
import requests
from datetime import datetime, timedelta

import os 
import sys

from stable_baselines3 import PPO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from preprocessing.preprocess import get_preprocessed_data
from rl_environments.stock_portfolio_env import StockPortfolioEnv
from rl_environments.stock_env import StockTradingEnv
from rl_environments.simple_stock_env import SimpleStockEnv
from resources.helper import load_configs
from models.prebuilt.deep_rl_agent import PPOAgent

configs=load_configs()


def run_preprocessing():
    train, val, test = get_preprocessed_data(symbols=configs["SYMBOLS"])
    train.to_csv("./data/train_large_cap_no_fundamentals.csv")
    val.to_csv("./data/val_large_cap_no_fundamentals.csv")
    test.to_csv("./data/test_large_cap_no_fundamentals.csv")

def create_stock_portfolio_env(path):
    df = pd.read_csv(path, index_col="Unnamed: 0")
    df.index = df.reset_index()["index"] - df.reset_index()["index"].min()
    num_stocks = len(df.tic.unique())
    print(f"Number of unique stocks: {num_stocks}")
    env = StockPortfolioEnv(
        df=df,
        stock_dim=num_stocks,
        hmax=None,
        initial_amount=100000,
        start_all_cash=True,
        transaction_cost_pct=0,
        reward_scaling=1000,
        state_space=num_stocks,
        action_space=num_stocks + 1, # Add 1 for cash
        tech_indicator_list=configs["INDICATORS"]["TECHNICAL"] + configs["INDICATORS"]["FUNDAMENTAL"]
    )
    return env

def create_stock_env(path, model_name, mode):
    df = pd.read_csv(path, index_col="Unnamed: 0")
    df.index = df.reset_index()["index"] - df.reset_index()["index"].min()
    num_stocks = len(df.tic.unique())
    # if num_stocks != 1:
    #     raise ValueError("You can only use 1 stock for this env.")
    ind = configs["INDICATORS"]["TECHNICAL"] + configs["INDICATORS"]["FUNDAMENTAL"]
    env = StockTradingEnv(
        df=df,
        stock_dim=num_stocks,
        hmax=10000,
        initial_amount=1000000,
        num_stock_shares=[0]*num_stocks,
        reward_scaling=1e-4,
        state_space=1 + 2 * num_stocks + len(ind) * num_stocks,
        action_space=num_stocks,
        tech_indicator_list=ind,
        buy_cost_pct=[0.001] * num_stocks,
        sell_cost_pct=[0.001] * num_stocks,
        model_name=model_name,
        mode=mode,
    )
    return env

def create_simple_stock_env(path):
    df = pd.read_csv(path, index_col="Unnamed: 0")
    df.index = df.reset_index()["index"] - df.reset_index()["index"].min()
    num_stocks = len(df.tic.unique())
    if num_stocks != 1:
        raise ValueError("You can only use 1 stock for this env.")
    ind = configs["INDICATORS"]["TECHNICAL"] #+ configs["INDICATORS"]["FUNDAMENTAL"]
    env = SimpleStockEnv(
        df=df,
        stock_dim=num_stocks,
        initial_amount=1000000,
        indicators=ind,
    )
    return env

def test_base_agent(env, timesteps, specified_weight_arr=None):
    """
    base agent tests equal weights over time or specified constant weights over time.
    """
    num_weights = env.stock_dim + 1
    if specified_weight_arr is None:
        action = np.array([1]*num_weights)
    else:
        action = specified_weight_arr
    cumulative_rewards = {}
    episode=0
    for _ in range(timesteps):
        s, r, t, info = env.step(action)
        if t:
            cumulative_rewards[episode] = env.portfolio_value
            break
    return cumulative_rewards

def test_env_creation(env, action=None, steps=5):
    rewards = []
    if action is None:
        action = env.action_space.sample()
    for i in range(steps):
        if i != 0:
            action = env.action_space.sample()
        if env.__class__.__name__ != "StockPortfolioEnv":
            s, r, t, _, _ = env.step(action)
        else:
            s, r, t, i = env.step(action)
        rewards.append(r)
    return rewards


def train_agent(env, timesteps):
    # This is using FinRL implementation of PPOAgent
    train_env, _ = env.get_sb_env()
    agent = PPOAgent(env=train_env)
    model = agent.get_model(model_name="ppo")
    trained_ppo = agent.train_model(model, tb_log_name="ppo", total_timesteps=timesteps)
    return trained_ppo

def test_agent(model, test_env):
    df_daily_return_ppo, df_actions_ppo = PPOAgent.DRL_prediction(
        model=model,
        environment = test_env
    )
    return df_daily_return_ppo, df_actions_ppo

def main(problem="single_stock", needs_preproccess=True):
    if needs_preproccess:
        run_preprocessing()

    if problem == "stock_trader":
        env = create_stock_env(path="./data/train_large_cap_no_fundamentals.csv", model_name="PPO", mode="single_stock_SPY")
        trained_ppo_model = train_agent(env, 250000)

        try:
            trained_ppo_model.save("./trained_models/ppo_single_stock.zip")
        except Exception as e:
            print(e)
        
        test_env = create_stock_env(path="./data/test_large_cap_no_fundamentals.csv", model_name="PPO", mode="single_stock_SPY")
        trained_ppo_model = PPO.load("./trained_models/ppo_single_stock.zip")
        df_daily_return_ppo, df_actions_ppo = test_agent(trained_ppo_model, test_env)
        print(df_daily_return_ppo)
        print(df_actions_ppo)
    elif problem == "portfolio_allocation":
        env = create_stock_portfolio_env("./data/train_large_cap_no_fundamentals.csv")
        trained_ppo_model = train_agent(env, 250000)

        try:
            trained_ppo_model.save("./trained_models/ppo_large_cap.zip")
        except Exception as e:
            print(e)

        test_env = create_stock_portfolio_env("./data/test_large_cap_no_fundamentals.csv")
        trained_ppo_model = PPO.load("./trained_models/ppo_large_cap.zip")
        df_daily_return_ppo, df_actions_ppo = test_agent(trained_ppo_model, test_env)
        print(df_daily_return_ppo)
        print(df_actions_ppo)
    elif problem == "simple_stock_trader":
        env = create_simple_stock_env("./data/train_large_cap_no_fundamentals.csv")
        trained_ppo_model = train_agent(env, 250000)

        try:
            trained_ppo_model.save("./trained_models/ppo_simple_stock.zip")
        except Exception as e:
            print(e)

        test_env = create_simple_stock_env("./data/test_large_cap_no_fundamentals.csv")
        trained_ppo_model = PPO.load("./trained_models/ppo_simple_stock.zip")
        df_daily_return_ppo, df_actions_ppo = test_agent(trained_ppo_model, test_env)
        print(df_daily_return_ppo)
        print(df_actions_ppo)

    else:
        raise ValueError("please use 'single_stock' or 'portfolio_allocation' for problem")

if __name__ == "__main__":
    main(problem="simple_stock_trader", needs_preproccess=True)

## Implement own PPO algorithm with 1 stock over time