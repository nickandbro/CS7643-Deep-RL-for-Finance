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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.preprocessing.preprocess import get_preprocessed_data
from src.rl_environments.stock_portfolio_env import StockPortfolioEnv
from src.rl_environments.stock_env import StockTradingEnv
from src.rl_environments.simple_stock_env import SimpleStockEnv
from src.resources.helper import load_configs
from src.models.prebuilt.deep_rl_agent import PPOAgent
from src.models.scratch.dqn import Agent, DQN

configs=load_configs()

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