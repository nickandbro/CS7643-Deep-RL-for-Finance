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

from preprocessing.preprocess import get_preprocessed_data
from rl_environments.stock_portfolio_env import StockPortfolioEnv
from rl_environments.stock_env import StockTradingEnv
from rl_environments.simple_stock_env import SimpleStockEnv
from resources.helper import load_configs
from models.prebuilt.deep_rl_agent import PPOAgent
from models.scratch.dqn import Agent, DQN

configs=load_configs()


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

def create_simple_stock_env(path, data_set):
    df = pd.read_csv(path, index_col="Unnamed: 0")
    df.index = df.reset_index()["index"] - df.reset_index()["index"].min()
    num_stocks = len(df.tic.unique())
    if num_stocks != 1:
        raise ValueError("You can only use 1 stock for this env.")
    ind = configs["INDICATORS"]["TECHNICAL"] #+ configs["INDICATORS"]["FUNDAMENTAL"]
    env = SimpleStockEnv(
        df=df,
        data_set=data_set,
        stock_dim=num_stocks,
        initial_amount=1000000,
        indicators=ind,
    )
    return env
