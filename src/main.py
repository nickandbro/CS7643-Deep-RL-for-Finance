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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from preprocessing.preprocess import get_preprocessed_data
from rl_environments.stock_portfolio_env import StockPortfolioEnv
from resources.helper import load_configs

configs=load_configs()


def run_preprocessing():
    train, val, test = get_preprocessed_data(symbols=configs["SYMBOLS"])
    train.to_csv("./data/train_baseline_data_large_cap_no_fundamentals.csv")
    val.to_csv("./data/val_baseline_data_large_cap_no_fundamentals.csv")
    test.to_csv("./data/test_baseline_data_large_cap_no_fundamentals.csv")

def create_stock_portfolio_env(path):
    df = pd.read_csv(path, index_col="Unnamed: 0")
    num_stocks = len(df.tic.unique())
    env = StockPortfolioEnv(
        df=df,
        stock_dim=num_stocks,
        hmax=None,
        initial_amount=100000,
        transaction_cost_pct=0,
        reward_scaling=1,
        state_space=num_stocks + 1, # Add 1 for cash
        action_space=num_stocks + 1, # Add 1 for cash
        tech_indicator_list=configs["INDICATORS"]["TECHNICAL"]
    )
    return env

if __name__ == "__main__":
    # run_preprocessing()
    env = create_stock_portfolio_env("./data/baseline_data_large_cap_no_fundamentals.csv")