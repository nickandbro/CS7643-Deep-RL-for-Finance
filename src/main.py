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


def run_preprocessing():
    data = get_preprocessed_data(symbols=["NVDA", "AAPL", "META", "IBM"])
    data.to_csv("./data/baseline_data_large_cap_no_fundamentals.csv")


if __name__ == "__main__":
    run_preprocessing()