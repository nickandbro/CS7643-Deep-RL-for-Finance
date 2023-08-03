import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
import numpy as np

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.preprocessing.preprocess import get_preprocessed_data

def get_current_state(needs_preprocessing, symbols=None):
    if symbols is None:
        symbols = list(pd.read_csv("./data/spy_companies.csv")['Symbol'].dropna())
    if needs_preprocessing:
        _, _, test = get_preprocessed_data(symbols)
        test.to_csv("./data/all_spy_test.csv")
    else:
        test = pd.read_csv("./data/all_spy_test.csv", index_col="Unnamed: 0")
        test.index = test.reset_index()["index"] - test.reset_index()["index"].min()
        max_day = test.index.max()
    
    current_states = test.loc[[max_day]].sort_values(by="tic")

    return current_states