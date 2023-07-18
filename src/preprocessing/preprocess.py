import yaml
import numpy as np
import pandas as pd

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from resources.helper import load_configs
from preprocessing.data_loader import YahooDownloader, AlphaVantageLoader
from preprocessing.feature_engineering import FeatureEngineer


def get_preprocessed_data(symbols):

    configs = load_configs()
    
    df = YahooDownloader(
        start_date=configs["DATES"]["TRAIN"]["START_DATE"],
        end_date=configs["DATES"]["TEST"]["END_DATE"],
        ticker_list=symbols,
    ).fetch_data()

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=configs["INDICATORS"]["TECHNICAL"],
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False)

    df = fe.preprocess_data(df)
    df["date"] = pd.to_datetime(df["date"])

    # CAN'T USE THIS UNTIL API IS UPGRADED
    
    # av_loader = AlphaVantageLoader(
    #     api_key=configs['API_KEYS']["ALPHA_VANTAGE"], 
    #     symbols=symbols
    # )
    # df = av_loader.fetch_data(
    #     df=df,
    #     fundamental_indicators=configs["INDICATORS"]["FUNDAMENTAL"]
    # )
    # df = df.drop("Unnamed: 0", axis=1)

    df = df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]

    return df