import pandas as pd
import numpy as np

import yfinance as yf

import requests
import os
import sys
import tqdm

from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Local Imports
from resources.helper import load_configs

configs = load_configs()

class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from neofinrl_config.py)
        end_date : str
            end date of the data (modified from neofinrl_config.py)
        ticker_list : list
            a list of stock tickers (modified from neofinrl_config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self, proxy=None) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        num_failures = 0
        for tic in self.ticker_list:
            temp_df = yf.download(
                tic, start=self.start_date, end=self.end_date, proxy=proxy
            )
            temp_df["tic"] = tic
            if len(temp_df) > 0:
                # data_df = data_df.append(temp_df)
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                num_failures += 1
        if num_failures == len(self.ticker_list):
            raise ValueError("no data is fetched.")
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
                "tic",
            ]
            # use adjusted close price instead of close price
            data_df["close"] = data_df["adjcp"]
            # drop the adjusted close price column
            data_df = data_df.drop(labels="adjcp", axis=1)
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        return data_df

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df
    
class AlphaVantageLoader:

    def __init__(self, api_key, symbols):
        self.api_key = api_key
        self.symbols = symbols

    def fetch_data(self, df: pd.DataFrame, fundamental_indicators: list):
        """
        fetches data from the AlphaVantage API and joins it to current dataframe

        parameters
        ----------
        df: pd.DataFrame
            time series data from yahoo with features
        fundamental_indicators: list
            list of fundamental indicators to include in the model
        
        returns
        -------
        df: pd.DataFrame
            df with all fundamental data attached.
        """
        res_df = pd.DataFrame()
        for tic in tqdm.tqdm(self.symbols):
            print(f"{tic} fundamental data loading...")
            temp_df = self._join_features(df, tic, fundamental_indicators)
            res_df = pd.concat([res_df, temp_df], axis=0)
        return res_df
    
    def _join_features(self, df, tic, fundamental_indicators):
        if "3month_yield" in fundamental_indicators:
            feat_df = self.get_treasury_data("daily", "3month")
            df = df.merge(
                feat_df, 
                how="left", 
                on=["date", "tic"]
            ).fillna(method="ffill")

        if "2year_yield" in fundamental_indicators:
            feat_df = self.get_treasury_data("daily", "2year")
            df = df.merge(
                feat_df, 
                how="left", 
                on=["date", "tic"]
            ).fillna(method="ffill")

        if "5year_yield" in fundamental_indicators:
            feat_df = self.get_treasury_data("daily", "5year")
            df = df.merge(
                feat_df, 
                how="left", 
                on=["date", "tic"]
            ).fillna(method="ffill")

        if "7year_yield" in fundamental_indicators:
            feat_df = self.get_treasury_data("daily", "7year")
            df = df.merge(
                feat_df, 
                how="left", 
                on=["date", "tic"]
            ).fillna(method="ffill")

        if "10year_yield" in fundamental_indicators:
            feat_df = self.get_treasury_data("daily", "10year")
            df = df.merge(
                feat_df, 
                how="left", 
                on=["date", "tic"]
            ).fillna(method="ffill")

        if "30year_yield" in fundamental_indicators:
            feat_df = self.get_treasury_data("daily", "30year")
            df = df.merge(
                feat_df, 
                how="left", 
                on=["date", "tic"]
            ).fillna(method="ffill")

        if "earnings" in fundamental_indicators:
            feat_df = self.get_fundamental_data(
                function="EARNINGS",
                symbol=tic
            )
            df = df.merge(
                feat_df, 
                how="left", 
                on=["date", "tic"]
            ).fillna(method="ffill")

        if "balance_sheet" in fundamental_indicators:
            feat_df = self.get_fundamental_data(
                function="BALANCE_SHEET",
                symbol=tic
            )
            df = df.merge(
                feat_df, 
                how="left", 
                on=["date", "tic"]
            ).fillna(method="ffill")

        if "income_statement" in fundamental_indicators:
            feat_df = self.get_fundamental_data(
                function="INCOME_STATEMENT",
                symbol=tic
            )
            df = df.merge(
                feat_df, 
                how="left", 
                on=["date", "tic"]
            ).fillna(method="ffill")

        if "cash_flow" in fundamental_indicators:
            feat_df = self.get_fundamental_data(
                function="CASH_FLOW",
                symbol=tic
            )
            df = df.merge(
                feat_df, 
                how="left", 
                on=["date", "tic"]
            ).fillna(method="ffill")

        return df
            

    def get_fundamental_data(self, function, symbol):
        """
        gets fundamental data for a symbol

        parameters
        ---------
        function: str
            type of fund data
            EXAMPLES:
                - "EARNINGS"
                - "BALANCE_SHEET"
                - "INCOME_STATEMENT"
                - "CASH_FLOW"
        symbol: str
            stock ticker
        
        returns
        -------
        df: pd.DataFrame
            pandas dataframe containing fundamental data
        """
        url = configs["URLS"]["FUNDAMENTAL_DATA"].format(
            function=function, 
            symbol=symbol, 
            key=self.api_key,
        )
        r = requests.get(url)
        data = r.json()

        if function == "EARNINGS":
            df = pd.DataFrame(data["quarterlyEarnings"])
            df["tic"] = symbol
            df = df.rename(columns={"reportedDate": "date"})
            df = df.drop(["fiscalDateEnding", "surprise"], axis=1)
            df["date"] = pd.to_datetime(df["date"])
        else:
            df = pd.DataFrame(data['quarterlyReports'])
            df["tic"] = symbol
            df = df.rename(columns={"fiscalDateEnding": "date"})
            df = df.drop(["reportedCurrency"], axis=1)
            df["date"] = pd.to_datetime(df["date"]) + timedelta(1)

        return df

    def get_treasury_data(self, interval, maturity):
        """
        gets treasury yield over a certain period.

        parameters
        ---------
        interval: str
            data frequence (E.G. DAILY)
        maturity: str
            yield curve maturity date
            EXAMPLES:
                - "3month, 2year, 5year, 7year, 10year, 30year"
        
        returns
        -------
        df: pd.DataFrame
            pandas dataframe containing treasury data
        """
        url = configs["URLS"]["TREASURY_DATA"].format(
            interval=interval, 
            maturity=maturity, 
            key=self.api_key,
        )
        r = requests.get(url)
        data = r.json()

        df = pd.DataFrame(data['data'])
        df = df.rename(columns={"value": f"{maturity}_rate"})
        df["date"] = pd.to_datetime(df["date"])

        return df

    def get_time_series_data(self, symbol, interval, function="TIME_SERIES_INTRADAY"):
        """
        gets time series for a certain stock symbol.

        **DON'T USE YET**

        parameters
        ---------
        symbol: str
            stock ticker
        interval: str
            data frequence (E.G. DAILY)
        function: str
            type of time series
        
        returns
        -------
        df: pd.DataFrame
            pandas dataframe containing time series data
        """
        url = configs["URLS"]["TIME_SERIES_DATA"].format(
            symbol=symbol, 
            interval=interval, 
            key=self.api_key,
            function=function,
        )
        r = requests.get(url)
        data = r.json()

        return data
