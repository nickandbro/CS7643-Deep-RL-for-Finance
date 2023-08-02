import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
import cv2
from stable_baselines3.common.vec_env import DummyVecEnv

import matplotlib.dates as mdates

import sys
import os

#matplotlib.use("Agg")

# from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat

class SimpleStockEnv(gym.Env):
    
    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            df: pd.DataFrame,
            stock_dim: int,
            initial_amount: int,
            indicators: list[str],
            start_all_cash: bool = True,

        ):
        self.df = df
        self.stock_dim = stock_dim
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(indicators) + 2,)) # +2 for holding info and % change.
        self.initial_amount = initial_amount
        self.indicators = indicators
        self.start_all_cash = start_all_cash
        
        self.reward = 0
        self.trades = 0
        self.day = 0
        self.data = self.df.loc[[self.day]]
        self.state = np.array([self.data[tech].values.tolist() for tech in self.indicators])
        # Apend holding info and % change since begining.
        self.state = np.append(self.state, 0)
        self.state = np.append(self.state, 0)
        self.initial_close = self.data.close
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.action_memory = []
        self.reward_memory = []
        self.fig, self.ax = plt.subplots()  # Add this line
        self.date_memory = [self.data.date.unique()[0]]

        self.terminal = False
        self.holds_stock = False
        self._seed()

    def render(self, mode='human'):
        if mode == 'human':
            self.ax.clear()

            # plot the stock price
            stock_prices = self.df['close'][:self.day + 1]
            self.ax.plot(stock_prices, label='Stock price')

            # mark the actions
            for i, action in enumerate(self.action_memory):
                print("action", action)
                if action == 2:  # Buy
                    self.ax.plot(i, self.df['close'].iloc[i], 'go')
                elif action == 1:  # Sell
                    self.ax.plot(i, self.df['close'].iloc[i], 'ro')

            plt.legend()
            plt.pause(0.01)  # pause for a moment
        else:
            super().render(mode=mode)

    def step(self, action):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            df = pd.DataFrame(np.array(self.asset_memory)/self.initial_amount)
            df.columns = ["daily_return"]
            df["date"] = self.date_memory
            fig, ax = plt.subplots(1,1,figsize=(8,6))
            ax.plot("date", "daily_return", data=df)
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))

            path = f"./results/simple_stock_env"
            if not os.path.exists(path):
                os.makedirs(path)
            
            plt.gcf().autofmt_xdate()
            plt.savefig("./results/simple_stock_env/cumulative_reward2.png")
            plt.close()

            plt.plot(self.portfolio_return_memory, "r")
            plt.savefig("./results/simple_stock_env/rewards2.png")
            plt.close()

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))
            print("end_total_asset:{}".format(self.portfolio_value))
            print(self.day)

            weight_df = pd.DataFrame(self.action_memory)
            weight_df.columns = list(self.df.tic.unique())
            weight_df.index = self.date_memory[:-1]
            weight_df.to_csv("./results/simple_stock_env/actions_memory.csv")
            

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]
            if df_daily_return["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_daily_return["daily_return"].mean()
                    / df_daily_return["daily_return"].std()
                )
                print("Sharpe: ", sharpe)
            print("=================================")


            return self.state, self.reward, self.terminal, False, {}

        else:
            last_day_memory = self.data
            prev_close = last_day_memory.close.values

            if action == 2:
                self.holds_stock = True
            elif action == 0:
                self.holds_stock = False

            self.day += 1
            self.data = self.df.loc[[self.day]]
            close = self.data.close.values
            self.state = np.array([self.data[tech].values.tolist() for tech in self.indicators])
            
            if self.holds_stock:
                self.reward = np.sum((close - prev_close)/prev_close) * self.portfolio_value
            else:
                self.reward = 0
            
            self.portfolio_value += self.reward
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(self.portfolio_value)
            self.action_memory.append(action)
            self.portfolio_return_memory.append(self.reward)

            holding = 1 if self.holds_stock else 0
            self.state = np.append(self.state, holding)
            pct_change_since_init = (close - self.initial_close)/self.initial_close
            self.state = np.append(self.state, pct_change_since_init)

            return self.state, self.reward, self.terminal, False, {}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        self.reward = 0
        self.trades = 0
        self.day = 0
        self.data = self.df.loc[[self.day]]
        self.state = np.array([self.data[tech].values.tolist() for tech in self.indicators])
        # Apend holding info and % change since begining.
        self.state = np.append(self.state, 0)
        self.state = np.append(self.state, 0)
        self.initial_close = self.data.close
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.action_memory = []
        self.reward_memory = []
        self.date_memory = [self.data.date.unique()[0]]

        self.terminal = False
        self.holds_stock = False

        #return (self.state, {})
        return self.state, {}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
    
    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        #print(len(date_list))
        #print(len(asset_list))
        df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']
        
        action_list = self.action_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = list(self.data.tic.values)
        df_actions.index = df_date.date[:-1]
        #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions


# TODO: (way in the future): add a multi layered env to model multi agent rewards.