# Stock Market Portfolio Environment
The stock market environment allows you to instatiate a "world" where an "agent" takes actions to reach the highest reward.
In our domain, the agent is trying to use a given set of indicators (RSI, SMA, etc.) to properly allocate capital to each stock.
The agent is provided the indicators for a given day and must come up with a weight to assign to each stock.

## States
The states in the environment are indicators given by YahooFinance

## Actions
The actions are weights for each stock. 
For example, if the agent is given 3 stocks and $1,000,000 to invest, there are 4 weights the agent needs to assign based on the current
state. The first 3 weights are for stocks A,B, and C, while the 4th weight represents how much cash the agent should have on hand.

## Rewards
The rewards are provided by mulitplying the weights by the percentage price change for a stock. It represents the daily gain based on portfolio
weights.

## Example Usage
```lang:python
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
```