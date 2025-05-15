# TD3-Based Stock Trading Agent
This project implements a Twin Delayed Deep Deterministic Policy Gradient (TD3) agent to simulate and optimize trading strategies in a custom-built financial market environment.

NOTE: The project is still a work in progress.

## Overview
The agent learns to make buy/sell/hold decisions based on historical stock price data. The training loop involves:

- Preprocessing data and constructing a trading environment
- Training a TD3 agent using a replay buffer
- Evaluating policy on a validation set based on the defined trading metrics
- Saving the best model for final testing

## Data fetch & preprocess
We use OANDA for fetching stock data with granularity ranging (30 seconds to 5 minutes). Based on OHLCV, we calculate technical indicators such as RSI, MACD, Log Returns, etc.

## Environment
The trading environment is defined with the following parameters:

| Parameter           | Value                    | Description                                                                 |
|---------------------|--------------------------|-----------------------------------------------------------------------------|
| `lookback_window`   | 60                       | Number of past timesteps considered for state representation               |
| `frame_stack`       | 4                        | Number of past frames stacked to form state                                |
| `transaction_cost`  | 0.0003                   | Cost incurred per trade as a fraction of transaction value                 |
| `max_position`      | 1.0                      | Maximum allowable position (long/short) per stock                          |
| `max_episodes`      | 10                       | Total number of episodes to train the agent                                |
| `max_timesteps`     | 1.5 × len(train_df)      | Maximum timesteps per episode                                              |
| `batch_size`        | 512                      | Size of mini-batch for training                                            |
| `discount`          | 0.995                    | Discount factor (γ) for future rewards                                     |
| `tau`               | 0.0005                   | Soft update parameter for target networks                                  |
| `policy_noise`      | 0.15                     | Standard deviation of noise added to target policy during critic update    |
| `noise_clip`        | 0.35                     | Maximum allowed noise during target policy smoothing                       |
| `policy_freq`       | 2                        | Frequency of policy (actor) updates relative to critic updates             |
| `exploration_noise` | 0.2                      | Noise added to actions for exploration during training                     |
| `eval_freq`         | 2                        | Frequency (in episodes) of model evaluation                                |

The agent's action can range from (-1, 1) where -1 means sell all, 0 means to hold, and 1 means to buy maximum amount possible.

## Reward structure

The main aim of this reward structure is to encourage profitable trading, penalize inefficient behavior, and take timely action during market trends.

| Component                 | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `performance_reward`     | `np.clip(portfolio_return * 10, 0, 1)` — Rewards growth in portfolio value. |
| `action_reward`          | `np.clip(expected_profit / old_value * 10, 0, 1)` — Rewards good trades.     |
| `excessive_cost_penalty`| Penalizes trades where cost > gain.                                          |
| `drawdown_penalty`       | `np.clip(drawdown * 2, 0, 1)` — Penalizes big losses from peak value.        |
| `hold_penalty`           | Penalizes inactivity when the market is trending.                           |

**Final Reward**  
```python
reward = (performance_reward + action_reward) 
         - excessive_cost_penalty 
         - drawdown_penalty 
         - hold_penalty
```

## Evaluation
The agent is evaluated using the following metrics:

- Sharpe Ratio: Risk-adjusted return
- Max Drawdown: Largest observed loss from a peak
- Standard Deviation of Returns
- Portfolio Value over time

## Results so far

- Stock used: AAPL
- Granularity: 1D
- Start = 2018-01-01
- End = 2023-12-31

|       Metric        |        Value         |
|---------------------|----------------------|
| Max Return          | 92.26459503173828    |
| Min Return          | -66.63027954101562   |
| Mean Return         | 34.61904547184706    |
| Std Dev of Return   | 30.909593847396692   |
| Sharpe Ratio        | 1.1200097174606773   |

![results](https://github.com/user-attachments/assets/7f611b4b-37bc-486d-8cf1-69247b617173)
