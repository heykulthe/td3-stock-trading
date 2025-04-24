import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import deque

logger = logging.getLogger('td3-stock-trading')

class TradingEnvironment:
    def __init__(self,
                 data,
                 lookback_window=60,
                 transaction_cost=0.001,
                 max_position=1.0,
                 frame_stack=4
        ):

        self.data = data
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.frame_stack = frame_stack

        self.feature_columns = [col for col in data.columns if col != 'time']
        self.num_features = len(self.feature_columns)

        self.current_idx = lookback_window
        self.end_idx = len(data) - 1
        self.current_position = 0.0
        self.cash = 1.0  # Start with 1 unit of cash
        self.portfolio_value = 1.0
        self.portfolio_history = [1.0]
        self.max_portfolio_value = 1.0
        self.last_action = 0.0

        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_alpha = 0.01

        self.trade_count = 0

        self.max_drawdown = 0.0

        self.state_buffer = deque(maxlen=frame_stack)
        for _ in range(frame_stack):
            self.state_buffer.append(np.zeros(self.num_features * lookback_window))

    def get_state_dim(self):
        return self.num_features * self.lookback_window * self.frame_stack + 1  # +1 for position

    def reset(self):
        self.current_idx = self.lookback_window
        self.current_position = 0.0
        self.cash = 1.0
        self.portfolio_value = 1.0
        self.portfolio_history = [1.0]
        self.max_portfolio_value = 1.0
        self.last_action = 0.0
        self.max_drawdown = 0.0

        self.trade_count = 0

        self.state_buffer = deque(maxlen=self.frame_stack)
        for _ in range(self.frame_stack):
            self.state_buffer.append(np.zeros(self.num_features * self.lookback_window))

        return self._get_observation()

    def _get_observation(self):
        lookback_data = self.data.iloc[self.current_idx - self.lookback_window:self.current_idx]

        features = lookback_data[self.feature_columns].values.flatten()

        self.state_buffer.append(features)

        stacked_state = np.concatenate(list(self.state_buffer) + [np.array([self.current_position])])

        return stacked_state

    def step(self, action):

        action = np.clip(action, -1, 1)

        target_position = action * self.max_position
        position_change = target_position - self.current_position

        current_price = self.data.iloc[self.current_idx]['close']

        transaction_cost = abs(position_change) * self.transaction_cost * current_price

        self.cash -= position_change * current_price + transaction_cost
        self.current_position = target_position

        self.current_idx += 1

        done = self.current_idx >= self.end_idx or self.portfolio_value < 0.25

        next_price = self.data.iloc[self.current_idx]['close']

        self.portfolio_value = self.cash + self.current_position * next_price
        self.portfolio_history.append(self.portfolio_value)

        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)

        drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        self.max_drawdown = max(self.max_drawdown, drawdown)

        price_return = (next_price / current_price) - 1
        position_return = self.current_position * price_return

        base_reward = position_return - transaction_cost

        drawdown_penalty = drawdown * drawdown * 2.0
        # drawdown_penalty = min(drawdown, 0.2) ** 2 * 2.0

        stability_penalty = abs(self.last_action - action) * 0.05

        change_penalty = (abs(position_change) ** 1.5) * 0.01
        # trade_penalty = 0.0025 if abs(position_change) > 0.01 else 0.0
        trade_penalty = abs(position_change) * 0.001
        # reversal_penalty = 0.1 if self.last_action * action < -0.5 else 0.0
        reversal_penalty = abs(self.last_action * action) * 0.1 if self.last_action * action < 0 else 0

        reward = (base_reward * 0.4
                  - drawdown_penalty * 0.2
                  - change_penalty * 0.1
                  - stability_penalty * 0.1
                  - trade_penalty * 0.1
                  - reversal_penalty * 0.1)

        self.last_action = action

        next_observation = self._get_observation()

        self.reward_mean = (1 - self.reward_alpha) * self.reward_mean + self.reward_alpha * reward
        self.reward_std = (1 - self.reward_alpha) * self.reward_std + self.reward_alpha * (
                    reward - self.reward_mean) ** 2

        normalized_reward = (reward - self.reward_mean) / (np.sqrt(self.reward_std) + 1e-8)

        normalized_reward = np.clip(normalized_reward, -10, 10)

        info = {
            'portfolio_value': self.portfolio_value,
            'max_portfolio_value' : self.max_portfolio_value,
            'position': self.current_position,
            'cash' : self.cash,
            'transaction_cost': transaction_cost,
            'drawdown': drawdown,
            'price_return': price_return,
            'reward': reward,
            'normalized_reward': normalized_reward,
        }

        return next_observation, normalized_reward, done, info

    def render(self, mode='human'):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.portfolio_history)
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Portfolio Value')
        plt.grid(True)

        prices = self.data['close'][self.lookback_window:self.current_idx + 1].values
        normalized_prices = prices / prices[0]

        plt.subplot(2, 1, 2)
        plt.plot(normalized_prices, label='Buy & Hold')
        plt.plot(self.portfolio_history, label='Strategy')
        plt.title('Strategy vs Buy & Hold')
        plt.xlabel('Time Steps')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()