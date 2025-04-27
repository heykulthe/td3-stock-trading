import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import datetime
from collections import deque
import random

from src.model.trading_environment import TradingEnvironment
from src.model.replay_buffer import ReplayBuffer

# Import TD3 implementation
from src.model.td3 import TD3

from src.utils.logger import setup_logging
from src.data.perform_ops import PerformDataOperations
from src.data.preprocess import PreprocessData
logger = setup_logging()


def set_seeds(seed=42):
    logger.info(f"Setting random seed: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def evaluate_policy(policy, eval_env, eval_episodes=3):
    portfolio_values = []
    sharpe_ratios = []
    max_drawdowns = []
    std_devs = []

    for _ in range(eval_episodes):
        state = eval_env.reset()
        done = False

        while not done:
            action = policy.select_action(np.array(state))
            state, _, done, _ = eval_env.step(action[0])

        returns = np.array(eval_env.portfolio_history[1:]) / np.array(eval_env.portfolio_history[:-1]) - 1
        std_return = np.std(returns)
        sharpe = np.mean(returns) / (std_return + 1e-8) * np.sqrt(19656)

        portfolio_values.append(eval_env.portfolio_value)
        sharpe_ratios.append(sharpe)
        max_drawdowns.append(eval_env.max_drawdown)
        std_devs.append(std_return)

    logger.info(f"Sizes: Portfolio Values: {len(portfolio_values)} | Sharpe Values: {len(sharpe_ratios)} | Max Drawdowns: {len(max_drawdowns)} | Std Devs: {len(std_devs)}")

    return np.mean(portfolio_values), np.mean(sharpe_ratios), np.mean(max_drawdowns), np.mean(std_devs), portfolio_values[-1], sharpe_ratios[-1], max_drawdowns[-1], std_devs[-1]


def train_td3_for_trading(train_df,
                          val_df,
                          test_df,
                          lookback_window=60,
                          frame_stack=4,
                          transaction_cost=0.001,
                          max_position=1.0,
                          max_episodes=1000,
                          max_timesteps=100000,
                          batch_size=256,
                          discount=0.99,
                          tau=0.005,
                          policy_noise=0.2,
                          noise_clip=0.5,
                          policy_freq=2,
                          exploration_noise=0.1,
                          eval_freq=10,
                          save_dir='results'
    ):

    set_seeds()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = train_df


    if 'time' in data.columns:
        data['time'] = pd.to_datetime(data['time'])

    # train_size = int(0.7 * len(data))
    # val_size = int(0.15 * len(data))

    train_data = data
    val_data = val_df
    test_data = test_df

    logger.info(f"Data split - Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

    train_env = TradingEnvironment(
        train_data,
        lookback_window=lookback_window,
        transaction_cost=transaction_cost,
        max_position=max_position,
        frame_stack=frame_stack
    )

    val_env = TradingEnvironment(
        val_data,
        lookback_window=lookback_window,
        transaction_cost=transaction_cost,
        max_position=max_position,
        frame_stack=frame_stack
    )

    test_env = TradingEnvironment(
        test_data,
        lookback_window=lookback_window,
        transaction_cost=transaction_cost,
        max_position=max_position,
        frame_stack=frame_stack
    )

    logger.info("Train, Val, Test environment created successfully")

    state_dim = train_env.get_state_dim()
    action_dim = 1
    max_action = 1.0

    logger.info(f"State dimension: {state_dim}, Action dimension: {action_dim}")

    policy = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=discount,
        tau=tau,
        policy_noise=policy_noise,
        noise_clip=noise_clip,
        policy_freq=policy_freq
    )

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    logger.info("TD3 policy and ReplayBuffer initialized.")

    best_val_sharpe = -float('inf')
    episode_rewards = []
    val_sharpes = []
    val_returns = []
    val_drawdowns = []

    total_timesteps = 0

    logger.info("----- Starting TD3 training loop -----")

    for episode in range(1, max_episodes + 1):
        state = train_env.reset()

        logger.info(f"Episode {episode} started. Total timesteps so far: {total_timesteps}")

        episode_reward = 0
        episode_timesteps = 0
        done = False

        while not done and episode_timesteps < max_timesteps:
            episode_timesteps += 1
            total_timesteps += 1

            if total_timesteps < 1000:
                action = np.random.uniform(-max_action, max_action, size=(action_dim,))
                logger.debug(f"Episode {episode}, timestep {episode_timesteps}: Random action selected.")

            else:
                action = policy.select_action(np.array(state))
                action = action + np.random.normal(0, exploration_noise, size=action_dim)
                action = np.clip(action, -max_action, max_action)
                logger.debug(f"Episode {episode}, timestep {episode_timesteps}: Policy action with exploration selected.")

            next_state, reward, done, info = train_env.step(action[0])
            episode_reward += reward

            logger.info(
                f"Episode {episode} Summary:\n"
                f"  Portfolio Value: {info['portfolio_value']:.2f}\n"
                f"  Max Portfolio Value: {info['max_portfolio_value']:.2f}\n"
                f"  Position: {info['position']}\n"
                f"  Cash: {info['cash']}\n"
                # f"  Transaction Cost: {info['transaction_cost']}\n"
                f"  Drawdown: {info['drawdown']:.4f}\n"
                f"  Price Return: {info['price_return']:.4f}\n"
                f"  Reward: {info['reward']:.4f}\n"
                f"  Normalized Reward: {info['normalized_reward']:.4f}\n"
            )

            replay_buffer.add(state, action, next_state, reward, done)

            state = next_state

            if total_timesteps >= 50000:
                policy.train(replay_buffer, batch_size)
                logger.debug(f"Training policy at timestep {total_timesteps}")

        episode_rewards.append(episode_reward)

        if episode % eval_freq == 0:
            avg_portfolio, avg_sharpe, avg_drawdown, avg_std_dev, last_portfolio_value, last_sharpe_ratio, last_max_drawdown, last_std_dev  = evaluate_policy(policy, val_env)
            val_returns.append(avg_portfolio - 1.0)  # Convert to return
            val_sharpes.append(avg_sharpe)
            val_drawdowns.append(avg_drawdown)

            logger.info(f"Episode {episode}/{max_episodes} | Train Reward: {episode_reward:.4f} | "
                  f"Val Return: {(avg_portfolio - 1.0) * 100:.2f}% | Val Sharpe: {avg_sharpe:.4f} | "
                  f"Val Max DD: {avg_drawdown * 100:.2f}% | Avg Std Dev: {avg_std_dev} | ")

            logger.info(f"Last Portfolio Values: {last_portfolio_value}% | Last Val Return: {(last_portfolio_value - 1.0) * 100:.2f}% | Last Sharpe Ratio: {last_sharpe_ratio} | Last Max Drawdown: {last_max_drawdown}% | Last Std Dev: {last_std_dev}")

            if avg_sharpe > best_val_sharpe:
                best_val_sharpe = avg_sharpe
                policy.save(f"{save_dir}/td3_best_model")
                logger.info(f"New best model saved with Sharpe ratio: {best_val_sharpe:.4f}")

        if episode % 100 == 0:
            policy.save(f"{save_dir}/td3_checkpoint_ep{episode}")
            logger.info(f"Checkpoint saved at episode {episode}")

    logger.info("Training complete. Saving final model and evaluating on test set.")

    policy.save(f"{save_dir}/td3_final_model")

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(val_returns)
    plt.title('Validation Returns')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(val_sharpes)
    plt.title('Validation Sharpe Ratios')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png")

    logger.info("\nEvaluating best model on test data...")
    policy.load(f"{save_dir}/td3_best_model")

    test_state = test_env.reset()
    test_done = False
    test_actions = []
    test_positions = []

    while not test_done:
        test_action = policy.select_action(np.array(test_state))
        test_actions.append(test_action[0])
        test_state, _, test_done, _ = test_env.step(test_action[0])
        test_positions.append(test_env.current_position)

    test_returns = np.array(test_env.portfolio_history[1:]) / np.array(test_env.portfolio_history[:-1]) - 1
    test_sharpe = np.mean(test_returns) / (np.std(test_returns) + 1e-8) * np.sqrt(252)
    test_portfolio = test_env.portfolio_value
    test_return = (test_portfolio - 1.0) * 100
    test_drawdown = test_env.max_drawdown * 100

    logger.info(f"Test Return: {test_return:.2f}%")
    logger.info(f"Test Sharpe Ratio: {test_sharpe:.4f}")
    logger.info(f"Test Max Drawdown: {test_drawdown:.2f}%")

    logger.info("Saving test results plots.")

    plt.figure(figsize=(15, 12))

    plt.subplot(3, 1, 1)
    plt.plot(test_env.portfolio_history)
    plt.title('Portfolio Value')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(test_positions)
    plt.title('Positions')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(test_actions)
    plt.title('Actions')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/test_results.png")

    test_env.render()

    logger.info("TD3 training process completed successfully.")

    return policy

def main():
    logger.info("----- Starting data fetch stage -----")

    pdo = PerformDataOperations(
        instrument="NAS100_USD",
        start="2023-01-01",
        end="2023-02-01",
        granularity="M5",
        years=1)

    combined_data = pdo.perform_chunking()

    logger.info("----- Data fetch stage completed -----")
    logger.info("----- Starting preprocessing fetch stage -----")

    ppd = PreprocessData(
        instrument="NAS100_USD",
        start="2023-01-01",
        end="2023-02-01",
        granularity="M5",
    )

    train_df, val_df, test_df, features = ppd.preprocess_data(combined_data=combined_data)

    logger.info("----- Preprocessing stage completed -----")

    logger.info("----- Starting Train/Val/Test stage -----")
    train_td3_for_trading(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        lookback_window=60,
        frame_stack=4,
        transaction_cost=0.0003,
        max_position=1.0,
        max_episodes=100,
        max_timesteps=150000,
        batch_size=512,
        discount=0.995,
        tau=0.0005,
        policy_noise=0.15,
        noise_clip=0.35,
        policy_freq=4,
        exploration_noise=0.08,
        eval_freq=10,
        save_dir='results'
    )
    logger.info("----- Train/Val/Test stage completed -----")


if __name__ == "__main__":
    main()