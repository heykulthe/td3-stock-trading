import os
import time

import pandas as pd
import numpy as np
import logging

from src.data.dr import DimensionReducer

logger = logging.getLogger("td3-stock-trading")


def apply_dimension_reduction(data, method='pca', n_components=20):
    reducer = DimensionReducer(method=method, n_components=n_components)
    transformed_data = reducer.fit_transform(data)
    return transformed_data, reducer

class PreprocessData:
    def __init__(self,
                 instrument: str,
                 start: str,
                 end: str,
                 granularity: str,
                 output_dir="processed-data",
        ):
        self.instrument = instrument
        self.start = start
        self.end = end
        self.granularity = granularity
        self.features = []
        self.data_dir = output_dir
        os.makedirs(self.data_dir, exist_ok=True)


    def preprocess_data(self, combined_data):
        df = combined_data.copy()

        # logger.info("Convert time column to datetime")
        # df['time'] = pd.to_datetime(df['time'])
        # df.sort_values("time", inplace=True)
        # df.reset_index(drop=True, inplace=True)

        logger.info(
            "Performing feature engineering (returns / ma5 / ma10 / volatility / momentum / RSI / MACD / Bollinger Bands)")

        df['return'] = df['close'].pct_change().fillna(0)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)

        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()

        df['ma_norm'] = df['close'] / df['ma5']

        df['cumulative_return'] = (1 + df['return']).cumprod() - 1

        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()

        df['natr'] = (df['atr'] / df['close']) * 100

        df['dpo'] = df['close'].shift(11) - df['close'].rolling(window=21).mean().shift(1)
        df['dpo'] = df['dpo'].fillna(0)

        df['volatility'] = df['close'].rolling(window=10).std()

        df['momentum'] = df['close'] - df['close'].shift(5)
        df['momentum'] = df['momentum'].fillna(0)

        window_length = 14
        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(window=window_length, min_periods=1).mean()
        ma_down = down.rolling(window=window_length, min_periods=1).mean()
        rs = ma_up / (ma_down + 1e-6)
        df['rsi'] = 100 - (100 / (1 + rs))

        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        ma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        df['boll_upper'] = ma20 + 2 * std20
        df['boll_lower'] = ma20 - 2 * std20

        logger.info("Applying Min-Max normalization to numerical features")

        features_to_normalize = [
            # 'o_bid', 'h_bid', 'l_bid', 'c_bid', 'volume_bid',
            # 'o_ask', 'h_ask', 'l_ask', 'c_ask', 'volume_ask',
            'open', 'high', 'low', 'close', 'spread',
            'return', 'log_return', 'ma5', 'ma10', 'volatility', 'momentum',
            'macd', 'macd_signal', 'boll_upper', 'boll_lower', 'atr', 'dpo',
            'cumulative_return'
            # Note: we exclude features already in percentage/normalized form like 'rsi', 'natr', 'ma_norm'
        ]

        df.drop(columns=['o_bid', 'h_bid', 'l_bid', 'c_bid', 'volume_bid',
                         'o_ask', 'h_ask', 'l_ask', 'c_ask', 'volume_ask', 'complete_bid', 'complete_ask'],
                inplace=True, axis=1)

        total_rows = len(df)
        train_end = int(0.7 * total_rows)
        val_end = int(0.85 * total_rows)

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        # Apply scaling only to the training set
        min_max_scaler = {}
        for feature in features_to_normalize:
            min_val = train_df[feature].min()
            max_val = train_df[feature].max()
            min_max_scaler[feature] = (min_val, max_val)

            if min_val == max_val:
                train_df[feature] = 0
            else:
                train_df[feature] = (train_df[feature] - min_val) / (max_val - min_val)

        # Apply the same scaling to validation and test sets
        for feature in features_to_normalize:
            min_val, max_val = min_max_scaler[feature]

            if min_val == max_val:
                val_df[feature] = 0
                test_df[feature] = 0
            else:
                val_df[feature] = (val_df[feature] - min_val) / (max_val - min_val)
                test_df[feature] = (test_df[feature] - min_val) / (max_val - min_val)

        # Fill NaN values with median values
        train_df = train_df.fillna(train_df.median())
        val_df = val_df.fillna(val_df.median())
        test_df = test_df.fillna(test_df.median())

        logger.info(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")

        data_file = f"{self.data_dir}/normalized_{self.instrument}_{self.start}_{self.end}_{self.granularity}"
        train_df.to_csv(data_file + "_train.csv")
        val_df.to_csv(data_file + "_val.csv")
        test_df.to_csv(data_file + "_test.csv")

        logger.info(f"Saved full chunked data to path: {data_file}_train/_val/_test")

        features = train_df.columns

        return train_df, val_df, test_df, features


# normalized_df = df.copy()
# for feature in features_to_normalize:
#     min_val = df[feature].min()
#     max_val = df[feature].max()
#     if min_val == max_val:
#         normalized_df[feature] = 0
#     else:
#         normalized_df[feature] = (df[feature] - min_val) / (max_val - min_val)
#
#
# logger.info(f"Filling NaN values with median values")
# normalized_df = normalized_df.fillna(normalized_df.median())
#
# data_file = f"{self.data_dir}/normalized_{self.instrument}_{self.start}_{self.end}_{self.granularity}.csv"
# normalized_df.to_csv(data_file)
#
# logger.info(f"Saved full chunked data to path: {data_file}")
# logger.info(f"Final data shape: {normalized_df.shape}")
# features = normalized_df.columns
#
# # transformed_data, reducer = apply_dimension_reduction(data=normalized_df, method='pca', n_components=20)
#
# return normalized_df, features