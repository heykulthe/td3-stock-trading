import time
import os
from datetime import datetime, timedelta
import logging
import tpqoa
import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nasdaq-fetcher")

INFLUXDB_TOKEN = os.environ.get("INFLUXDB_TOKEN")
INFLUXDB_ORG = "Hillbert Lab"
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_BUCKET = "dexblaze-web"

influx_client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)

INSTRUMENT = "NAS100_USD"
GRANULARITY = "S30"


def _filter_market_hours(data):
    logger.info("Filtering dataframe with relevant market hours")
    df = data.copy()
    df['day_of_week'] = df.index.dayofweek
    df['hour'] = df.index.hour

    trading_mask = (
        (df['day_of_week'] <= 4) &  # Monday to Friday
        (df['hour'] >= 13) & (df['hour'] <= 21)  # 13:00 to 21:00 UTC
    )

    filtered_df = df[trading_mask].copy()
    filtered_df.drop(['day_of_week', 'hour'], axis=1, inplace=True)

    return filtered_df


def _add_indicators(df):
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

    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    ma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['boll_upper'] = ma20 + 2 * std20
    df['boll_lower'] = ma20 - 2 * std20

    return df


def fetch_ohlc(instrument: str, start: str, end: str, granularity: str = "S30"):
    api = tpqoa.tpqoa("oanda.cfg")

    try:
        df_bid = api.get_history(instrument=instrument, start=start, end=end, granularity=granularity, price="B")
        df_ask = api.get_history(instrument=instrument, start=start, end=end, granularity=granularity, price="A")

        if df_bid.empty or df_ask.empty:
            logger.warning("No data returned.")
            return None

        df = pd.DataFrame(index=df_bid.index)
        df["open"] = (df_bid["o"] + df_ask["o"]) / 2
        df["high"] = (df_bid["h"] + df_ask["h"]) / 2
        df["low"] = (df_bid["l"] + df_ask["l"]) / 2
        df["close"] = (df_bid["c"] + df_ask["c"]) / 2
        df["spread"] = df_ask["c"] - df_bid["c"]
        df["volume"] = (df_bid["volume"] + df_ask["volume"]) / 2

        # Filter by market hours
        df = _filter_market_hours(df)

        # Add indicators
        df = _add_indicators(df)

        for col in df.columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        # Convert to InfluxDB points
        points = []
        for ts, row in df.iterrows():
            point = Point("nasdaq100_5min").tag("instrument", instrument).time(ts, WritePrecision.S)
            for col in df.columns:
                point.field(col, float(row[col]))
            points.append(point)

        return points

    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None


if __name__ == "__main__":
    end_date = "2025-04-15"
    start_date = "2025-02-01"

    logger.info(f"Fetching data from {start_date} to {end_date}...")
    data_points = fetch_ohlc(instrument=INSTRUMENT, granularity=GRANULARITY, start=start_date, end=end_date)

    if data_points:
        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=data_points)
        logger.info(f"Wrote {len(data_points)} data points.")
    else:
        logger.warning("No data written due to fetch failure.")
