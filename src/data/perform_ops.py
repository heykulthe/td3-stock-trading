import gc
import os
import pandas as pd
import logging

from src.data.fetch_data import fetch_stock_data

logger = logging.getLogger("td3-stock-trading")


def _filter_market_hours(data):
    logger.info("Filtering dataframe with relevant market hours")
    df = data.copy()

    df['day_of_week'] = df.index.dayofweek
    df['hour'] = df.index.hour

    trading_mask = (
            (df['day_of_week'] <= 4) &
            (df['hour'] >= 13) & (df['hour'] <= 21)
    )

    filtered_df = df[trading_mask].copy()

    filtered_df.drop(['day_of_week', 'hour'], axis=1, inplace=True)

    return filtered_df


class PerformDataOperations:
    def __init__(self,
                 instrument : str,
                 start : str,
                 end : str,
                 granularity : str
        ):
        self.instrument = instrument
        self.start = start
        self.end = end
        self.granularity = granularity
        self.data_dir = "stock-data"
        os.makedirs(self.data_dir, exist_ok=True)


    def fetch_historical_data(self):
        logger.info(f"Fetching BID data from {self.start} to {self.end} for instrument: {self.instrument}")
        bid_data = fetch_stock_data(
            instrument=self.instrument,
            start=self.start,
            end=self.end,
            granularity=self.granularity,
            price="B"
        )
        bid_data = bid_data.add_suffix('_bid')

        logger.info(f"Fetching ASK data from {self.start} to {self.end} for instrument: {self.instrument}")
        ask_data = fetch_stock_data(
            instrument=self.instrument,
            start=self.start,
            end=self.end,
            granularity=self.granularity,
            price="A"
        )
        ask_data = ask_data.add_suffix('_ask')

        logger.info(f"Combining BID and ASK data, removing extra dataframes")
        combined_data = pd.concat([bid_data, ask_data], axis=1)

        del bid_data, ask_data
        gc.collect()

        logger.info(f"Add mid-price calculations (inplace operation)")

        combined_data['open'] = (combined_data['o_bid'] + combined_data['o_ask']) / 2
        combined_data['high'] = (combined_data['h_bid'] + combined_data['h_ask']) / 2
        combined_data['low'] = (combined_data['l_bid'] + combined_data['l_ask']) / 2
        combined_data['close'] = (combined_data['c_bid'] + combined_data['c_ask']) / 2
        combined_data['spread'] = combined_data['c_ask'] - combined_data['c_bid']

        combined_data = _filter_market_hours(combined_data)

        data_size_mb = combined_data.memory_usage(deep=True).sum() / (1024 * 1024)

        logger.info(f"Data size: {data_size_mb:.2f} MB with {len(combined_data)} rows")

        data_file = f"{self.data_dir}/{self.instrument}_{self.start}_{self.end}_{self.granularity}.csv"

        combined_data.to_csv(data_file)

        logger.info(f"Saved combined data to path: {data_file}")

        return combined_data

