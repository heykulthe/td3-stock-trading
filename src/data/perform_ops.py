import gc
import glob
import os
import time

import pandas as pd
import logging
import datetime as dt

from dateutil.relativedelta import relativedelta

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

def _remove_chunks(instrument : str, data_dir="stock-data"):
    pattern = os.path.join(data_dir, f"{instrument}*.csv")
    for file_path in glob.glob(pattern):
        try:
            os.remove(file_path)
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")


class PerformDataOperations:
    def __init__(self,
                 instrument : str,
                 start : str,
                 end : str,
                 granularity : str,
                 years : int
        ):
        self.years = years
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

    def perform_chunking(self, granularity="M5"):
        years = self.years
        end_date = dt.datetime.now()

        if end_date.weekday() >= 5:
            days_to_subtract = end_date.weekday() - 4
            end_date -= dt.timedelta(days=days_to_subtract)

        start_date = end_date - relativedelta(years=years)

        current_start = start_date
        chunk_delta = relativedelta(months=3)

        all_chunks = []

        logger.info(f"Fetching {years} years of data in 3-month chunks for instrument: {self.instrument}")

        while current_start < end_date:
            current_end = min(current_start + chunk_delta, end_date)

            logger.info(f"Fetching chunk: {current_start.date()} to {current_end.date()}")

            try:
                self.start = current_start.strftime("%Y-%m-%d")
                self.end = current_end.strftime("%Y-%m-%d")
                self.granularity = granularity

                chunk_df = self.fetch_historical_data()

                if chunk_df is not None and not chunk_df.empty:
                    all_chunks.append(chunk_df)
                    logger.info(f"Finished chunk: {current_start.date()} to {current_end.date()} with {len(chunk_df)} rows")
                else:
                    logger.warning(f"Chunk {current_start.date()} to {current_end.date()} returned no data")

                time.sleep(1)
                del chunk_df
                gc.collect()

            except Exception as e:
                logger.error(f"Failed to fetch chunk {current_start.date()} to {current_end.date()}: {e}")

            current_start = current_end

        if all_chunks:
            final_data = pd.concat(all_chunks)
            data_file = f"{self.data_dir}/stock_data_{self.instrument}_{start_date.date()}_{end_date.date()}_{granularity}.csv"
            final_data.to_csv(data_file)

            logger.info(f"Saved full chunked data to path: {data_file}")
            logger.info(f"Final data shape: {final_data.shape}")


            logger.info(f"Removing chunks from {self.data_dir}")
            _remove_chunks(instrument=self.instrument, data_dir=self.data_dir)

            return final_data
        else:
            logger.error("No data fetched across all chunks, returning empty dataframe")
            return pd.DataFrame()



