import tpqoa
import pandas as pd
import logging

logger = logging.getLogger("td3-stock-trading")

def fetch_stock_data(instrument : str,
                     start : str,
                     end : str,
                     granularity : str ="5M",
                     price : str ="B"
    ):
    api = tpqoa.tpqoa('oanda.cfg')
    try:
        logger.info(f"Fetching price data of {instrument}")
        data = api.get_history(instrument=instrument,
                               start=start,
                               end=end,
                               granularity=granularity,
                               price=price
        )
        logger.debug(f"data has the type: {type(data)}")
        return data
    except Exception as e:
        logger.error(f"Unable to fetch stock data: {e}")