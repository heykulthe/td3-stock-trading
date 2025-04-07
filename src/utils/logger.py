import json
import logging.config
import os
from pathlib import Path
from datetime import datetime


def setup_logging():
    config_file = Path("log-config.json")
    with open(config_file) as f:
        log_config = json.load(f)

    filename = f"log-{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    log_path = Path("logs") / filename
    os.makedirs(log_path.parent, exist_ok=True)
    log_config['handlers']['file']['filename'] = f"logs/{filename}"

    logging.config.dictConfig(log_config)

    logger = logging.getLogger("td3-stock-trading")
    logger.info("Logging setup successful")
    logger.info(f"Logs being saved to - log/{filename}")

    return logger