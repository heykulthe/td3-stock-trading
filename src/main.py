from src.utils.logger import setup_logging
from src.data.perform_ops import PerformDataOperations
logger = setup_logging()

def main():
    logger.info("----- Starting data fetch stage -----")

    pdo = PerformDataOperations(
        instrument="NAS100_USD",
        start="2023-01-01",
        end="2023-02-01",
        granularity="M5",
        years=2)
    combined_data = pdo.perform_chunking()

    logger.info("----- Data fetch stage completed -----")

if __name__ == "__main__":
    main()