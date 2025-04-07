from src.utils.logger import setup_logging
from src.data.perform_ops import PerformDataOperations
from src.data.preprocess import PreprocessData
logger = setup_logging()

def main():
    logger.info("----- Starting data fetch stage -----")

    pdo = PerformDataOperations(
        instrument="NAS100_USD",
        start="2023-01-01",
        end="2023-02-01",
        granularity="M5",
        years=5)

    combined_data = pdo.perform_chunking()

    logger.info("----- Data fetch stage completed -----")
    logger.info("----- Starting preprocessing fetch stage -----")

    ppd = PreprocessData(
        instrument="NAS100_USD",
        start="2023-01-01",
        end="2023-02-01",
        granularity="M5",
    )

    normalized_data, features = ppd.preprocess_data(combined_data=combined_data)

    logger.info("----- Preprocessing stage completed -----")

if __name__ == "__main__":
    main()