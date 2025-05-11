"""
Script: twitter_data.py
Purpose: Download and process Bitcoin-related Twitter datasets using Kaggle API with .env file.
"""

import os
import pandas as pd
import logging
from dotenv import load_dotenv

# Load environment variables from .env BEFORE importing Kaggle
load_dotenv()

# ✅ MUST be after loading .env
from kaggle.api.kaggle_api_extended import KaggleApi

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables manually (as fallback)
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

# Authenticate with Kaggle API
api = KaggleApi()
try:
    api.authenticate()
    logger.info("Kaggle API authenticated successfully.")
except Exception as e:
    logger.error(f"Failed to authenticate with Kaggle API: {e}")
    exit(1)

# Define Twitter datasets
TWITTER_DATASETS = {
    "bitcoin-tweets-alaix14": "alaix14/bitcoin-tweets-20160101-to-20190329",
    "bitcoin-tweets-kaushik": "kaushiksuresh147/bitcoin-tweets",
    "bitcoin-tweets-hirad": "hiraddolatzadeh/bitcoin-tweets-2021-2022",
    "sentiment140": "kazanova/sentiment140"
}

DATA_DIR = os.path.join("data", "twitter")
os.makedirs(DATA_DIR, exist_ok=True)

def download_and_extract(dataset_name: str, kaggle_ref: str):
    logger.info(f"Downloading {dataset_name} from Kaggle...")
    try:
        api.dataset_download_files(kaggle_ref, path=DATA_DIR, unzip=True)
        logger.info(f"{dataset_name} downloaded and extracted.")
    except Exception as e:
        logger.error(f"Failed to download {dataset_name}: {e}")

def collect_twitter_data() -> pd.DataFrame:
    for name, ref in TWITTER_DATASETS.items():
        download_and_extract(name, ref)

    combined = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            file_path = os.path.join(DATA_DIR, file)
            logger.info(f"Attempting to load {file}...")

            df = None
            for encoding in ["utf-8", "ISO-8859-1", "latin1"]:
                try:
                    df = pd.read_csv(
                        file_path,
                        encoding=encoding,
                        low_memory=False,
                        on_bad_lines='skip'  # requires pandas >= 1.3
                    )
                    logger.info(f"Loaded {file} with encoding {encoding}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {file} with {encoding}: {e}")

            if df is None or df.empty:
                logger.warning(f"Skipping {file} — failed all encoding attempts or empty.")
                continue

            try:
                date_col = next((col for col in df.columns if "date" in col.lower() or "created" in col.lower()), None)
                if date_col:
                    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
                    df = df.dropna(subset=["date"])
                    combined.append(df)
                    logger.info(f"Parsed date column '{date_col}' in {file}")
                else:
                    logger.warning(f"No date column found in {file}")
            except Exception as e:
                logger.warning(f"Failed date parsing in {file}: {e}")

    if combined:
        final_df = pd.concat(combined, ignore_index=True)
        final_df = final_df.sort_values("date")
        return final_df
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    df = collect_twitter_data()

    if not df.empty:
        output_path = os.path.join(DATA_DIR, "bitcoin_twitter_combined.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Combined Twitter dataset saved to {output_path}")

        print("\n📌 First 5 rows of combined dataset:")
        print(df.head())
    else:
        logger.warning("⚠️ No valid Twitter data processed.")
