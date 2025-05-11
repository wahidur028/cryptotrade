"""
Cryptocurrency News Collector

A robust class for collecting news articles related to cryptocurrencies from multiple sources,
supporting CSV and optional SQL storage. Follows Python best practices and logging.
"""

import os
import logging
import pandas as pd
from typing import Dict, List, Optional, Callable
from duckduckgo_search import DDGS
from openbb import obb
from sqlalchemy.engine.base import Engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DuckDuckGoNewsSource:
    def __init__(self, timelimit="m", safesearch="off"):
        self.timelimit = timelimit
        self.safesearch = safesearch
        self.client = DDGS()

    def search(self, keyword: str, max_results: int = 100) -> List[Dict]:
        try:
            return list(
                self.client.news(
                    keywords=keyword,
                    safesearch=self.safesearch,
                    timelimit=self.timelimit,
                    max_results=max_results,
                )
            )
        except Exception as e:
            logger.error(f"[ERROR] DuckDuckGoNewsSource failed: {e}")
            return []

class OpenBBNewsSource:
    def __init__(self, provider: str = "benzinga"):
        self.provider = provider

    def search(self, symbol: Optional[str], max_results: int = 100) -> List[Dict]:
        if not symbol:
            logger.info(f"[INFO] Skipping OpenBB search — no symbol provided.")
            return []

        try:
            news_data = obb.news.company(
                symbol=symbol,
                limit=max_results,
                provider=self.provider,
                order="desc"
            ).to_df()

            return news_data.to_dict(orient="records")
        except Exception as e:
            logger.error(f"[ERROR] OpenBBNewsSource failed for symbol '{symbol}': {e}")
            return []

class NewsCollector:
    def __init__(
        self,
        keyword_map: Dict[str, Optional[str]],
        sources: List[Callable],
        max_results: int = 100,
    ):
        self.keyword_map = keyword_map
        self.sources = sources
        self.max_results = max_results
        self.news_df: Optional[pd.DataFrame] = None

    def run(self) -> bool:
        logger.info("Starting news collection pipeline...")
        all_data = []

        for keyword, ticker in self.keyword_map.items():
            for source in self.sources:
                try:
                    logger.info(f"Fetching '{keyword}' from {source.__class__.__name__}")
                    results = source.search(ticker or keyword, self.max_results)
                    df = pd.DataFrame.from_records(results)

                    if df.empty:
                        continue

                    if 'date' in df.columns:
                        df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    elif 'published' in df.columns:
                        df["date"] = pd.to_datetime(df["published"], errors="coerce")
                    elif 'datetime' in df.columns:
                        df["date"] = pd.to_datetime(df["datetime"], errors="coerce")
                    else:
                        df["date"] = pd.NaT

                    df["ticker"] = ticker
                    df["keyword"] = keyword
                    df["source"] = source.__class__.__name__

                    all_data.append(df)
                except Exception as e:
                    logger.warning(f"Failed to fetch from {source.__class__.__name__}: {e}")

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df.dropna(subset=["date"], inplace=True)
            combined_df.sort_values(by="date", ascending=False, inplace=True)
            self.news_df = combined_df
            logger.info("News collection completed successfully.")
            return True

        logger.error("No news data collected.")
        return False

    def save_csv(self, folder: str = "cryptotrade/collected_news") -> Optional[str]:
        if self.news_df is None or self.news_df.empty:
            logger.error("No data to save.")
            return None

        os.makedirs(folder, exist_ok=True)
        filename = f"crypto_news_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"
        filepath = os.path.join(folder, filename)

        try:
            self.news_df.to_csv(filepath, index=False)
            logger.info(f"News saved to CSV: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save CSV: {str(e)}", exc_info=True)
            return None

    def save_sql(self, engine: Engine, table_name: str = "news") -> bool:
        if self.news_df is None or self.news_df.empty:
            logger.error("No data to save to SQL.")
            return False
        try:
            self.news_df.to_sql(table_name, engine, if_exists="replace", index=False)
            logger.info(f"News saved to SQL table: {table_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save to SQL: {str(e)}", exc_info=True)
            return False

    def get_data(self) -> Optional[pd.DataFrame]:
        return self.news_df

if __name__ == "__main__":
    # Define keywords and their corresponding ticker symbols
    keyword_map = {
        "Cryptocurrency": None,   # General keyword for DuckDuckGo
        "Bitcoin": "BTC",
        "Ethereum": "ETH",
        "Solana": "SOL"
    }

    # Initialize news sources
    duckduckgo_source = DuckDuckGoNewsSource()
    openbb_source = OpenBBNewsSource(provider="benzinga")  # Or "yfinance", "fmp", etc.

    # Initialize the collector with multiple sources
    collector = NewsCollector(
        keyword_map=keyword_map,
        sources=[duckduckgo_source, openbb_source],
        max_results=100
    )

    # Run the collection process
    if collector.run():
        csv_path = collector.save_csv()
        print(f"\n✅ News saved to: {csv_path}")

        # Optional: Save to SQLite
        from sqlalchemy import create_engine
        engine = create_engine("sqlite:///crypto_news.db")
        collector.save_sql(engine)






