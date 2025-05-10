"""
Cryptocurrency Data Processor with Technical Indicators

A robust class for loading cryptocurrency data and computing technical indicators,
following Python best practices and error handling.
"""
import os
import pandas as pd
import pandas_ta as ta  # Using pandas_ta instead of ta for better maintenance
from openbb import obb
from typing import Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoDataProcessor:
    def __init__(
        self,
        symbol: str = 'BTC-USD',
        start_date: str = '2021-01-01',
        end_date: str = '2024-01-01',
        interval: str = '1d'
    ):
        """
        Initialize the data processor.
        
        Parameters:
        -----------
        symbol : str
            Cryptocurrency pair symbol (default: 'BTC-USD')
        start_date : str
            Start date in YYYY-MM-DD format (default: '2021-01-01')
        end_date : str
            End date in YYYY-MM-DD format (default: '2024-01-01')
        interval : str
            Data frequency ('1d', '1h', etc.) (default: '1d')
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.data: Optional[pd.DataFrame] = None

    def load_data(self) -> bool:
        """
        Loads OHLCV data using OpenBB.
        
        Returns:
        --------
        bool
            True if data was loaded successfully, False otherwise
        """
        try:
            df = obb.crypto.price.historical(
                symbol=self.symbol,
                provider='yfinance',
                start_date=self.start_date,
                end_date=self.end_date,
                interval=self.interval
            ).to_df()

            # Validate and clean data
            if df.empty:
                logger.error("Received empty DataFrame from OpenBB")
                return False

            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"
            
            # Standardize column names
            column_map = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            
            # Only rename columns that exist in the DataFrame
            df.rename(columns={k: v for k, v in column_map.items() if k in df.columns}, inplace=True)
            
            # Validate required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                logger.error(f"Missing required columns: {missing}")
                return False

            df.dropna(inplace=True)
            self.data = df
            logger.info(f"Successfully loaded data for {self.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            self.data = None
            return False

    def compute_indicators(self) -> bool:
        """
        Computes technical indicators and updates self.data.
        
        Returns:
        --------
        bool
            True if indicators were computed successfully, False otherwise
        """
        if self.data is None:
            logger.error("Data not loaded. Call load_data() first.")
            return False
        
        try:
            # Create a working copy
            df = self.data.copy()
            
            # Convert to lowercase for pandas_ta compatibility
            df.columns = [col.lower() for col in df.columns]
            
            # Calculate indicators
            
            # Momentum indicators
            df.ta.rsi(length=14, append=True)
            df.ta.stoch(append=True)
            
            # Trend indicators
            df.ta.sma(length=50, append=True)
            df.ta.ema(length=20, append=True)
            df.ta.adx(length=14, append=True)
            
            # Volatility indicator
            df.ta.atr(length=14, append=True)
            
            # Volume indicator
            df.ta.obv(append=True)
            
            # Convert column names back to title case for consistency
            df.columns = [col.title() for col in df.columns]
            
            # Drop rows with NaN values that might have been created by indicators
            df.dropna(inplace=True)
            
            if df.empty:
                logger.error("DataFrame became empty after indicator calculation")
                return False
                
            self.data = df
            logger.info("Successfully computed indicators")
            return True
            
        except Exception as e:
            logger.error(f"Error computing indicators: {str(e)}", exc_info=True)
            return False

    def get_data(self) -> Union[pd.DataFrame, None]:
        """
        Returns the processed DataFrame.
        
        Returns:
        --------
        pd.DataFrame or None
            The processed data if available, None otherwise
        """
        if self.data is None:
            logger.warning("No data available. Call load_data() first.")
        return self.data


def save_processed_data(processor: CryptoDataProcessor) -> Optional[str]:
    """
    Saves the processed data to a CSV file under 'cryptotrade/processed_data/'.
    """
    df = processor.get_data()
    if df is None:
        logger.error("No data to save.")
        return None

    save_dir = os.path.join(os.getcwd(), 'cryptotrade', 'processed_data')
    os.makedirs(save_dir, exist_ok=True)

    class_name = processor.__class__.__name__
    symbol_clean = processor.symbol.replace("-", "_")
    filename = f"{class_name}_{symbol_clean}_{processor.start_date}_to_{processor.end_date}.csv"
    filepath = os.path.join(save_dir, filename)

    try:
        df.to_csv(filepath)
        logger.info(f"Saved processed data to: {filepath}")
        print(f"\n✅ Saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save data: {str(e)}", exc_info=True)
        return None

if __name__ == "__main__":
    # Example usage
    processor = CryptoDataProcessor()
    
    if processor.load_data():
        if processor.compute_indicators():
            save_processed_data(processor)
            
            # Optionally, print the first few rows of the processed data
            df = processor.get_data()
            if df is not None:
                print("Data successfully processed:")
                print(df.head())
                print("\nAvailable columns:", df.columns.tolist())