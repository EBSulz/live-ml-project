from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    """Abstract base class for all financial data sources."""

    @abstractmethod
    async def fetch_ohlcv(
        self, symbol: str, interval: str, outputsize: int = 500
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data.

        Returns a DataFrame with columns:
        [timestamp, open, high, low, close, volume]
        sorted by timestamp ascending.
        """
        ...

    @abstractmethod
    async def fetch_latest(self, symbol: str, interval: str) -> dict:
        """Fetch the most recent OHLCV bar as a flat dictionary."""
        ...
