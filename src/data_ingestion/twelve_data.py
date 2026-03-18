import logging

import httpx
import pandas as pd

from src.config.settings import get_settings
from src.data_ingestion.base import DataSource

logger = logging.getLogger(__name__)

BASE_URL = "https://api.twelvedata.com"


class TwelveDataSource(DataSource):
    """Twelve Data API client implementing the DataSource interface."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or get_settings().twelve_data_api_key

    def _build_params(
        self, symbol: str, interval: str, outputsize: int | None = None
    ) -> dict:
        params: dict = {
            "symbol": symbol,
            "interval": interval,
            "apikey": self._api_key,
            "format": "JSON",
        }
        if outputsize is not None:
            params["outputsize"] = outputsize
        return params

    @staticmethod
    def _normalize(raw_values: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(raw_values)
        df = df.rename(columns={"datetime": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "volume" not in df.columns:
            df["volume"] = 0
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = df["volume"].fillna(0)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        return df.sort_values("timestamp").reset_index(drop=True)

    async def fetch_ohlcv(
        self, symbol: str, interval: str, outputsize: int = 500
    ) -> pd.DataFrame:
        params = self._build_params(symbol, interval, outputsize)
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{BASE_URL}/time_series", params=params)
            resp.raise_for_status()
            data = resp.json()

        if "values" not in data:
            raise ValueError(
                f"Twelve Data error: {data.get('message', 'unknown error')}"
            )
        logger.info("Fetched %d bars for %s", len(data["values"]), symbol)
        return self._normalize(data["values"])

    async def fetch_latest(self, symbol: str, interval: str) -> dict:
        params = self._build_params(symbol, interval, outputsize=1)
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{BASE_URL}/time_series", params=params)
            resp.raise_for_status()
            data = resp.json()

        if "values" not in data or len(data["values"]) == 0:
            raise ValueError(
                f"Twelve Data error: {data.get('message', 'no data returned')}"
            )
        row = data["values"][0]
        return {
            "timestamp": pd.Timestamp(row["datetime"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0)),
        }
