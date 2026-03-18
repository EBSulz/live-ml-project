import logging

import httpx
import pandas as pd

from src.config.settings import get_settings
from src.data_ingestion.base import DataSource

logger = logging.getLogger(__name__)

BASE_URL = "https://www.alphavantage.co/query"

_INTERVAL_MAP = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "1h": "60min",
    "60min": "60min",
}

_DAILY_INTERVALS = {"1d", "1day", "daily"}


class AlphaVantageSource(DataSource):
    """Alpha Vantage API client implementing the DataSource interface."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or get_settings().alpha_vantage_api_key

    def _is_crypto(self, symbol: str) -> bool:
        return "/" in symbol

    def _build_params(self, symbol: str, interval: str) -> dict:
        is_crypto = self._is_crypto(symbol)
        is_daily = interval in _DAILY_INTERVALS

        if is_crypto:
            from_sym, to_sym = symbol.split("/")
            if is_daily:
                return {
                    "function": "DIGITAL_CURRENCY_DAILY",
                    "symbol": from_sym,
                    "market": to_sym,
                    "apikey": self._api_key,
                }
            return {
                "function": "CRYPTO_INTRADAY",
                "symbol": from_sym,
                "market": to_sym,
                "interval": _INTERVAL_MAP.get(interval, interval),
                "apikey": self._api_key,
                "outputsize": "full",
            }

        if is_daily:
            return {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "apikey": self._api_key,
                "outputsize": "full",
            }
        return {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": _INTERVAL_MAP.get(interval, interval),
            "apikey": self._api_key,
            "outputsize": "full",
        }

    @staticmethod
    def _extract_series(data: dict) -> list[dict]:
        """Find the time series key in the AV response and normalize rows."""
        series_key = None
        for key in data:
            if "Time Series" in key or "time series" in key.lower():
                series_key = key
                break
        if series_key is None:
            note = data.get("Note", data.get("Information", ""))
            raise ValueError(f"Alpha Vantage error: no time series found. {note}")

        rows = []
        for ts_str, values in data[series_key].items():
            row = {"timestamp": ts_str}
            for k, v in values.items():
                clean_key = k.split(". ", 1)[-1].lower().strip()
                if "open" in clean_key:
                    row["open"] = v
                elif "high" in clean_key:
                    row["high"] = v
                elif "low" in clean_key:
                    row["low"] = v
                elif "close" in clean_key and "adj" not in clean_key:
                    row["close"] = v
                elif "volume" in clean_key:
                    row["volume"] = v
            rows.append(row)
        return rows

    @staticmethod
    def _normalize(rows: list[dict], outputsize: int | None = None) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "volume" not in df.columns:
            df["volume"] = 0
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = df["volume"].fillna(0)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("timestamp").reset_index(drop=True)
        if outputsize is not None:
            df = df.tail(outputsize).reset_index(drop=True)
        return df

    async def fetch_ohlcv(
        self, symbol: str, interval: str, outputsize: int = 500
    ) -> pd.DataFrame:
        params = self._build_params(symbol, interval)
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        rows = self._extract_series(data)
        logger.info("Fetched %d bars for %s", len(rows), symbol)
        return self._normalize(rows, outputsize)

    async def fetch_latest(self, symbol: str, interval: str) -> dict:
        df = await self.fetch_ohlcv(symbol, interval, outputsize=1)
        if df.empty:
            raise ValueError("Alpha Vantage returned no data")
        row = df.iloc[-1]
        return {
            "timestamp": row["timestamp"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
