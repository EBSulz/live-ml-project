from src.data_ingestion.alpha_vantage import AlphaVantageSource
from src.data_ingestion.base import DataSource
from src.data_ingestion.twelve_data import TwelveDataSource

_REGISTRY: dict[str, type[DataSource]] = {
    "twelve_data": TwelveDataSource,
    "alpha_vantage": AlphaVantageSource,
}


def get_data_source(source_name: str | None = None) -> DataSource:
    """Factory that returns a DataSource instance by name.

    Falls back to the configured default if *source_name* is ``None``.
    """
    if source_name is None:
        from src.config.settings import get_settings

        source_name = get_settings().data_source

    cls = _REGISTRY.get(source_name)
    if cls is None:
        available = ", ".join(_REGISTRY)
        raise ValueError(
            f"Unknown data source '{source_name}'. Available: {available}"
        )
    return cls()
