from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Data Source ---
    data_source: str = "twelve_data"
    twelve_data_api_key: str = ""
    alpha_vantage_api_key: str = ""
    default_symbol: str = "BTC/USD"
    default_interval: str = "1h"

    # --- MLflow ---
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "signal-classification-v1"
    mlflow_live_experiment_name: str = "signal-classification-live"
    mlflow_log_interval: int = 100
    mlflow_model_name: str = "market-predictor"

    # --- Model quality thresholds ---
    min_f1_threshold: float = 0.60
    min_accuracy_threshold: float = 0.55
    drift_window: int = 500
    signal_threshold: float = 0.5

    # --- GCP ---
    gcp_project_id: str = ""
    gcp_region: str = "us-central1"
    gcs_model_bucket: str = ""
    gcs_dvc_bucket: str = ""
    vertex_ai_staging_bucket: str = ""

    # --- API ---
    app_env: str = "development"
    log_level: str = "INFO"
    port: int = 8080

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_gcp(self) -> bool:
        return self.app_env in ("production", "staging")


@lru_cache
def get_settings() -> Settings:
    return Settings()
