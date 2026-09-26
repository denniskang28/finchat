from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/semantic_rag"
    upload_dir: Path = Path("data/uploads")
    cors_origins: str = "http://localhost:5173"
    table_repair_enabled: bool = True
    table_parse_mode: str = "llm_primary"
    table_parse_workers: int = 4
    table_repair_provider: str = "alibaba"
    table_repair_timeout_seconds: float = 90.0
    dashscope_api_key: str = ""
    alibaba_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    alibaba_vision_model: str = "qwen3.8-flash"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_vision_model: str = "deepseek-flash"
    openai_compatible_api_key: str = ""
    openai_compatible_base_url: str = ""
    openai_compatible_vision_model: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        return [value.strip() for value in self.cors_origins.split(",") if value.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
