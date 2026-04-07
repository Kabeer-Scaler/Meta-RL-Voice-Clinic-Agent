"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API configuration
    api_base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-3.5-turbo"
    hf_token: str = ""
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 7860
    
    # Environment configuration
    max_concurrent_episodes: int = 10
    scenario_dir: str = "scenarios"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Global settings instance
settings = Settings()
