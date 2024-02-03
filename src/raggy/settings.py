from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGGY_",
        env_file=".env",
        extra="allow",
        validate_assignment=True,
    )


settings = Settings()
