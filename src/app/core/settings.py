from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Class to manage application settings.

    This class uses Pydantic to load settings from environment variables.
    """

    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "ap-northeast-1"
    S3_BUCKET_NAME: str
    HUGGING_FACE_TOKEN: str

    class Config:
        """Internal class that defines the configuration of the Pydantic model."""

        env_file = ".env"


settings = Settings()  # type: ignore  # noqa: PGH003
