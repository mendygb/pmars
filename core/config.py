from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    pinecone_api_key: str
    tavily_api_key: str
    google_maps_api_key: str = ""
    redis_url: str = "redis://localhost:6379"
    firebase_sa_path: str = "./firebase-service-account.json"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
