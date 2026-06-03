from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    pinecone_api_key: str
    tavily_api_key: str
    google_maps_api_key: str = ""
    redis_url: str = "redis://localhost:6379"
    firebase_sa_path: str = "./firebase-service-account.json"
    director_model: str = "gpt-4o-mini"
    research_model: str = "gpt-4o-mini"
    copywriter_model: str = "gpt-4o-mini"
    critic_model: str = "gpt-4o-mini"
    title_model: str = "gpt-4o-mini"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
