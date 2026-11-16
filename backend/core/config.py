"""Application configuration loaded from environment variables.

Never hardcode API keys or secrets here. All values are read from the
environment (or a `.env` file) via pydantic-settings. Copy
``.env.example`` to ``.env`` and adjust for your deployment.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ------------------------------------------------------------------ #
    # Application
    # ------------------------------------------------------------------ #
    APP_NAME: str = "Customer Feedback Intelligence"
    APP_VERSION: str = "0.1.0"
    LOG_LEVEL: str = "INFO"

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    DATA_DIR: str = "data"
    DATABASE_PATH: str = "data/feedback.db"
    INDEX_DIR: str = "backend/index"
    INDEX_PATH: str = "backend/index/vector_index.bin"
    RECORDS_PATH: str = "backend/index/records.json"
    SAMPLE_DATA_PATH: str = "data/sample_feedback.csv"
    EVALUATION_DATASET_PATH: str = "data/evaluation_dataset.jsonl"
    UPLOAD_DIRECTORY: str = "data/uploads"

    # ------------------------------------------------------------------ #
    # Ollama (used for both embeddings and the RAG LLM)
    # ------------------------------------------------------------------ #
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    OLLAMA_LLM_MODEL: str = "llama3.2:3b"

    # ------------------------------------------------------------------ #
    # RAG
    # ------------------------------------------------------------------ #
    RAG_TOP_K: int = 5
    RAG_TEMPERATURE: float = 0.2

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
