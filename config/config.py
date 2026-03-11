"""
config.py — Centraliza toda a configuração do ambiente RAG TCE.
Carrega o .env, valida campos obrigatórios e expõe um objeto Config.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega .env a partir da raiz do projeto
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


def _require(key: str) -> str:
    """Lê variável obrigatória; levanta erro descritivo se ausente."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"[config] Variável obrigatória '{key}' não encontrada no .env"
        )
    return value


class _InferenceConfig:
    endpoint: str = _require("AZURE_ENDPOINT")
    api_key: str = _require("AZURE_API_KEY")
    model_name: str = _require("AZURE_MODEL_NAME")
    deployment_name: str = _require("AZURE_DEPLOYMENT_NAME")


class _EmbeddingConfig:
    provider: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    # Campos opcionais para Azure OpenAI embedding
    azure_endpoint: str = os.getenv("EMBEDDING_AZURE_ENDPOINT", "")
    azure_key: str = os.getenv("EMBEDDING_AZURE_KEY", "")
    azure_deployment: str = os.getenv("EMBEDDING_AZURE_DEPLOYMENT", "")


class _DatabaseConfig:
    host: str = os.getenv("PG_HOST", "localhost")
    port: int = int(os.getenv("PG_PORT", "5432"))
    database: str = os.getenv("PG_DATABASE", "rag_tce")
    user: str = os.getenv("PG_USER", "postgres")
    password: str = _require("PG_PASSWORD")

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class _PipelineConfig:
    pdf_root_dir: Path = Path(os.getenv("PDF_ROOT_DIR", "./pdfs"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    chunk_separator: str = os.getenv("CHUNK_SEPARATOR", "\n\n")
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    retrieval_min_score: float = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.30"))


class _FlaskConfig:
    secret_key: str = os.getenv("FLASK_SECRET_KEY", "dev-insecure-key")
    port: int = int(os.getenv("FLASK_PORT", "5000"))
    debug: bool = os.getenv("FLASK_DEBUG", "True").lower() == "true"


class Config:
    """Ponto único de acesso a toda a configuração da aplicação."""
    inference = _InferenceConfig()
    embedding = _EmbeddingConfig()
    database = _DatabaseConfig()
    pipeline = _PipelineConfig()
    flask = _FlaskConfig()


config = Config()