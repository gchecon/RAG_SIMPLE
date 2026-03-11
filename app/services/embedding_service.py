"""
app/services/embedding_service.py
Abstração de embedding que suporta dois providers:
  - huggingface : sentence-transformers local (padrão, gratuito, multilingual)
  - azure_openai: text-embedding-ada-002 via Azure
"""
from __future__ import annotations

import threading
from typing import List

from config.config import config


class EmbeddingService:
    """Singleton thread-safe para geração de embeddings."""

    _instance: "EmbeddingService | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "EmbeddingService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._provider = config.embedding.provider
        self._model_name = config.embedding.model
        self._dimension = config.embedding.dimension
        self._model = None
        self._initialized = True

    def _load_model(self):
        """Carrega o modelo na primeira chamada (lazy loading)."""
        if self._model is not None:
            return

        if self._provider == "huggingface":
            from sentence_transformers import SentenceTransformer
            print(f"[Embedding] Carregando modelo local: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            print("[Embedding] Modelo carregado.")

        elif self._provider == "azure_openai":
            from openai import AzureOpenAI
            self._model = AzureOpenAI(
                azure_endpoint=config.embedding.azure_endpoint,
                api_key=config.embedding.azure_key,
                api_version="2024-02-01",
            )
        else:
            raise ValueError(f"EMBEDDING_PROVIDER inválido: '{self._provider}'")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Gera embeddings para uma lista de textos.
        Retorna lista de vetores (List[float]).
        """
        self._load_model()

        if self._provider == "huggingface":
            vectors = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,  # cosine similarity direta
            )
            return [v.tolist() for v in vectors]

        elif self._provider == "azure_openai":
            response = self._model.embeddings.create(
                input=texts,
                model=config.embedding.azure_deployment,
            )
            return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        """Conveniência para embedding de uma única query."""
        return self.embed_texts([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension