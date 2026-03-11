"""
app/services/rag_service.py
Retrieval-Augmented Generation:
  1. Converte a pergunta em embedding
  2. Busca top-k chunks via pgvector (função search_chunks)
  3. Monta prompt de contexto
  4. Chama o modelo de inferência (DeepSeek via Azure OpenAI SDK)
  5. Retorna resposta + chunks de suporte (para transparência)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from openai import OpenAI

from app.models.database import get_conn
from app.services.embedding_service import EmbeddingService
from config.config import config


@dataclass
class ChunkResult:
    chunk_id: int
    document_id: int
    file_name: str
    file_path: str
    chunk_index: int
    page_number: Optional[int]
    content: str
    score: float


@dataclass
class RAGResponse:
    question: str
    answer: str
    chunks: List[ChunkResult] = field(default_factory=list)
    model_used: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None


_SYSTEM_PROMPT = """Você é um assistente especializado em análise de documentos institucionais.
Responda APENAS com base nas informações fornecidas nos trechos de contexto abaixo.
Se a informação não estiver nos trechos, diga claramente que não encontrou nos documentos.
Seja preciso, objetivo e cite os documentos de origem quando relevante."""


class RAGService:

    def __init__(self):
        self._embedder = EmbeddingService()
        self._client = OpenAI(
            base_url=config.inference.endpoint,
            api_key=config.inference.api_key,
        )

    def retrieve(self, question: str) -> List[ChunkResult]:
        """Busca os chunks mais relevantes para a pergunta."""
        query_vector = self._embedder.embed_query(question)
        # Converte para formato aceito pelo pgvector
        vector_str = "[" + ",".join(f"{v:.8f}" for v in query_vector) + "]"

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM search_chunks(%s::vector, %s, %s)",
                    (vector_str, config.pipeline.retrieval_top_k, config.pipeline.retrieval_min_score),
                )
                rows = cur.fetchall()

        return [
            ChunkResult(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                file_name=r["file_name"],
                file_path=r["file_path"],
                chunk_index=r["chunk_index"],
                page_number=r["page_number"],
                content=r["content"],
                score=float(r["score"]),
            )
            for r in rows
        ]

    def answer(self, question: str, model_override: str = "") -> RAGResponse:
        """Pipeline completo: retrieve → build prompt → infer."""
        model = model_override or config.inference.deployment_name

        # 1. Retrieve
        chunks = self.retrieve(question)

        if not chunks:
            return RAGResponse(
                question=question,
                answer="Não foram encontrados trechos relevantes nos documentos indexados para responder à sua pergunta.",
                chunks=[],
                model_used=model,
            )

        # 2. Montar contexto
        context_parts = []
        for i, c in enumerate(chunks, 1):
            page_info = f" (p. {c.page_number})" if c.page_number else ""
            context_parts.append(
                f"[Trecho {i} — {c.file_name}{page_info} | similaridade: {c.score:.2f}]\n{c.content}"
            )
        context_block = "\n\n---\n\n".join(context_parts)

        user_message = f"""CONTEXTO DOS DOCUMENTOS:
{context_block}

---

PERGUNTA: {question}"""

        # 3. Inferência
        try:
            completion = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            answer_text = completion.choices[0].message.content or ""
            usage = completion.usage

            return RAGResponse(
                question=question,
                answer=answer_text,
                chunks=chunks,
                model_used=model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            )

        except Exception as exc:
            return RAGResponse(
                question=question,
                answer="",
                chunks=chunks,
                model_used=model,
                error=str(exc),
            )