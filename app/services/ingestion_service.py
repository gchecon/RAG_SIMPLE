"""
app/services/ingestion_service.py
Pipeline de ingestão de PDFs:
  1. Varre diretório recursivamente
  2. Calcula SHA-256 de cada arquivo
  3. Ignora arquivos já indexados (deduplicação por hash)
  4. Extrai texto via pdfplumber
  5. Divide em chunks (LangChain RecursiveCharacterTextSplitter)
  6. Gera embeddings em lote
  7. Persiste no PostgreSQL/pgvector
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional

import pdfplumber
import psycopg2.extras
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.database import get_conn
from app.services.embedding_service import EmbeddingService
from config.config import config


# ── DTOs ─────────────────────────────────────────────────────────────────────

@dataclass
class IngestionProgress:
    """Emitido via generator para o caller acompanhar o progresso."""
    stage: str          # 'scanning' | 'hashing' | 'extracting' | 'embedding' | 'saving' | 'done' | 'error'
    file_name: str = ""
    message: str = ""
    files_found: int = 0
    files_new: int = 0
    files_skipped: int = 0
    files_error: int = 0
    chunks_created: int = 0


@dataclass
class IngestionResult:
    files_found: int = 0
    files_new: int = 0
    files_skipped: int = 0
    files_error: int = 0
    chunks_created: int = 0
    errors: List[str] = field(default_factory=list)
    log_id: Optional[int] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _scan_pdfs(root: Path) -> List[Path]:
    return sorted(root.rglob("*.pdf"))


def _extract_text(path: Path) -> tuple[str, int]:
    """Extrai texto completo e número de páginas via pdfplumber."""
    pages_text = []
    with pdfplumber.open(path) as pdf:
        n_pages = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
    return "\n\n".join(pages_text), n_pages


def _chunk_text(text: str) -> List[dict]:
    """Divide texto em chunks e retorna lista de dicts com content."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.pipeline.chunk_size,
        chunk_overlap=config.pipeline.chunk_overlap,
        separators=[
            config.pipeline.chunk_separator,
            "\n",
            ". ",
            " ",
            "",
        ],
    )
    docs = splitter.create_documents([text])
    return [{"content": d.page_content, "index": i} for i, d in enumerate(docs)]


# ── Serviço principal ─────────────────────────────────────────────────────────

class IngestionService:

    def __init__(self):
        self._embedder = EmbeddingService()

    def get_indexed_hashes(self) -> set[str]:
        """Retorna conjunto de hashes já presentes no banco."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT file_hash FROM documents")
                rows = cur.fetchall()
        return {r["file_hash"] for r in rows}

    def get_documents_summary(self) -> List[dict]:
        """Lista documentos indexados para exibição na UI."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT file_name, file_path, total_chunks,
                           ingested_at, embedding_model
                    FROM documents
                    ORDER BY ingested_at DESC
                """)
                return cur.fetchall()

    def run(self) -> Generator[IngestionProgress, None, IngestionResult]:
        """
        Executa o pipeline completo como generator.
        Permite streaming de progresso via SSE ou polling.
        """
        result = IngestionResult()
        log_id = self._start_log()
        result.log_id = log_id

        root = config.pipeline.pdf_root_dir
        yield IngestionProgress(stage="scanning", message=f"Varrendo {root}...")

        if not root.exists():
            msg = f"Diretório não encontrado: {root}"
            self._finish_log(log_id, result, status="error", error=msg)
            yield IngestionProgress(stage="error", message=msg)
            return result

        pdfs = _scan_pdfs(root)
        result.files_found = len(pdfs)
        yield IngestionProgress(
            stage="scanning",
            message=f"{len(pdfs)} PDFs encontrados.",
            files_found=len(pdfs),
        )

        indexed = self.get_indexed_hashes()

        for pdf_path in pdfs:
            try:
                yield IngestionProgress(
                    stage="hashing",
                    file_name=pdf_path.name,
                    message=f"Calculando hash: {pdf_path.name}",
                )
                file_hash = _sha256(pdf_path)

                if file_hash in indexed:
                    result.files_skipped += 1
                    yield IngestionProgress(
                        stage="hashing",
                        file_name=pdf_path.name,
                        message=f"Já indexado, ignorado: {pdf_path.name}",
                        files_skipped=result.files_skipped,
                    )
                    continue

                # ── Extração
                yield IngestionProgress(
                    stage="extracting",
                    file_name=pdf_path.name,
                    message=f"Extraindo texto: {pdf_path.name}",
                )
                full_text, n_pages = _extract_text(pdf_path)

                if not full_text.strip():
                    result.files_error += 1
                    result.errors.append(f"{pdf_path.name}: texto vazio (PDF escaneado?)")
                    yield IngestionProgress(
                        stage="error",
                        file_name=pdf_path.name,
                        message=f"Texto vazio (PDF escaneado?): {pdf_path.name}",
                    )
                    continue

                # ── Chunking
                chunks = _chunk_text(full_text)

                # ── Embedding em lote
                yield IngestionProgress(
                    stage="embedding",
                    file_name=pdf_path.name,
                    message=f"Gerando {len(chunks)} embeddings: {pdf_path.name}",
                )
                texts = [c["content"] for c in chunks]
                vectors = self._embedder.embed_texts(texts)

                # ── Persistência
                yield IngestionProgress(
                    stage="saving",
                    file_name=pdf_path.name,
                    message=f"Salvando no PostgreSQL: {pdf_path.name}",
                )
                doc_id = self._save_document(
                    file_hash=file_hash,
                    file_name=pdf_path.name,
                    file_path=str(pdf_path),
                    file_size=pdf_path.stat().st_size,
                    n_pages=n_pages,
                    n_chunks=len(chunks),
                )
                self._save_chunks(doc_id, chunks, vectors)

                indexed.add(file_hash)
                result.files_new += 1
                result.chunks_created += len(chunks)
                yield IngestionProgress(
                    stage="saving",
                    file_name=pdf_path.name,
                    message=f"Concluído: {pdf_path.name} ({len(chunks)} chunks)",
                    files_new=result.files_new,
                    chunks_created=result.chunks_created,
                )

            except Exception as exc:
                result.files_error += 1
                result.errors.append(f"{pdf_path.name}: {exc}")
                yield IngestionProgress(
                    stage="error",
                    file_name=pdf_path.name,
                    message=f"Erro: {pdf_path.name} — {exc}",
                    files_error=result.files_error,
                )

        self._finish_log(log_id, result, status="completed")
        yield IngestionProgress(
            stage="done",
            message=(
                f"Ingestão concluída. "
                f"Novos: {result.files_new} | "
                f"Ignorados: {result.files_skipped} | "
                f"Erros: {result.files_error} | "
                f"Chunks criados: {result.chunks_created}"
            ),
            files_found=result.files_found,
            files_new=result.files_new,
            files_skipped=result.files_skipped,
            files_error=result.files_error,
            chunks_created=result.chunks_created,
        )
        return result

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _save_document(
        self, file_hash, file_name, file_path, file_size, n_pages, n_chunks
    ) -> int:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents
                        (file_hash, file_name, file_path, file_size,
                         total_pages, total_chunks, embedding_model,
                         chunk_size, chunk_overlap)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        file_hash, file_name, file_path, file_size,
                        n_pages, n_chunks,
                        config.embedding.model,
                        config.pipeline.chunk_size,
                        config.pipeline.chunk_overlap,
                    ),
                )
                return cur.fetchone()["id"]

    def _save_chunks(self, doc_id: int, chunks: List[dict], vectors: List[List[float]]):
        rows = [
            (doc_id, c["index"], c["content"], v)
            for c, v in zip(chunks, vectors)
        ]
        with get_conn() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO chunks (document_id, chunk_index, content, embedding)
                    VALUES %s
                    ON CONFLICT (document_id, chunk_index) DO NOTHING
                    """,
                    rows,
                    template="(%s, %s, %s, %s::vector)",
                )

    def _start_log(self) -> int:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ingestion_log (pdf_root, status)
                    VALUES (%s, 'running') RETURNING id
                    """,
                    (str(config.pipeline.pdf_root_dir),),
                )
                return cur.fetchone()["id"]

    def _finish_log(self, log_id: int, result: IngestionResult, status: str, error: str = ""):
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE ingestion_log SET
                        finished_at   = NOW(),
                        files_found   = %s,
                        files_new     = %s,
                        files_skipped = %s,
                        files_error   = %s,
                        chunks_created= %s,
                        status        = %s,
                        error_message = %s
                    WHERE id = %s
                    """,
                    (
                        result.files_found, result.files_new,
                        result.files_skipped, result.files_error,
                        result.chunks_created, status,
                        error or None, log_id,
                    ),
                )