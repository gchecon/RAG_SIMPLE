"""
app/routes/api.py — Endpoints REST da aplicação RAG TCE
"""
import json
from flask import Blueprint, Response, jsonify, request, stream_with_context

from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGService

api = Blueprint("api", __name__, url_prefix="/api")

_ingestion = IngestionService()
_rag = RAGService()


# ── Ingestão ──────────────────────────────────────────────────────────────────

@api.route("/ingest/start", methods=["POST"])
def ingest_start():
    """
    Inicia o pipeline de ingestão com streaming SSE.
    O frontend abre um EventSource em /api/ingest/stream.
    Esta rota apenas aciona; o progresso vem via /stream.
    """
    return jsonify({"status": "ok", "message": "Use /api/ingest/stream para acompanhar."})


@api.route("/ingest/stream")
def ingest_stream():
    """Server-Sent Events: progresso da ingestão em tempo real."""
    def generate():
        svc = IngestionService()
        for progress in svc.run():
            data = {
                "stage": progress.stage,
                "file": progress.file_name,
                "message": progress.message,
                "files_found": progress.files_found,
                "files_new": progress.files_new,
                "files_skipped": progress.files_skipped,
                "files_error": progress.files_error,
                "chunks_created": progress.chunks_created,
            }
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@api.route("/documents")
def list_documents():
    """Lista documentos já indexados."""
    docs = _ingestion.get_documents_summary()
    return jsonify([dict(d) for d in docs])


# ── Chat / RAG ────────────────────────────────────────────────────────────────

@api.route("/chat", methods=["POST"])
def chat():
    """
    Body JSON: { "question": "...", "model": "tce-DeepSeek-R1" }
    Retorna resposta + chunks de suporte.
    """
    body = request.get_json(force=True)
    question = (body.get("question") or "").strip()
    model = (body.get("model") or "").strip()

    if not question:
        return jsonify({"error": "Campo 'question' obrigatório."}), 400

    response = _rag.answer(question, model_override=model)

    return jsonify({
        "question": response.question,
        "answer": response.answer,
        "model_used": response.model_used,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "error": response.error,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "file_name": c.file_name,
                "chunk_index": c.chunk_index,
                "page_number": c.page_number,
                "score": round(c.score, 4),
                "content": c.content,
            }
            for c in response.chunks
        ],
    })


# ── Status ────────────────────────────────────────────────────────────────────

@api.route("/status")
def status():
    """Health check e contagens gerais."""
    from app.models.database import get_conn
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM documents")
            n_docs = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(*) AS n FROM chunks")
            n_chunks = cur.fetchone()["n"]

    from config.config import config
    return jsonify({
        "status": "ok",
        "documents_indexed": n_docs,
        "chunks_indexed": n_chunks,
        "embedding_model": config.embedding.model,
        "inference_model": config.inference.deployment_name,
        "retrieval_top_k": config.pipeline.retrieval_top_k,
    })