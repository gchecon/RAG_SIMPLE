-- ============================================================
-- RAG TCE — Schema PostgreSQL + pgvector
-- Execute como superusuário ou owner do banco rag_tce
-- ============================================================

-- Extensão vetorial (requer pgvector instalado)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm; -- para busca textual auxiliar

-- ------------------------------------------------------------
-- documents — controle de arquivos já ingeridos
-- O file_hash (SHA-256) é a chave de deduplicação.
-- Um mesmo arquivo movido de diretório não é reprocessado.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id            SERIAL PRIMARY KEY,
    file_hash     CHAR(64)     NOT NULL UNIQUE,   -- SHA-256 hex
    file_name     TEXT         NOT NULL,
    file_path     TEXT         NOT NULL,
    file_size     BIGINT,
    total_pages   INTEGER,
    total_chunks  INTEGER      DEFAULT 0,
    embedding_model TEXT       NOT NULL,           -- ex: paraphrase-multilingual-mpnet-base-v2
    chunk_size    INTEGER      NOT NULL,
    chunk_overlap INTEGER      NOT NULL,
    ingested_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_name ON documents(file_name);

-- ------------------------------------------------------------
-- chunks — fragmentos de texto com seus embeddings
-- Cada chunk referencia um documento pai via FK.
-- O índice HNSW do pgvector acelera a busca por similaridade.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id              BIGSERIAL    PRIMARY KEY,
    document_id     INTEGER      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER      NOT NULL,           -- posição ordinal dentro do documento
    page_number     INTEGER,                          -- página de origem (quando extraível)
    content         TEXT         NOT NULL,
    content_length  INTEGER      GENERATED ALWAYS AS (char_length(content)) STORED,
    embedding       vector(768),                      -- ajuste a dimensão conforme EMBEDDING_DIMENSION
    metadata        JSONB        NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    UNIQUE (document_id, chunk_index)               -- evita duplicação de chunks do mesmo doc
);

-- Índice HNSW para busca aproximada por cosseno (mais rápido que IVFFlat para <1M vetores)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Índice para recuperação por documento
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

-- ------------------------------------------------------------
-- Função utilitária: busca semântica por similaridade de cosseno
-- Parâmetros:
--   query_embedding : vetor da pergunta do usuário
--   top_k           : número de chunks a retornar
--   min_score       : threshold de similaridade (0.0 a 1.0)
-- ------------------------------------------------------------
CREATE OR REPLACE FUNCTION search_chunks(
    query_embedding vector,
    top_k           INTEGER DEFAULT 5,
    min_score       FLOAT   DEFAULT 0.30
)
RETURNS TABLE (
    chunk_id        BIGINT,
    document_id     INTEGER,
    file_name       TEXT,
    file_path       TEXT,
    chunk_index     INTEGER,
    page_number     INTEGER,
    content         TEXT,
    score           FLOAT
)
LANGUAGE SQL STABLE AS $$
    SELECT
        c.id            AS chunk_id,
        c.document_id,
        d.file_name,
        d.file_path,
        c.chunk_index,
        c.page_number,
        c.content,
        1 - (c.embedding <=> query_embedding) AS score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE 1 - (c.embedding <=> query_embedding) >= min_score
    ORDER BY c.embedding <=> query_embedding
    LIMIT top_k;
$$;

-- ------------------------------------------------------------
-- ingestion_log — auditoria de execuções do pipeline
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingestion_log (
    id              SERIAL       PRIMARY KEY,
    started_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ,
    pdf_root        TEXT         NOT NULL,
    files_found     INTEGER      DEFAULT 0,
    files_new       INTEGER      DEFAULT 0,
    files_skipped   INTEGER      DEFAULT 0,
    files_error     INTEGER      DEFAULT 0,
    chunks_created  INTEGER      DEFAULT 0,
    status          TEXT         CHECK (status IN ('running','completed','error')),
    error_message   TEXT
);