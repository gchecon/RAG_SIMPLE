# RAG TCE — Sistema de Consulta Documental

Sistema RAG (Retrieval-Augmented Generation) para consulta de documentos PDF,
com backend Flask, embeddings via sentence-transformers (ou Azure OpenAI) e
armazenamento vetorial no PostgreSQL/pgvector.

---

## Pré-requisitos

- Python 3.11+
- PostgreSQL 14+ com extensão **pgvector** instalada
- PyCharm Pro (recomendado para desenvolvimento)

## Setup

### 1. Banco de dados

```sql
-- Como superusuário PostgreSQL:
CREATE DATABASE rag_tce;
\c rag_tce
CREATE EXTENSION vector;
```

O schema (tabelas, índices, função search_chunks) é aplicado automaticamente
na primeira execução do Flask via `init_schema()`.

### 2. Ambiente Python

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 3. Configuração

```bash
copy .env.example .env          # Windows
# Edite .env com suas credenciais
```

Campos obrigatórios no `.env`:
- `AZURE_ENDPOINT`, `AZURE_API_KEY`, `AZURE_MODEL_NAME`, `AZURE_DEPLOYMENT_NAME`
- `PG_PASSWORD`
- `PDF_ROOT_DIR` (caminho para os PDFs)

### 4. Executar

```bash
python run.py
# Acesse http://localhost:5000
```

---

## Estrutura

```
rag_tce/
├── run.py                      # Ponto de entrada Flask
├── requirements.txt
├── .env.example
├── config/
│   ├── config.py               # Carrega e valida .env
│   └── schema.sql              # Schema PostgreSQL
└── app/
    ├── __init__.py             # App factory Flask
    ├── models/
    │   └── database.py         # Pool de conexões psycopg2
    ├── services/
    │   ├── embedding_service.py  # HuggingFace ou Azure OpenAI
    │   ├── ingestion_service.py  # Pipeline PDF → chunks → pgvector
    │   └── rag_service.py        # Retrieval + inferência DeepSeek
    ├── routes/
    │   ├── api.py              # /api/* endpoints REST + SSE
    │   └── main.py             # Serve index.html
    └── templates/
        └── index.html          # Interface web
```

---

## Modelo de dados

| Tabela           | Propósito                                              |
|------------------|--------------------------------------------------------|
| `documents`      | Controle de arquivos; `file_hash` SHA-256 garante dedup|
| `chunks`         | Fragmentos de texto + embedding vector                 |
| `ingestion_log`  | Auditoria de execuções do pipeline                     |

**Deduplicação:** antes de processar um PDF, o sistema calcula seu SHA-256
e verifica na tabela `documents`. Se o hash já existe, o arquivo é ignorado
mesmo que tenha sido movido de diretório.

---

## Perguntas pendentes (a responder pelo usuário)

1. **Embedding provider**: `huggingface` (local, gratuito) ou `azure_openai`?
2. **Idioma dos PDFs**: majoritariamente português?
   - Se sim, `paraphrase-multilingual-mpnet-base-v2` é a escolha recomendada.
3. **Schema/database PostgreSQL**: `rag_tce` está OK ou prefere outro nome?
4. **Porta Flask**: `5000` não conflita com outros serviços locais?
5. **PDFs escaneados** (imagem sem OCR): deseja integrar Tesseract/OCR?
   - `pdfplumber` não faz OCR; PDFs escaneados resultarão em texto vazio.
6. **Múltiplos modelos**: planeja adicionar GPT-4o, Claude, Qwen etc. pelo mesmo combobox?
   - Se sim, a lógica de roteamento por provider precisará ser adicionada ao `rag_service.py`.