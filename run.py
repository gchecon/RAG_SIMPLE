"""run.py — Ponto de entrada da aplicação RAG TCE."""
from app import create_app
from config.config import config

app = create_app()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=config.flask.port,
        debug=config.flask.debug,
    )