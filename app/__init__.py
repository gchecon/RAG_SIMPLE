"""app/__init__.py — Flask application factory."""
from flask import Flask

from app.models.database import init_pool, init_schema
from app.routes.api import api
from app.routes.main import main
from config.config import config


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )
    app.secret_key = config.flask.secret_key

    # Inicializa pool de conexões e schema
    init_pool()
    init_schema()

    # Blueprints
    app.register_blueprint(main)
    app.register_blueprint(api)

    return app