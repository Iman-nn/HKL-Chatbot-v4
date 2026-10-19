"""Application factory and HTTP boundary for the HKL information chatbot."""

import math
import os
import re
import unicodedata
from pathlib import Path
import sqlite3
import uuid

import click
from flask import Flask, g, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from .knowledge import KnowledgeBase, normalize
from .provider import FALLBACK, OpenRouter
from .storage import CapacityExceeded, ConversationMissing, Store

ROOT = Path(__file__).resolve().parent.parent


def create_app(config=None) -> Flask:
    app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
    app.config.from_mapping(
        MAX_CONTENT_LENGTH=16 * 1024,
        FAQ_PATH=str(ROOT / "hklFAQsv2.csv"),
        NAVIGATION_PATH=str(ROOT / "hklNavigation.csv"),
        DATABASE=os.getenv("HKL_DATABASE", str(ROOT / "instance" / "chatbot.sqlite3")),
        CONVERSATION_TTL_SECONDS=3600, MAX_CONVERSATIONS=1000, MAX_HISTORY=50,
        MAX_MESSAGE_LENGTH=2000, SIMILARITY_THRESHOLD=0.3,
        RATE_LIMIT_PER_MINUTE=60, START_LIMIT_PER_MINUTE=10,
        OPENROUTER_ENABLED=os.getenv("OPENROUTER_ENABLED", "false").lower() == "true",
        OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY", ""),
        OPENROUTER_MODEL=os.getenv("OPENROUTER_MODEL", ""),
        TRUSTED_HOSTS=[host.strip() for host in os.getenv("HKL_TRUSTED_HOSTS", "localhost,127.0.0.1,[::1]").split(",") if host.strip()],
        # Enable only after HTTPS is enforced at the reverse proxy.
        HSTS_ENABLED=os.getenv("HKL_HSTS_ENABLED", "false").lower() == "true",
    )
    if config:
        app.config.update(config)
    for key in ("CONVERSATION_TTL_SECONDS", "MAX_CONVERSATIONS", "MAX_HISTORY", "MAX_MESSAGE_LENGTH",
                "RATE_LIMIT_PER_MINUTE", "START_LIMIT_PER_MINUTE", "MAX_CONTENT_LENGTH"):
        value = app.config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    threshold = app.config["SIMILARITY_THRESHOLD"]
    if not isinstance(threshold, (float, int)) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("SIMILARITY_THRESHOLD must be in (0, 1]")
    if not app.config["TRUSTED_HOSTS"]:
        raise ValueError("TRUSTED_HOSTS must not be empty")

    knowledge = KnowledgeBase(Path(app.config["FAQ_PATH"]), Path(app.config["NAVIGATION_PATH"]), threshold)
    store = Store(str(app.config["DATABASE"]), app.config["CONVERSATION_TTL_SECONDS"],
                  app.config["MAX_CONVERSATIONS"], app.config["MAX_HISTORY"])
    provider = OpenRouter(app.config["OPENROUTER_API_KEY"], app.config["OPENROUTER_MODEL"], app.config["OPENROUTER_ENABLED"])
    app.extensions.update(knowledge=knowledge, store=store, provider=provider)

    @app.before_request
    def protect_request():
        g.request_id = uuid.uuid4().hex
        if request.endpoint in {"static", "health"}:
            return None
        # Caller-supplied X-Forwarded-For is deliberately ignored.
        identity = request.remote_addr or "unknown"
        if not store.allow(f"requests:{identity}", app.config["RATE_LIMIT_PER_MINUTE"]):
            return jsonify(error="Too many requests. Please try again later."), 429, {"Retry-After": "60"}
        if request.method == "POST" and not request.is_json:
            return jsonify(error="Content-Type must be application/json."), 415
        if request.method in {"POST", "DELETE"} and request.headers.get("Sec-Fetch-Site") == "cross-site":
            return jsonify(error="Cross-site requests are not permitted."), 403

    @app.after_request
    def response_headers(response):
        response.headers["X-Request-ID"] = getattr(g, "request_id", uuid.uuid4().hex)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), geolocation=(), microphone=(self)"
        if request.endpoint != "static":
            response.headers["Cache-Control"] = "no-store"
        if app.config["HSTS_ENABLED"]:
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    @app.errorhandler(HTTPException)
    def http_error(error):
        # Preserve protocol headers such as Allow on 405 responses.
        response = error.get_response()
        response.data = app.json.dumps({"error": error.name})
        response.content_type = "application/json"
        return response

    @app.errorhandler(ConversationMissing)
    def missing_conversation(_error):
        return jsonify(error="Conversation not found or expired."), 404

    @app.errorhandler(CapacityExceeded)
    @app.errorhandler(sqlite3.Error)
    def storage_unavailable(_error):
        app.logger.warning("Storage unavailable request_id=%s", getattr(g, "request_id", "unknown"))
        return jsonify(error="Service temporarily unavailable."), 503, {"Retry-After": "60"}

    @app.errorhandler(Exception)
    def unexpected_error(error):
        # Event type and correlation ID only; no bodies, tokens, or exception text.
        app.logger.error("Unhandled %s request_id=%s", type(error).__name__, getattr(g, "request_id", "unknown"))
        return jsonify(error="An unexpected error occurred."), 500

    def token() -> str:
        header = request.headers.get("Authorization", "")
        scheme, _, value = header.partition(" ")
        if scheme.lower() != "bearer" or not re.fullmatch(r'[A-Za-z0-9_-]{43}', value):
            raise ConversationMissing
        return value

    @app.get("/")
    def home():
        return render_template("index.html", external_enabled=provider.enabled)

    @app.get("/health")
    def health():
        with store.connection() as db:
            db.execute("SELECT 1 FROM conversations LIMIT 1").fetchone()
        return jsonify(status="ok")

    @app.post("/start_conversation")
    def start_conversation():
        body = request.get_json()
        if not isinstance(body, dict) or body:
            return jsonify(error="Expected an empty JSON object."), 400
        if not store.allow(f"start:{request.remote_addr or 'unknown'}", app.config["START_LIMIT_PER_MINUTE"]):
            return jsonify(error="Too many new conversations."), 429, {"Retry-After": "60"}
        conversation_id, access_token = store.create()
        return jsonify(conversation_id=conversation_id, conversation_token=access_token,
                       expires_in=app.config["CONVERSATION_TTL_SECONDS"]), 201

    @app.post("/ask/<conversation_id>")
    def ask(conversation_id):
        access_token = token()
        store.authorize(conversation_id, access_token)
        body = request.get_json()
        if not isinstance(body, dict) or set(body) != {"message"}:
            return jsonify(error="Expected a JSON object containing only message."), 400
        message = body["message"]
        if (not isinstance(message, str) or not 1 <= len(message) <= app.config["MAX_MESSAGE_LENGTH"]
                or not normalize(message) or any(unicodedata.category(char) in {'Cc', 'Cs'} and char not in "\n\r\t" for char in message)):
            return jsonify(error=f"Message must contain text, up to {app.config['MAX_MESSAGE_LENGTH']} characters."), 400
        result = knowledge.answer(message.strip())
        if result["response"] is None:
            if provider.enabled and not store.allow("external-provider", 20):
                return jsonify(error="External service is busy. Please try again later."), 429, {"Retry-After": "60"}
            result["response"] = provider.answer(message.strip())
            if result["response"] != FALLBACK:
                result["source"] = "external_ai"
        store.append(conversation_id, access_token, message.strip(), result["response"])
        return jsonify(result)

    @app.get("/history/<conversation_id>")
    def history(conversation_id):
        return jsonify(history=store.history(conversation_id, token()))

    @app.delete("/history/<conversation_id>")
    def delete_history(conversation_id):
        store.delete(conversation_id, token())
        return "", 204

    @app.get("/faq_classes")
    def faq_classes():
        return jsonify(classes=knowledge.categories())

    @app.get("/faqs/<class_name>")
    def faqs(class_name):
        return jsonify(faqs=knowledge.category(class_name))

    @app.get("/navigation/<location>")
    def navigation(location):
        if len(location) > 200 or not normalize(location):
            return jsonify(error="Invalid location."), 400
        rows = knowledge.navigation(location)
        if not rows:
            return jsonify(message="Location not found."), 404
        return jsonify(navigation=rows)

    @app.cli.command("prune-data")
    def prune_data():
        """Remove expired conversation and rate-limit rows."""
        store.prune()
        click.echo("Expired data removed.")

    return app
