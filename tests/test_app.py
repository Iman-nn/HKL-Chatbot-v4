"""Behavior, security boundaries, concurrency, and provider failure regression tests."""

from concurrent.futures import ThreadPoolExecutor
import csv
import importlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import requests

from corpus_gen import convert
from hkl_chatbot import create_app, ROOT
from hkl_chatbot.knowledge import KnowledgeBase, read_records
from hkl_chatbot.provider import FALLBACK, OpenRouter
from hkl_chatbot.storage import CapacityExceeded, ConversationMissing, Store


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = {
            "TESTING": True, "DATABASE": str(Path(self.temp.name) / "test.sqlite3"),
            "RATE_LIMIT_PER_MINUTE": 10000, "START_LIMIT_PER_MINUTE": 10000,
            "OPENROUTER_ENABLED": False, "OPENROUTER_API_KEY": "", "OPENROUTER_MODEL": "",
        }
        self.app = create_app(self.config)
        self.client = self.app.test_client()
        response = self.client.post("/start_conversation", json={})
        self.assertEqual(response.status_code, 201)
        self.session = response.get_json()
        self.identifier = self.session["conversation_id"]
        self.headers = {"Authorization": "Bearer " + self.session["conversation_token"]}

    def ask(self, message):
        return self.client.post("/ask/" + self.identifier, json={"message": message}, headers=self.headers)

    def test_home_and_security_headers(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"/static/chat.js", response.data)
        self.assertNotIn(b"cdn.", response.data)
        self.assertNotIn(b"<script>", response.data)
        self.assertNotIn("unsafe-inline", response.headers["Content-Security-Policy"])
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(len(response.headers["X-Request-ID"]), 32)

    def test_exact_faq_and_history(self):
        question = "What are the library opening hours?"
        response = self.ask(question)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["source"], "faq")
        self.assertIn("8:00", response.json["response"])
        history = self.client.get("/history/" + self.identifier, headers=self.headers).json["history"]
        self.assertEqual(history, [{"user": question, "bot": response.json["response"]}])

    def test_cross_conversation_access_denied(self):
        other = self.client.post("/start_conversation", json={}).json
        wrong = {"Authorization": "Bearer " + other["conversation_token"]}
        for endpoint, method in (("/history/", "GET"), ("/history/", "DELETE"), ("/ask/", "POST")):
            response = self.client.open(endpoint + self.identifier, method=method, headers=wrong, json={"message": "test"})
            self.assertEqual(response.status_code, 404)
        for headers in ({}, {"Authorization": "Bearer short"}, {"Authorization": "Bearer " + "!" * 43}):
            self.assertEqual(self.client.get("/history/" + self.identifier, headers=headers).status_code, 404)
        self.assertEqual(self.client.get("/history/not-real", headers=self.headers).status_code, 404)

    def test_expired_session_and_pruning(self):
        store = self.app.extensions["store"]
        with store.connection() as db:
            db.execute("UPDATE conversations SET expires = ?", (time.time() - 1,))
        self.assertEqual(self.ask("opening hours").status_code, 404)
        result = self.app.test_cli_runner().invoke(args=["prune-data"])
        self.assertEqual(result.exit_code, 0)
        with store.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0], 0)

    def test_invalid_json_and_input_types(self):
        for body in ([], None, "message", {}, {"message": None}, {"message": 8}, {"message": []},
                     {"message": {}}, {"message": False}, {"message": ""}, {"message": "   "},
                     {"message": "!!!"}, {"message": "x" * 2001}, {"message": "a\\u0000"},
                     {"message": "\ud800"}, {"message": "ok", "extra": True}):
            if body == {"message": "a\\u0000"}:
                body = {"message": "a" + chr(0)}
            with self.subTest(body=repr(body)):
                response = self.client.post("/ask/" + self.identifier, data=json.dumps(body),
                                            content_type="application/json", headers=self.headers)
                self.assertEqual(response.status_code, 400)
        response = self.client.post("/ask/" + self.identifier, data="{", content_type="application/json", headers=self.headers)
        self.assertEqual(response.status_code, 400)
        response = self.client.post("/ask/" + self.identifier, data="{}", headers=self.headers)
        self.assertEqual(response.status_code, 415)

    def test_request_body_limit(self):
        response = self.client.post("/ask/" + self.identifier, data=json.dumps({"message": "x" * 20000}),
                                    content_type="application/json", headers=self.headers)
        self.assertEqual(response.status_code, 413)

    def test_unicode_and_xss_are_plain_data(self):
        payload = '<img src=x onerror="alert(1)"> lokasi hospital'
        self.assertEqual(self.ask(payload).status_code, 200)
        history = self.client.get("/history/" + self.identifier, headers=self.headers).json["history"]
        self.assertEqual(history[0]["user"], payload)
        self.assertEqual(self.ask("Di manakah perpustakaan di HKL?").status_code, 200)

    def test_navigation_with_punctuation_and_natural_question(self):
        for question in ("Blood Collection Room 2 (Special)", "Where is the Diabetes Resource Center?"):
            response = self.ask(question)
            self.assertEqual(response.json["source"], "navigation")
            self.assertIn("Location:", response.json["response"])
            self.assertNotIn("<b>", response.json["response"])
        self.assertEqual(self.client.get("/navigation/Pharmacy").status_code, 200)
        self.assertEqual(self.client.get("/navigation/%5B").status_code, 400)
        self.assertEqual(self.client.get("/navigation/.*").status_code, 400)
        self.assertEqual(self.client.get("/navigation/zzzznotfound").status_code, 404)
        self.assertIn("not available", self.client.get("/navigation/RHU").json["navigation"][0]["Instructions"])

    def test_exact_faqs_are_not_overridden_by_location_names(self):
        knowledge = self.app.extensions["knowledge"]
        unique = {row["Question"]: row["Answer"] for row in knowledge.faqs}
        for question in unique:
            with self.subTest(question=question):
                self.assertEqual(knowledge.answer(question)["source"], "faq")
        rows = knowledge.navigation("Pharmacy")
        self.assertEqual(rows[0]["Location"], "Pharmacy")

    def test_external_fallback_has_a_label_and_shared_budget(self):
        provider = self.app.extensions["provider"]
        provider.enabled = True
        with patch.object(provider, "answer", return_value="Generated answer") as answer:
            for _ in range(20):
                response = self.ask("zxqv unknownword")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json["source"], "external_ai")
            self.assertEqual(self.ask("zxqv unknownword").status_code, 429)
            self.assertEqual(answer.call_count, 20)

    def test_busy_storage_returns_safe_error(self):
        import sqlite3
        with patch.object(self.app.extensions["store"], "authorize", side_effect=sqlite3.OperationalError("private path")):
            response = self.ask("hours")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(b"private path", response.data)

    def test_categories(self):
        categories = self.client.get("/faq_classes").json["classes"]
        self.assertIn("Library", categories)
        self.assertTrue(self.client.get("/faqs/Library").json["faqs"])

    def test_disabled_provider_never_contacts_external_service(self):
        with patch("hkl_chatbot.provider.requests.post") as post:
            response = self.ask("zxqv nonmatchingword")
        post.assert_not_called()
        self.assertEqual(response.json["response"], FALLBACK)
        self.assertEqual(response.json["suggestions"], [])

    def test_query_only_transforms_user_input(self):
        knowledge = self.app.extensions["knowledge"]
        with patch.object(knowledge.vectorizer, "transform", wraps=knowledge.vectorizer.transform) as transform:
            self.ask("What are the library opening hours?")
        transform.assert_called_once_with(["What are the library opening hours?"])

    def test_delete_cascades_history(self):
        self.ask("library opening hours")
        self.assertEqual(self.client.delete("/history/" + self.identifier, headers=self.headers).status_code, 204)
        self.assertEqual(self.client.get("/history/" + self.identifier, headers=self.headers).status_code, 404)
        with self.app.extensions["store"].connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 0)

    def test_safe_error_and_protocol_headers(self):
        with patch.object(self.app.extensions["knowledge"], "answer", side_effect=RuntimeError("secret-patient-data")):
            response = self.ask("library opening hours")
        self.assertEqual(response.status_code, 500)
        self.assertNotIn(b"secret-patient-data", response.data)
        response = self.client.get("/start_conversation")
        self.assertEqual(response.status_code, 405)
        self.assertIn("POST", response.headers["Allow"])

    def test_rate_limit_shared_with_another_worker(self):
        app = create_app({**self.config, "RATE_LIMIT_PER_MINUTE": 1})
        other = create_app({**self.config, "RATE_LIMIT_PER_MINUTE": 1})
        address = {"REMOTE_ADDR": "192.0.2.99"}
        self.assertEqual(app.test_client().get("/faq_classes", environ_overrides=address).status_code, 200)
        response = other.test_client().get("/faq_classes", environ_overrides=address)
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["Retry-After"], "60")
        self.assertEqual(other.test_client().get("/health", environ_overrides=address).status_code, 200)

    def test_start_limit_and_cross_site_request(self):
        app = create_app({**self.config, "START_LIMIT_PER_MINUTE": 1})
        client = app.test_client()
        address = {"REMOTE_ADDR": "192.0.2.55"}
        self.assertEqual(client.post("/start_conversation", json={}, environ_overrides=address).status_code, 201)
        self.assertEqual(client.post("/start_conversation", json={}, environ_overrides=address).status_code, 429)
        response = client.post("/start_conversation", json={}, headers={"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(response.status_code, 403)
        self.assertNotIn("Access-Control-Allow-Origin", response.headers)

    def test_untrusted_host(self):
        response = self.client.get("/", headers={"Host": "attacker.invalid"})
        self.assertEqual(response.status_code, 400)

    def test_restart_preserves_authorized_history(self):
        self.ask("library opening hours")
        restarted = create_app(self.config)
        response = restarted.test_client().get("/history/" + self.identifier, headers=self.headers)
        self.assertEqual(len(response.json["history"]), 1)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = str(Path(self.temp.name) / "store.sqlite3")
        self.store = Store(self.path, ttl=60, max_conversations=100, max_history=3)

    def test_parallel_creation_produces_unique_ids(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            sessions = list(pool.map(lambda _: self.store.create(), range(24)))
        self.assertEqual(len({identifier for identifier, _ in sessions}), 24)
        with self.store.connection() as db:
            hashes = {row[0] for row in db.execute("SELECT token_hash FROM conversations")}
        self.assertFalse(hashes.intersection(token for _, token in sessions))

    def test_atomic_capacity(self):
        store = Store(self.path, 60, 1, 3)
        def create(_):
            try:
                return store.create()
            except CapacityExceeded:
                return None
        with ThreadPoolExecutor(max_workers=4) as pool:
            sessions = list(pool.map(create, range(8)))
        self.assertEqual(sum(session is not None for session in sessions), 1)

    def test_history_bound_and_parallel_writes(self):
        identifier, token = self.store.create()
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda index: self.store.append(identifier, token, str(index), "answer"), range(12)))
        self.assertEqual(len(self.store.history(identifier, token)), 3)

    def test_atomic_rate_limit(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            accepted = list(pool.map(lambda _: self.store.allow("shared-user", 5), range(24)))
        self.assertEqual(sum(accepted), 5)

    def test_expiry_releases_capacity(self):
        store = Store(self.path, 60, 1, 3)
        identifier, token = store.create()
        with store.connection() as db:
            db.execute("UPDATE conversations SET expires = 0")
        with self.assertRaises(ConversationMissing):
            store.history(identifier, token)
        store.create()


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = OpenRouter("test-key", "test-model", True)

    def reply(self, body, status=200):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status_code = status
        response.iter_content.return_value = [json.dumps(body).encode()]
        return response

    def test_success_has_timeout_no_redirects_and_token_cap(self):
        response = self.reply({"choices": [{"message": {"content": "Administrative answer"}}]})
        with patch("hkl_chatbot.provider.requests.post", return_value=response) as post:
            self.assertEqual(self.provider.answer("question"), "Administrative answer")
        options = post.call_args.kwargs
        self.assertEqual(options["timeout"], (3.05, 10))
        self.assertFalse(options["allow_redirects"])
        self.assertEqual(options["json"]["max_tokens"], 512)

    def test_timeout_and_bad_shapes_return_safe_fallback(self):
        with patch("hkl_chatbot.provider.requests.post", side_effect=requests.Timeout("sensitive")):
            self.assertEqual(self.provider.answer("question"), FALLBACK)
        for body in ({}, {"choices": []}, {"choices": [{"message": {"content": []}}]},
                     {"choices": [{"message": {"content": ""}}]},
                     {"choices": [{"message": {"content": "\ud800"}}]}):
            with patch("hkl_chatbot.provider.requests.post", return_value=self.reply(body)):
                self.assertEqual(self.provider.answer("question"), FALLBACK)

    def test_response_byte_limit_and_redirect(self):
        response = self.reply({})
        response.iter_content.return_value = [b"x" * 65537]
        with patch("hkl_chatbot.provider.requests.post", return_value=response):
            self.assertEqual(self.provider.answer("question"), FALLBACK)
        with patch("hkl_chatbot.provider.requests.post", return_value=self.reply({}, status=302)):
            self.assertEqual(self.provider.answer("question"), FALLBACK)

    def test_total_deadline_and_slot_release(self):
        response = self.reply({})
        with patch("hkl_chatbot.provider.requests.post", return_value=response), patch(
                "hkl_chatbot.provider.time.monotonic", side_effect=[0, 30]):
            self.assertEqual(self.provider.answer("question"), FALLBACK)
        self.assertTrue(self.provider.slots.acquire(blocking=False))
        self.provider.slots.release()

    def test_configuration_requires_key_and_model(self):
        with self.assertRaises(ValueError):
            OpenRouter("", "model", True)
        with self.assertRaises(ValueError):
            OpenRouter("key", "", True)


class CorpusTests(unittest.TestCase):
    def test_empty_or_broken_corpus_fails_at_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            for text in ("Question,Answer,Class\n", "wrong,columns\n1,2\n",
                         "Question,Answer,Class\nquestion,,category\n"):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_records(path, {"Question", "Answer", "Class"})

    def test_export_does_not_clobber_on_validation_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source.json", Path(directory) / "target.csv"
            target.write_text("original", encoding="utf-8")
            source.write_text('{"hkl": {"Library": [["question"]]}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                convert(source, target)
            self.assertEqual(target.read_text(), "original")
            source.write_text('{"hkl": {"Library": [["question", "answer"]]}}', encoding="utf-8")
            self.assertEqual(convert(source, target), 1)
            with target.open(encoding="utf-8", newline="") as stream:
                self.assertEqual(list(csv.DictReader(stream))[0]["Answer"], "answer")

    def test_cli_import_has_no_interactive_side_effects(self):
        with patch("builtins.input", side_effect=AssertionError("Unexpected prompt")):
            importlib.import_module("FAQbot")


if __name__ == "__main__":
    unittest.main()
