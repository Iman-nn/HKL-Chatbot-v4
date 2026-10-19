"""Validate the rendered UI against its client bindings and security boundary."""

from html.parser import HTMLParser
from pathlib import Path
import re
import tempfile
import unittest

from hkl_chatbot import create_app, ROOT


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.elements = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        self.elements.append((tag, attrs))


class RenderedUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.app = create_app({
            "TESTING": True,
            "DATABASE": str(Path(cls.temporary.name) / "ui.sqlite3"),
            "OPENROUTER_ENABLED": False,
        })
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_client_controls_exist_once_in_rendered_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        parser = PageParser()
        parser.feed(response.get_data(as_text=True))
        self.assertEqual(len(parser.ids), len(set(parser.ids)), "Duplicate control IDs")
        source = (ROOT / "static" / "chat.js").read_text(encoding="utf-8")
        bindings = set(re.findall(r'#[a-z][a-z0-9-]+', source))
        self.assertFalse(bindings - {"#" + identifier for identifier in parser.ids},
                         "Client references a control that is missing from the page")
        questions = [attrs["data-question"] for _, attrs in parser.elements if "data-question" in attrs]
        self.assertTrue(questions)
        for question in questions:
            self.assertEqual(self.app.extensions["knowledge"].answer(question)["source"], "faq")

    def test_new_layout_keeps_scripts_and_styles_local(self):
        parser = PageParser()
        response = self.client.get("/")
        parser.feed(response.get_data(as_text=True))
        for tag, attrs in parser.elements:
            self.assertFalse(any(name.startswith("on") for name in attrs), "Inline event handler")
            self.assertNotIn("style", attrs, "Inline styling bypasses the CSP")
            if tag in {"script", "img"}:
                self.assertTrue(attrs["src"].startswith("/static/"))
            if tag == "link" and attrs.get("rel") == "stylesheet":
                self.assertTrue(attrs["href"].startswith("/static/"))
        self.assertNotIn("unsafe-inline", response.headers["Content-Security-Policy"])

    def test_external_ai_disclosure_follows_provider_setting(self):
        provider = self.app.extensions["provider"]
        self.assertNotIn("External AI assistance", self.client.get("/").get_data(as_text=True))
        try:
            provider.enabled = True
            markup = self.client.get("/").get_data(as_text=True)
            self.assertIn("External AI assistance", markup)
            self.assertIn("Unmatched questions may be sent to an external AI provider.", markup)
        finally:
            provider.enabled = False


if __name__ == "__main__":
    unittest.main()
