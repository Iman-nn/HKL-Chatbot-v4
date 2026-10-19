"""Real HTTP smoke test through Waitress; loopback only, no external provider calls."""

from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import requests


def main():
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as directory:
        # Let the OS choose a temporary port. Retry in CI if another process claims it.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        script = (
            "from hkl_chatbot import create_app; from waitress import serve; "
            f"app=create_app({{'DATABASE': {str(Path(directory) / 'smoke.sqlite3')!r}, "
            "'OPENROUTER_ENABLED': False}); "
            f"serve(app,host='127.0.0.1',port={port},threads=4)"
        )
        process = subprocess.Popen([sys.executable, "-c", script], cwd=root,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        origin = f"http://127.0.0.1:{port}"
        try:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError("WSGI server exited before becoming ready")
                try:
                    if requests.get(origin + "/health", timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.2)
            else:
                raise RuntimeError("WSGI startup timed out")
            response = requests.get(origin + "/", timeout=3)
            assert response.status_code == 200
            assert "unsafe-inline" not in response.headers["Content-Security-Policy"]
            for path in ("/static/chat.js", "/static/style.css"):
                assert requests.get(origin + path, timeout=3).status_code == 200
            response = requests.post(origin + "/start_conversation", json={}, timeout=3)
            assert response.status_code == 201
            session = response.json()
            identifier = session["conversation_id"]
            headers = {"Authorization": "Bearer " + session["conversation_token"]}
            response = requests.post(origin + "/ask/" + identifier,
                                     json={"message": "What are the library opening hours?"},
                                     headers=headers, timeout=3)
            assert response.status_code == 200
            assert response.json()["source"] == "faq"
            assert requests.get(origin + "/history/" + identifier, timeout=3).status_code == 404
            history = requests.get(origin + "/history/" + identifier, headers=headers, timeout=3)
            assert len(history.json()["history"]) == 1
            assert requests.delete(origin + "/history/" + identifier, headers=headers, timeout=3).status_code == 204
            assert requests.get(origin + "/history/" + identifier, headers=headers, timeout=3).status_code == 404
            print("PASS: Waitress HTTP, assets, session, FAQ, access control, history, deletion")
        finally:
            process.terminate()
            process.communicate(timeout=10)


if __name__ == "__main__":
    main()
