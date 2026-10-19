"""WSGI entry point. Use a production WSGI server rather than Flask's debugger."""

from hkl_chatbot import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=False)
