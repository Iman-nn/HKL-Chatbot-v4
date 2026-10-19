"""Command-line client using the same retrieval engine as the web application."""

from pathlib import Path

from hkl_chatbot.knowledge import KnowledgeBase
from hkl_chatbot.provider import FALLBACK

ROOT = Path(__file__).resolve().parent


def main():
    knowledge = KnowledgeBase(ROOT / "hklFAQsv2.csv", ROOT / "hklNavigation.csv")
    print("HKL Chatbot. Type Q to quit; TOP5 for suggestions, CONF for answers.")
    suggestions = False
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.upper() == "Q":
            break
        if message.upper() in {"TOP5", "CONF"}:
            suggestions = message.upper() == "TOP5"
            continue
        if not message or len(message) > 2000:
            print("Bot: Please enter a question of up to 2,000 characters.")
            continue
        result = knowledge.answer(message)
        print("Bot:", result["response"] or FALLBACK)
        if suggestions:
            for question in result["suggestions"]:
                print(" -", question)


if __name__ == "__main__":
    main()
