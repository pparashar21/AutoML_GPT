"""
Command-line chat loop + reusable ask() helper.

Usage
-----
$ python -m src.components.chatbot           # CLI mode
>>> ask("your question") inside another module

Dependencies: src.components.knowledge_base, src.utils.utility
"""

from __future__ import annotations
import json, os, warnings, readline  # readline → arrow-key history in shell
from typing import List, Tuple

import seaborn as sns
from src.components.knowledge_base import load_docs, rag
from src.utils.utility import parse_json_garbage, runner

warnings.filterwarnings("ignore")

Chat = List[Tuple[str, str]]
chat_history:Chat = []       

def _format_bot(text: str) -> str:
    """Blue colour for bot in terminal."""
    return f"\033[94mChatbot:\033[0m {text}"


def _format_user(text: str) -> str:
    return f"\033[1mUser:\033[0m {text}"


def ask(query: str) -> str:
    """
    Core dialogue handler:
      • loads docs on first question
      • otherwise routes through rag() or runner()
      • updates chat_history
    Returns the bot response string (for UI layers to display).
    """
    global chat_history

    # stop phrase
    if query.lower() == "exit":
        return "Thank you for using the State of the Union chatbot!"

    # first turn → bootstrap knowledge base
    if not chat_history:
        chat_history.append((query, "…loading docs…"))
        load_docs()
        response = "Machine-learning model documentation loaded into memory!"
        chat_history[-1] = (query, response)
        return response

    # subsequent turns
    response = rag(chat_history, query)

    if response == "-1":   # build + evaluate model branch
        chat_history.append((query, "building model…"))
        data_dict = parse_json_garbage(chat_history[-2][1])
        json_path = os.path.join("JSONs", "sample.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(data_dict, f, indent=2)

        metrics = runner("sample.json", "model_parameters.json")
        acc, cr, cm, bp = metrics["acc"], metrics["cr"], metrics["cm"], metrics["paramters"]

        response = (
            f"The parameters used are: {bp}\n"
            f"The accuracy of the model is {acc}\n"
            f"The classification report is\n{cr}\n"
            f"The confusion matrix is:\n{cm}"
        )
        # retain pretty heat-map for notebook / Streamlit display
        metrics["cm_plot"] = sns.heatmap(cm, annot=True, fmt="d").get_figure()

    # normal rag answer branch
    chat_history.append((query, response))
    return response

def cli_loop() -> None:
    print(
        "AutoML-GPT CLI\n"
        "Type your question and press Enter.\n"
        "Type 'exit' to quit.\n"
    )

    while True:
        try:
            user_in = input(">>>").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting…")
            break

        if not user_in:
            continue

        bot_out = ask(user_in)
        print(_format_user(user_in))
        print(_format_bot(bot_out))

        if user_in.lower() == "exit":
            break


if __name__ == "__main__":
    cli_loop()