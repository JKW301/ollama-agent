import ollama
from typing import Callable

from config import MODEL, MAX_STEPS, TEMPERATURE, NUM_PREDICT
from tools import SCHEMAS, dispatch

SYSTEM_PROMPT = """Tu es un assistant développeur CLI. Tu as accès à des outils pour:
- Lire, écrire, lister des fichiers sur le système local
- Exécuter des commandes shell (bash)
- Te connecter à des serveurs distants via SSH

Utilise les outils chaque fois que c'est nécessaire pour répondre à la demande.
Tu peux enchaîner plusieurs appels d'outils dans une même conversation.
Réponds toujours en français."""


class Agent:
    def __init__(self):
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def reset(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def run(
        self,
        user_input: str,
        on_tool_call: Callable[[str, dict, str], None] | None = None,
        confirm_tool: Callable[[str, dict], bool] | None = None,
    ) -> str:
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(MAX_STEPS):
            try:
                response = ollama.chat(
                    model=MODEL,
                    messages=self.messages,
                    tools=SCHEMAS,
                    options={
                        "temperature": TEMPERATURE,
                        "num_predict": NUM_PREDICT,
                    },
                )
            except Exception as e:
                return f"[ERR] Ollama: {e}"

            msg = response["message"]
            self.messages.append(msg)

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                return msg.get("content", "")

            for tc in tool_calls:
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]

                if confirm_tool and not confirm_tool(name, args):
                    result = "[ANNULÉ] Exécution refusée par l'utilisateur."
                else:
                    result = dispatch(name, args)

                if on_tool_call:
                    on_tool_call(name, args if isinstance(args, dict) else {}, result)

                self.messages.append({
                    "role": "tool",
                    "content": result,
                    "name": name,
                })

        return "[ERR] Nombre maximum d'étapes atteint."
