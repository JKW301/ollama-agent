import json
import ollama
from typing import Callable

from config import MODEL, MAX_STEPS, TEMPERATURE, NUM_PREDICT
from tools import SCHEMAS, dispatch


def _extract_inline_tool_calls(content: str) -> list[dict] | None:
    """Fallback: Mistral retourne parfois les tool calls en JSON dans le content."""
    content = content.strip()
    if not content.startswith("["):
        return None
    try:
        data = json.loads(content)
        if isinstance(data, list) and all(
            isinstance(item, dict) and "name" in item and "arguments" in item
            for item in data
        ):
            return [{"function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in data]
    except (json.JSONDecodeError, KeyError):
        pass
    return None

SYSTEM_PROMPT = """Tu es un assistant développeur CLI. Tu as accès à des outils pour:
- Lire, écrire, lister des fichiers sur le système local
- Exécuter des commandes shell (bash)
- Te connecter à des serveurs distants via SSH

Utilise les outils chaque fois que c'est nécessaire pour répondre à la demande.
Tu peux enchaîner plusieurs appels d'outils dans une même conversation.

Après avoir utilisé un outil, réponds en UNE PHRASE courte en français.
Ne répète jamais le contenu brut retourné par l'outil dans ta réponse finale.
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

            # Fallback : Mistral embed parfois les tool calls comme JSON dans content
            if not tool_calls and msg.get("content"):
                tool_calls = _extract_inline_tool_calls(msg["content"])
                if tool_calls:
                    # Remplace le message dans l'historique pour ne pas renvoyer le JSON brut
                    self.messages[-1] = {"role": "assistant", "content": ""}

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
