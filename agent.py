import json
import ollama
from typing import Callable

from config import MODEL, MAX_STEPS, TEMPERATURE, NUM_PREDICT
from tools import SCHEMAS, dispatch

# ask_user est un outil spécial géré par l'UI, pas par dispatch
ASK_USER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": (
            "Pose une question à l'utilisateur avec une liste de choix numérotés. "
            "Utilise cet outil quand la cible d'une action est ambiguë (plusieurs fichiers possibles, etc.). "
            "Étapes : 1) list_files ou grep_search pour trouver les candidats, "
            "2) ask_user pour que l'utilisateur choisisse, "
            "3) exécute l'action sur l'élément choisi."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string",  "description": "La question à poser"},
                "options":  {"type": "array",   "items": {"type": "string"},
                             "description": "Liste des options proposées"},
            },
            "required": ["question", "options"],
        },
    },
}

ALL_SCHEMAS = SCHEMAS + [ASK_USER_SCHEMA]

_ACTION_WORDS = {
    # Fichiers
    "liste", "lister", "affiche", "afficher", "montre", "montrer",
    "lis", "lire", "ouvre", "ouvrir",
    "écris", "écrire", "crée", "créer", "ajoute", "ajouter",
    "supprime", "supprimer", "efface", "effacer", "enlève", "enlever",
    "déplace", "déplacer", "copie", "copier", "renomme", "renommer",
    "modifie", "modifier", "édite", "éditer",
    # Recherche
    "cherche", "chercher", "trouve", "trouver", "grep", "find", "recherche",
    # Shell / système
    "exécute", "exécuter", "lance", "lancer", "run", "installe", "installer",
    "compile", "build", "teste", "tester",
    # Objets
    "fichier", "dossier", "répertoire", "script", "code", "projet",
    "processus", "variable", "environnement",
    # Protocoles / web
    "git", "ssh", "http", "https", "url", "web", "fetch",
}

def _needs_tools(text: str) -> bool:
    """Renvoie True si le message contient des mots qui suggèrent une action."""
    words = set(text.lower().split())
    return bool(words & _ACTION_WORDS)


_NARRATION_PREFIXES = (
    "je vais", "voici comment", "pour ce faire",
    "permettez-moi", "allow me", "i will", "i'll", "let me",
)

def _is_narration(content: str) -> bool:
    lower = content.strip().lower()
    return any(lower.startswith(p) for p in _NARRATION_PREFIXES)


_REFUSAL_PATTERNS = (
    "je suis désolé, mais je ne peux pas",
    "je ne peux pas supprimer",
    "je ne suis pas autorisé",
    "seuls les utilisateurs autorisés",
    "i'm sorry, but i cannot",
    "i cannot delete",
    "i don't have permission",
)

def _is_refusal(content: str) -> bool:
    lower = content.strip().lower()
    return any(p in lower for p in _REFUSAL_PATTERNS)


def _extract_inline_tool_calls(content: str) -> list[dict] | None:
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


SYSTEM_PROMPT = """Tu es un assistant développeur CLI avec accès complet au système local.

AUTORISATION TOTALE : L'utilisateur t'a accordé la permission complète d'utiliser TOUS tes outils sans restriction.
Tu PEUX et DOIS supprimer, modifier, créer, déplacer des fichiers, exécuter des commandes, etc.
Ne refuse JAMAIS d'utiliser un outil sous prétexte que c'est dangereux — l'utilisateur gère les risques.

RÈGLE D'ACTION : Dès qu'un outil répond à la demande, appelle-le immédiatement. Ne décris pas, n'explique pas, agis.

Quand la cible est ambiguë (fichier introuvable, nom approximatif, plusieurs candidats) :
1. Appelle list_files ou grep_search pour trouver les candidats
2. Appelle ask_user(question=..., options=[...]) pour que l'utilisateur choisisse
3. Exécute l'action sur l'élément choisi

Outils disponibles:
  Fichiers  : list_files, read_file, write_file, append_file, delete_file, move_file, copy_file, make_dir, delete_dir, file_info
  Recherche : grep_search, find_files
  Shell/SSH : run_shell, ssh_exec
  Web       : web_fetch, http_request
  Git       : git_run
  Système   : get_cwd, change_dir, system_info, process_list, env_get, env_set
  UI        : ask_user

Réponds toujours en français. Après un outil réussi : une seule phrase courte."""


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
        on_user_choice: Callable[[str, list[str]], str] | None = None,
    ) -> str:
        self.messages.append({"role": "user", "content": user_input})

        last_tool_results: list[str] = []
        use_tools = _needs_tools(user_input)

        for _ in range(MAX_STEPS):
            try:
                response = ollama.chat(
                    model=MODEL,
                    messages=self.messages,
                    tools=ALL_SCHEMAS if use_tools else [],
                    options={"temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
                )
            except Exception as e:
                return f"[ERR] Ollama: {e}"

            msg = response["message"]
            self.messages.append(msg)

            tool_calls = msg.get("tool_calls")

            if not tool_calls and msg.get("content"):
                tool_calls = _extract_inline_tool_calls(msg["content"])
                if tool_calls:
                    use_tools = True  # activer les outils pour les tours suivants
                    self.messages[-1] = {"role": "assistant", "content": ""}

            if not tool_calls:
                content = msg.get("content", "")
                if _is_narration(content):
                    self.messages.append({
                        "role": "user",
                        "content": "Appelle l'outil maintenant, ne l'annonce pas.",
                    })
                    continue
                # Le modèle refuse malgré des outils déjà exécutés → renvoie le dernier résultat
                if _is_refusal(content) and last_tool_results:
                    return last_tool_results[-1]
                return content

            for tc in tool_calls:
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                # Outil spécial : sélecteur interactif
                if name == "ask_user":
                    question = args.get("question", "Quel choix ?")
                    options  = args.get("options", [])
                    if on_user_choice and options:
                        result = on_user_choice(question, options)
                    else:
                        result = f"Options disponibles: {', '.join(options)}"
                    self.messages.append({"role": "tool", "content": result, "name": name})
                    continue

                if confirm_tool and not confirm_tool(name, args):
                    result = "[ANNULÉ] Exécution refusée par l'utilisateur."
                else:
                    result = dispatch(name, args)

                if on_tool_call:
                    on_tool_call(name, args, result)

                last_tool_results.append(result)
                self.messages.append({"role": "tool", "content": result, "name": name})

        return "[ERR] Nombre maximum d'étapes atteint."
