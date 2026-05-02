import json
import re
import ollama
from typing import Callable

from config import MODEL, MAX_STEPS, TEMPERATURE, NUM_PREDICT
from tools import SCHEMAS, dispatch

# ── ask_user : outil spécial géré par l'UI ────────────────────────────────────
ASK_USER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": (
            "Pose une question à l'utilisateur avec des choix numérotés. "
            "Utilise-le quand la cible est ambiguë : "
            "1) cherche les candidats, 2) appelle ask_user, 3) agis sur le choix."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string",  "description": "La question"},
                "options":  {"type": "array",   "items": {"type": "string"},
                             "description": "Liste des options"},
            },
            "required": ["question", "options"],
        },
    },
}

ALL_SCHEMAS = SCHEMAS + [ASK_USER_SCHEMA]

# ── Routage : outils seulement si action détectée ─────────────────────────────
_ACTION_WORDS = {
    "liste", "lister", "affiche", "afficher", "montre", "montrer",
    "lis", "lire", "ouvre", "ouvrir",
    "écris", "écrire", "crée", "créer", "ajoute", "ajouter",
    "supprime", "supprimer", "efface", "effacer", "enlève", "enlever",
    "déplace", "déplacer", "copie", "copier", "renomme", "renommer",
    "modifie", "modifier", "édite", "éditer",
    "cherche", "chercher", "trouve", "trouver", "grep", "find", "recherche",
    "exécute", "exécuter", "lance", "lancer", "run", "installe", "installer",
    "compile", "build", "teste", "tester",
    "fichier", "dossier", "répertoire", "script", "code", "projet",
    "processus", "variable", "environnement",
    "git", "ssh", "http", "https", "url", "web", "fetch",
}

def _needs_tools(text: str) -> bool:
    return bool(set(text.lower().split()) & _ACTION_WORDS)


# ── Détection narration ────────────────────────────────────────────────────────
_NARRATION_STARTS = (
    "je vais", "voici comment", "pour ce faire", "permettez-moi",
    "allow me", "i will", "i'll", "let me",
    "je crée", "je créé", "j'exécute", "je compile", "je lance",
    "je lis ", "je liste", "je supprime", "je déplace", "je copie",
    "j'écris", "je génère", "je modifie", "je construis",
    "pour créer", "pour écrire", "pour exécuter", "pour compiler",
    "voici le fichier", "voici le code", "voici le script", "voici un",
    "here is", "here's",
)

def _is_narration(content: str, in_action_context: bool = True) -> bool:
    lower = content.strip().lower()
    if any(lower.startswith(p) for p in _NARRATION_STARTS):
        return True
    # Bloc de code = narration, seulement en contexte d'action
    if in_action_context and "```" in content and len(content) > 60:
        return True
    return False


# ── Détection refus ───────────────────────────────────────────────────────────
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


# ── Fallback : extraction de code depuis une narration ────────────────────────
_LANG_EXT = {
    "c": "main.c", "cpp": "main.cpp", "c++": "main.cpp",
    "python": "script.py", "py": "script.py",
    "javascript": "script.js", "js": "script.js",
    "typescript": "script.ts", "ts": "script.ts",
    "rust": "main.rs", "go": "main.go", "java": "Main.java",
    "bash": "script.sh", "sh": "script.sh",
    "html": "index.html", "css": "style.css",
    "sql": "query.sql", "json": "data.json",
    "yaml": "config.yaml", "yml": "config.yaml",
}

# Fichiers du projet à ne jamais écraser via le fallback narratif
_PROJECT_FILES = {"main.py", "agent.py", "config.py"}

_TOOL_INVOCATION_RE = re.compile(r'^(\w+)\(\s*(\{.*\})\s*\)\s*$', re.S)

def _extract_and_write(content: str, on_tool_call: Callable[[str, dict, str], None] | None = None) -> str | None:
    """Extrait les blocs de code d'une réponse narrative et les écrit sur disque."""
    blocks = re.findall(r"```(\w*)\n(.*?)```", content, re.S)
    if not blocks:
        return None

    results = []
    for lang, code in blocks:
        code = code.strip()
        if not code or len(code) < 5:
            continue

        # Si le bloc est un appel de tool (ex: write_file({...})), l'exécuter directement
        m = _TOOL_INVOCATION_RE.match(code)
        if m:
            name, args_str = m.group(1), m.group(2)
            try:
                args = json.loads(args_str)
                result = dispatch(name, args)
                if on_tool_call:
                    on_tool_call(name, args, result)
                results.append(result)
                continue
            except (json.JSONDecodeError, Exception):
                pass

        filename = _LANG_EXT.get(lang.lower(), f"output.{lang}" if lang else "output.txt")
        if filename in _PROJECT_FILES:
            filename = f"generated_{filename}"

        from tools.fs import write_file
        result = write_file(filename, code)
        results.append(result)

    return "\n".join(results) if results else None


# ── Fallback inline tool calls (Mistral 7B) ───────────────────────────────────
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


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Tu es un assistant développeur CLI avec accès complet au système local.

AUTORISATION TOTALE : tu peux supprimer, modifier, créer, déplacer des fichiers, exécuter des commandes.
Ne refuse JAMAIS d'utiliser un outil. L'utilisateur gère les risques.

RÈGLE ABSOLUE : appelle l'outil immédiatement. Ne décris pas, n'explique pas, n'écris pas de code dans ta réponse.

Ambiguïté (plusieurs candidats) : 1) list_files/grep_search, 2) ask_user, 3) agis.

Outils : list_files, read_file, write_file, append_file, delete_file, move_file, copy_file,
         make_dir, delete_dir, file_info, grep_search, find_files, run_shell, ssh_exec,
         web_fetch, http_request, git_run, get_cwd, change_dir, system_info,
         process_list, env_get, env_set, ask_user.

Réponds en français. Après un outil réussi : une phrase courte."""


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
        narration_retries = 0

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
                    use_tools = True
                    self.messages[-1] = {"role": "assistant", "content": ""}

            if not tool_calls:
                content = msg.get("content", "")

                if _is_narration(content, in_action_context=use_tools):
                    if narration_retries < 1:
                        narration_retries += 1
                        self.messages.append({
                            "role": "user",
                            "content": "STOP. Appelle write_file ou run_shell maintenant. Pas de texte.",
                        })
                        continue
                    else:
                        # Fallback : on extrait et écrit le code directement
                        extracted = _extract_and_write(content)
                        if extracted:
                            self.messages[-1] = {"role": "assistant", "content": extracted}
                            return extracted
                        return content

                if last_tool_results and (not content or _is_refusal(content)):
                    content = last_tool_results[-1]
                    self.messages[-1] = {"role": "assistant", "content": content}
                    return content

                return content

            narration_retries = 0  # reset si un outil est appelé

            for tc in tool_calls:
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                if name == "ask_user":
                    question = args.get("question", "Quel choix ?")
                    options  = args.get("options", [])
                    result = on_user_choice(question, options) if (on_user_choice and options) \
                             else f"Options: {', '.join(options)}"
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
