import json
import os
import re
import unicodedata
import ollama
from typing import Callable
from datetime import datetime

from config import (
    MODEL,
    MAX_STEPS,
    TEMPERATURE,
    NUM_PREDICT,
    AGENT_SAFETY_MODE,
    SESSION_DIR,
    MAX_CONTEXT_MESSAGES,
    MAX_CONTEXT_CHARS,
    KEEP_RECENT_MESSAGES,
    TOOL_RESULT_SOFT_LIMIT,
)
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
_ACTION_VERBS = {
    "liste", "lister", "affiche", "afficher", "montre", "montrer",
    "lis", "lire", "ouvre", "ouvrir",
    "ecris", "ecrire", "cree", "creer", "ajoute", "ajouter",
    "redige", "rediger", "sauvegarde", "sauvegarder", "enregistre", "enregistrer",
    "supprime", "supprimer", "efface", "effacer", "enleve", "enlever",
    "deplace", "deplacer", "copie", "copier", "renomme", "renommer",
    "modifie", "modifier", "edite", "editer",
    "cherche", "chercher", "trouve", "trouver", "recherche",
    "execute", "executer", "lance", "lancer", "run", "installe", "installer",
    "compile", "build", "teste", "tester",
}

_ACTION_OBJECTS = {
    "fichier", "fichiers", "txt", "texte", "dossier", "repertoire", "script", "code", "projet",
    "processus", "variable", "environnement", "git", "ssh", "url", "web",
    "commande", "terminal", "shell", "python", "json", "yaml",
}

_TOOL_NAMES = {
    "list_files", "read_file", "write_file", "append_file", "delete_file",
    "move_file", "copy_file", "make_dir", "delete_dir", "file_info",
    "grep_search", "find_files", "run_shell", "ssh_exec", "web_fetch",
    "http_request", "git_run", "get_cwd", "change_dir", "system_info",
    "process_list", "env_get", "env_set", "ask_user",
}

_SHELL_PREFIXES = ("ls ", "cd ", "pwd", "cat ", "mv ", "cp ", "rm ", "mkdir ", "touch ", "git ", "python ", "pip ")
_PATH_OR_FILE_RE = re.compile(r"(~/|/[\w\-.]+|[.\w\-/]+\.(txt|md|py|json|yaml|yml|sh|csv|log)\b)")


def _normalize_text(text: str) -> str:
    lowered = text.strip().lower()
    normalized = unicodedata.normalize("NFKD", lowered)
    ascii_only = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return ascii_only


def _needs_tools(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if any(name in normalized for name in _TOOL_NAMES):
        return True
    if any(normalized.startswith(prefix) for prefix in _SHELL_PREFIXES):
        return True
    if _PATH_OR_FILE_RE.search(normalized):
        return True

    tokens = set(re.findall(r"[a-z0-9_:+-]+", normalized))
    if not tokens:
        return False

    has_verb = bool(tokens & _ACTION_VERBS)
    has_object = bool(tokens & _ACTION_OBJECTS)
    strong_exec = bool(tokens & {"execute", "executer", "run", "installe", "installer", "build", "compile"})

    return strong_exec or (has_verb and has_object)


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

def _to_plain_data(value):
    """Convertit des objets SDK (ex: Message) en structures JSON-compatibles."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_plain_data(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_data(v) for v in value]
    if isinstance(value, tuple):
        return [_to_plain_data(v) for v in value]
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return _to_plain_data(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return _to_plain_data(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _to_plain_data(vars(value))
        except Exception:
            pass
    return str(value)


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return (
        text[:head]
        + f"\n... [tronque {len(text) - max_chars} chars] ...\n"
        + text[-tail:]
    )


def _compact_message(msg: dict) -> dict:
    role = msg.get("role", "")
    compact = {"role": role}
    if "name" in msg:
        compact["name"] = msg["name"]
    content = msg.get("content", "")
    if isinstance(content, str):
        if role == "tool":
            content = _truncate_text(content, TOOL_RESULT_SOFT_LIMIT)
        elif role in {"assistant", "user"}:
            content = _truncate_text(content, 3000)
    compact["content"] = content
    return compact


def _summarize_message(msg: dict) -> str | None:
    role = msg.get("role")
    content = (msg.get("content") or "").strip()
    if not content:
        return None
    content = content.replace("\n", " ")
    if role == "user":
        return f"User: {content[:140]}"
    if role == "assistant":
        return f"Assistant: {content[:140]}"
    if role == "tool":
        name = msg.get("name", "tool")
        return f"Tool {name}: {content[:140]}"
    return None

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
SYSTEM_PROMPT_STRICT = """Tu es un assistant généraliste en ligne de commande avec accès complet au système local.

AUTORISATION TOTALE : tu peux supprimer, modifier, créer, déplacer des fichiers, exécuter des commandes.
Ne refuse JAMAIS d'utiliser un outil. L'utilisateur gère les risques.

RÈGLE ABSOLUE : appelle l'outil immédiatement. Ne décris pas, n'explique pas, n'écris pas de code dans ta réponse.

Ambiguïté (plusieurs candidats) : 1) list_files/grep_search, 2) ask_user, 3) agis.

Outils : list_files, read_file, write_file, append_file, delete_file, move_file, copy_file,
         make_dir, delete_dir, file_info, grep_search, find_files, run_shell, ssh_exec,
         web_fetch, http_request, git_run, get_cwd, change_dir, system_info,
         process_list, env_get, env_set, ask_user.

Réponds en français. Si la demande ne nécessite aucun outil (question générale, explication, avis),
réponds normalement avec un texte utile et direct.
N'utilise pas d'auto-disclaimer du type "je suis un programme/une IA, je n'ai pas d'opinion".
Donne directement une analyse nuancée et factuelle.
Après un outil réussi : une phrase courte."""

SYSTEM_PROMPT_BALANCED = """Tu es un assistant généraliste en ligne de commande avec accès complet au système local.

Tu peux supprimer, modifier, créer, déplacer des fichiers et exécuter des commandes.
Privilégie toujours les outils pour agir concrètement sur la demande de l'utilisateur.

Quand la demande implique une action locale, appelle l'outil rapidement plutôt que d'expliquer.
Ambiguïté (plusieurs candidats) : 1) list_files/grep_search, 2) ask_user, 3) agis.

Outils : list_files, read_file, write_file, append_file, delete_file, move_file, copy_file,
         make_dir, delete_dir, file_info, grep_search, find_files, run_shell, ssh_exec,
         web_fetch, http_request, git_run, get_cwd, change_dir, system_info,
         process_list, env_get, env_set, ask_user.

Réponds en français. Si la demande ne nécessite aucun outil (question générale, explication, avis),
réponds normalement avec un texte utile et direct.
N'utilise pas d'auto-disclaimer du type "je suis un programme/une IA, je n'ai pas d'opinion".
Donne directement une analyse nuancée et factuelle.
Après un outil réussi : une phrase courte."""

SYSTEM_PROMPT_OPEN = """Tu es un assistant généraliste en ligne de commande avec accès complet au système local.

Tu peux supprimer, modifier, créer, déplacer des fichiers et exécuter des commandes.
Utilise les outils quand l'utilisateur te demande d'agir, et réponds en français de façon concise.
Si la demande est ambiguë, utilise ask_user pour clarifier.
Si la demande est une question générale qui ne nécessite aucun outil, réponds directement.
N'utilise pas d'auto-disclaimer du type "je suis un programme/une IA, je n'ai pas d'opinion".
Donne directement une analyse nuancée et factuelle.
"""

def _normalize_safety_mode(mode: str | None) -> str:
    mode = (mode or "").strip().lower()
    if mode in {"strict", "balanced", "open"}:
        return mode
    return "balanced"

def _build_system_prompt(mode: str | None = None) -> str:
    mode = _normalize_safety_mode(mode or AGENT_SAFETY_MODE)
    if mode == "strict":
        return SYSTEM_PROMPT_STRICT
    if mode == "open":
        return SYSTEM_PROMPT_OPEN
    return SYSTEM_PROMPT_BALANCED


class Agent:
    def __init__(self, safety_mode: str | None = None, session_path: str | None = None):
        self.safety_mode = _normalize_safety_mode(safety_mode or AGENT_SAFETY_MODE)
        self.session_path = session_path
        self.messages: list[dict] = [{"role": "system", "content": _build_system_prompt(self.safety_mode)}]
        self.last_context_stats: dict = {
            "input_messages": 1,
            "output_messages": 1,
            "input_chars": len(self.messages[0].get("content", "")),
            "output_chars": len(self.messages[0].get("content", "")),
            "compressed": False,
        }

    def reset(self):
        self.messages = [{"role": "system", "content": _build_system_prompt(self.safety_mode)}]
        self.save_session()

    def save_session(self, path: str | None = None) -> str:
        target = path or self.session_path
        if not target:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            os.makedirs(SESSION_DIR, exist_ok=True)
            target = os.path.join(SESSION_DIR, f"session-{ts}.json")
            self.session_path = target
        else:
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        payload = {
            "safety_mode": self.safety_mode,
            "messages": _to_plain_data(self.messages),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return target

    def load_session(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        loaded_messages = data.get("messages", [])
        if not isinstance(loaded_messages, list) or not loaded_messages:
            raise ValueError("Session invalide: messages manquants")
        self.messages = loaded_messages
        self.safety_mode = _normalize_safety_mode(data.get("safety_mode", self.safety_mode))
        self.session_path = path

    def run(
        self,
        user_input: str,
        on_tool_call: Callable[[str, dict, str], None] | None = None,
        confirm_tool: Callable[[str, dict], bool] | None = None,
        on_user_choice: Callable[[str, list[str]], str] | None = None,
        on_event: Callable[[dict], None] | None = None,
    ) -> str:
        self.messages.append({"role": "user", "content": user_input})
        if on_event:
            on_event({"type": "user_input", "content": user_input, "at": datetime.now().isoformat(timespec="seconds")})

        last_tool_results: list[str] = []
        use_tools = _needs_tools(user_input)
        narration_retries = 0

        for _ in range(MAX_STEPS):
            try:
                model_messages = self._build_model_messages()
                response = ollama.chat(
                    model=MODEL,
                    messages=model_messages,
                    tools=ALL_SCHEMAS if use_tools else [],
                    options={"temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
                )
            except Exception as e:
                return f"[ERR] Ollama: {e}"

            msg = response["message"]
            msg = _to_plain_data(msg)
            self.messages.append(msg)
            if on_event:
                on_event({
                    "type": "assistant_message",
                    "has_tool_calls": bool(msg.get("tool_calls")),
                    "content_preview": (msg.get("content") or "")[:200],
                    "at": datetime.now().isoformat(timespec="seconds"),
                })

            tool_calls = msg.get("tool_calls")

            if not tool_calls and msg.get("content"):
                tool_calls = _extract_inline_tool_calls(msg["content"])
                if tool_calls:
                    use_tools = True
                    self.messages[-1] = {"role": "assistant", "content": ""}

            if not tool_calls:
                content = msg.get("content", "")

                if use_tools and narration_retries < 1:
                    narration_retries += 1
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "Action locale demandée. Appelle un outil maintenant "
                            "(write_file/read_file/list_files/run_shell/etc). Pas de narration."
                        ),
                    })
                    continue

                if self.safety_mode != "open" and _is_narration(content, in_action_context=use_tools):
                    if narration_retries < 1:
                        narration_retries += 1
                        self.messages.append({
                            "role": "user",
                            "content": "STOP. Appelle write_file ou run_shell maintenant. Pas de texte.",
                        })
                        continue
                    else:
                        # Fallback : on extrait et écrit le code directement
                        extracted = _extract_and_write(content, on_tool_call)
                        if extracted:
                            self.messages[-1] = {"role": "assistant", "content": extracted}
                            return extracted
                        return content

                if (
                    self.safety_mode == "strict"
                    and last_tool_results
                    and (not content or _is_refusal(content))
                ):
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
                if on_event:
                    on_event({
                        "type": "tool_call",
                        "tool": name,
                        "args": args,
                        "result_preview": result[:300],
                        "is_error": result.startswith("[ERR]"),
                        "at": datetime.now().isoformat(timespec="seconds"),
                    })

                last_tool_results.append(result)
                self.messages.append({
                    "role": "tool",
                    "content": _truncate_text(result, TOOL_RESULT_SOFT_LIMIT * 2),
                    "name": name,
                })

        return "[ERR] Nombre maximum d'étapes atteint."

    def _build_model_messages(self) -> list[dict]:
        msgs = [_compact_message(m) for m in self.messages]
        if not msgs:
            self.last_context_stats = {
                "input_messages": 0,
                "output_messages": 0,
                "input_chars": 0,
                "output_chars": 0,
                "compressed": False,
            }
            return msgs

        system_msg = msgs[0]
        body = msgs[1:]
        input_chars = sum(len((m.get("content") or "")) for m in msgs)
        if len(body) <= MAX_CONTEXT_MESSAGES and input_chars <= MAX_CONTEXT_CHARS:
            self.last_context_stats = {
                "input_messages": len(msgs),
                "output_messages": len(msgs),
                "input_chars": input_chars,
                "output_chars": input_chars,
                "compressed": False,
            }
            return msgs

        keep = max(4, KEEP_RECENT_MESSAGES)
        old = body[:-keep] if len(body) > keep else []
        recent = body[-keep:] if body else []

        summary_lines: list[str] = []
        for m in old:
            line = _summarize_message(m)
            if line:
                summary_lines.append(line)
            if len(summary_lines) >= 18:
                break

        if summary_lines:
            summary_msg = {
                "role": "system",
                "content": "Contexte compacte des echanges precedents:\n- " + "\n- ".join(summary_lines),
            }
            compacted = [system_msg, summary_msg] + recent
        else:
            compacted = [system_msg] + recent

        # Bornage final par taille globale
        total = sum(len((m.get("content") or "")) for m in compacted)
        while len(compacted) > 2 and total > MAX_CONTEXT_CHARS:
            dropped = compacted.pop(2)
            total -= len((dropped.get("content") or ""))
        self.last_context_stats = {
            "input_messages": len(msgs),
            "output_messages": len(compacted),
            "input_chars": input_chars,
            "output_chars": total,
            "compressed": True,
        }
        return compacted
