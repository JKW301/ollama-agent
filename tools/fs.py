import os
from config import MAX_OUTPUT_CHARS


def _expand(path: str) -> str:
    p = os.path.expanduser(path)
    if not os.path.isabs(p):
        p = os.path.join(os.getcwd(), p)
    return p


def list_files(path: str = ".") -> str:
    """Liste les fichiers d'un répertoire."""
    try:
        full = _expand(path)
        if not os.path.isdir(full):
            return f"[ERR] Pas un répertoire: {full}"
        files = sorted(os.listdir(full))
        lines = "\n".join(f"  {f}" for f in files)
        return f"Contenu de {full}:\n{lines}"
    except PermissionError:
        return f"[ERR] Permission refusée: {path}"
    except Exception as e:
        return f"[ERR] {e}"


def read_file(path: str) -> str:
    """Lit le contenu d'un fichier texte."""
    try:
        full = _expand(path)
        with open(full, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > MAX_OUTPUT_CHARS:
            content = content[:MAX_OUTPUT_CHARS] + f"\n... [tronqué — {len(content)} chars total]"
        return content
    except FileNotFoundError:
        return f"[ERR] Fichier introuvable: {path}"
    except PermissionError:
        return f"[ERR] Permission refusée: {path}"
    except Exception as e:
        return f"[ERR] {e}"


def write_file(path: str, content: str) -> str:
    """Écrit du contenu dans un fichier (crée ou écrase)."""
    try:
        full = _expand(path)
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Fichier écrit: {full} ({len(content)} chars)"
    except Exception as e:
        return f"[ERR] {e}"


def append_file(path: str, content: str) -> str:
    """Ajoute du contenu à la fin d'un fichier existant."""
    try:
        full = _expand(path)
        with open(full, "a", encoding="utf-8") as f:
            f.write(content)
        return f"Contenu ajouté à: {full}"
    except Exception as e:
        return f"[ERR] {e}"
