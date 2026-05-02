import os
import shutil
from datetime import datetime
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


def move_file(src: str, dst: str) -> str:
    """Déplace ou renomme un fichier ou répertoire."""
    try:
        shutil.move(_expand(src), _expand(dst))
        return f"Déplacé: {src} → {dst}"
    except Exception as e:
        return f"[ERR] {e}"


def copy_file(src: str, dst: str) -> str:
    """Copie un fichier (préserve les métadonnées)."""
    try:
        shutil.copy2(_expand(src), _expand(dst))
        return f"Copié: {src} → {dst}"
    except Exception as e:
        return f"[ERR] {e}"


def make_dir(path: str) -> str:
    """Crée un répertoire et ses parents si nécessaire."""
    try:
        full = _expand(path)
        os.makedirs(full, exist_ok=True)
        return f"Répertoire créé: {full}"
    except Exception as e:
        return f"[ERR] {e}"


def delete_dir(path: str) -> str:
    """Supprime un répertoire et tout son contenu de façon récursive."""
    try:
        full = _expand(path)
        if not os.path.isdir(full):
            return f"[ERR] Pas un répertoire: {full}"
        shutil.rmtree(full)
        return f"Répertoire supprimé: {full}"
    except Exception as e:
        return f"[ERR] {e}"


def file_info(path: str) -> str:
    """Retourne les métadonnées d'un fichier (taille, dates, permissions)."""
    try:
        full = _expand(path)
        s = os.stat(full)
        size = s.st_size
        if size >= 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.1f} Mo"
        elif size >= 1024:
            size_str = f"{size / 1024:.1f} Ko"
        else:
            size_str = f"{size} octets"
        mtime = datetime.fromtimestamp(s.st_mtime).strftime("%d/%m/%Y %H:%M:%S")
        ctime = datetime.fromtimestamp(s.st_ctime).strftime("%d/%m/%Y %H:%M:%S")
        kind = "répertoire" if os.path.isdir(full) else "fichier"
        perms = oct(s.st_mode)[-3:]
        return (f"Type: {kind}\nChemin: {full}\nTaille: {size_str}\n"
                f"Modifié: {mtime}\nCréé: {ctime}\nPermissions: {perms}")
    except Exception as e:
        return f"[ERR] {e}"


def delete_file(path: str) -> str:
    """Supprime un fichier."""
    try:
        full = _expand(path)
        if not os.path.exists(full):
            return f"[ERR] Fichier introuvable: {full}"
        if os.path.isdir(full):
            return f"[ERR] {full} est un répertoire, pas un fichier"
        os.remove(full)
        return f"Fichier supprimé: {full}"
    except Exception as e:
        return f"[ERR] {e}"
