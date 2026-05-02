import os
import re
import fnmatch
from config import MAX_OUTPUT_CHARS


def grep_search(pattern: str, path: str = ".", recursive: bool = True, case_sensitive: bool = False) -> str:
    """Recherche un pattern (texte ou regex) dans les fichiers. Retourne fichier:ligne: contenu."""
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"[ERR] Pattern regex invalide: {e}"

    search_path = os.path.expanduser(path)
    if not os.path.isabs(search_path):
        search_path = os.path.join(os.getcwd(), search_path)

    if os.path.isfile(search_path):
        file_list = [search_path]
    elif recursive:
        file_list = []
        for root, dirs, files in os.walk(search_path):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                file_list.append(os.path.join(root, f))
    else:
        file_list = [
            os.path.join(search_path, f)
            for f in os.listdir(search_path)
            if os.path.isfile(os.path.join(search_path, f))
        ]

    results = []
    for filepath in file_list:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if regex.search(line):
                        rel = os.path.relpath(filepath, os.getcwd())
                        results.append(f"{rel}:{i}: {line.rstrip()}")
        except (PermissionError, IsADirectoryError, OSError):
            continue

    if not results:
        return f"Aucun résultat pour '{pattern}'"
    output = "\n".join(results)
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + f"\n... [tronqué — {len(results)} correspondances au total]"
    return output


def find_files(name_pattern: str, directory: str = ".", recursive: bool = True) -> str:
    """Trouve des fichiers par nom ou glob (*.py, test_*, etc.)."""
    search_dir = os.path.expanduser(directory)
    if not os.path.isabs(search_dir):
        search_dir = os.path.join(os.getcwd(), search_dir)

    matches = []
    try:
        if recursive:
            for root, dirs, files in os.walk(search_dir):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for f in files:
                    if fnmatch.fnmatch(f, name_pattern):
                        matches.append(os.path.relpath(os.path.join(root, f), os.getcwd()))
        else:
            for f in os.listdir(search_dir):
                if fnmatch.fnmatch(f, name_pattern):
                    matches.append(os.path.relpath(os.path.join(search_dir, f), os.getcwd()))
    except Exception as e:
        return f"[ERR] {e}"

    if not matches:
        return f"Aucun fichier trouvé pour '{name_pattern}'"
    return f"{len(matches)} fichier(s):\n" + "\n".join(f"  {m}" for m in sorted(matches))
