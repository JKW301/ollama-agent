import os
from pathlib import Path
from typing import Dict, Any

def read_file(path: str) -> str:
    """Lit le contenu d'un fichier."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Erreur: {str(e)}"

def write_file(path: str, content: str) -> str:
    """Écrit du contenu dans un fichier (écrase si existe)."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Fichier écrit avec succès : {path}"
    except Exception as e:
        return f"Erreur: {str(e)}"

def append_file(path: str, content: str) -> str:
    """Ajoute du contenu à la fin d'un fichier."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(content)
        return f"Contenu ajouté à : {path}"
    except Exception as e:
        return f"Erreur: {str(e)}"

def list_files(directory: str = ".") -> str:
    """Liste les fichiers d'un dossier."""
    try:
        return "\n".join(os.listdir(directory))
    except Exception as e:
        return f"Erreur: {str(e)}"

# Dictionnaire des tools pour Ollama
available_tools = {
    "read_file": read_file,
    "write_file": write_file,
    "append_file": append_file,
    "list_files": list_files,
}