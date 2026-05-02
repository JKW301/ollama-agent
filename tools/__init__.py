import json

from tools.fs import list_files, read_file, write_file, append_file
from tools.shell import run_shell
from tools.ssh import ssh_exec

FUNCTIONS: dict = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "append_file": append_file,
    "run_shell": run_shell,
    "ssh_exec": ssh_exec,
}

SCHEMAS: list = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Liste les fichiers d'un répertoire",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Chemin du répertoire (défaut: répertoire courant)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lit le contenu d'un fichier texte",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Chemin du fichier à lire"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Écrit du contenu dans un fichier (crée ou écrase si existant)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Chemin du fichier"},
                    "content": {"type": "string", "description": "Contenu à écrire"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": "Ajoute du contenu à la fin d'un fichier existant",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Chemin du fichier"},
                    "content": {"type": "string", "description": "Contenu à ajouter"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Exécute une commande shell (bash) sur la machine locale",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Commande shell à exécuter"},
                    "timeout": {"type": "integer", "description": "Timeout en secondes (défaut: 30)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ssh_exec",
            "description": "Exécute une commande sur un serveur distant via SSH",
            "parameters": {
                "type": "object",
                "properties": {
                    "host": {"type": "string", "description": "Adresse IP ou hostname du serveur"},
                    "user": {"type": "string", "description": "Nom d'utilisateur SSH"},
                    "command": {"type": "string", "description": "Commande à exécuter"},
                    "port": {"type": "integer", "description": "Port SSH (défaut: 22)"},
                },
                "required": ["host", "user", "command"],
            },
        },
    },
]


def dispatch(name: str, args: dict | str) -> str:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if name not in FUNCTIONS:
        return f"[ERR] Outil inconnu: {name}"
    try:
        return str(FUNCTIONS[name](**args))
    except TypeError as e:
        return f"[ERR] Arguments invalides pour {name}: {e}"
    except Exception as e:
        return f"[ERR] {name}: {e}"
