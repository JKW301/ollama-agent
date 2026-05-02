import json
import time

from tools.fs import list_files, read_file, write_file, append_file, delete_file
from tools.fs import move_file, copy_file, make_dir, delete_dir, file_info
from tools.shell import run_shell
from tools.ssh import ssh_exec
from tools.search import grep_search, find_files
from tools.web import web_fetch, http_request
from tools.git import git_run
from tools.system import get_cwd, change_dir, system_info, process_list, env_get, env_set
from config import TOOL_RETRY_MAX_ATTEMPTS, TOOL_RETRY_DELAY_SECONDS

FUNCTIONS: dict = {
    # Fichiers
    "list_files":  list_files,
    "read_file":   read_file,
    "write_file":  write_file,
    "append_file": append_file,
    "delete_file": delete_file,
    "move_file":   move_file,
    "copy_file":   copy_file,
    "make_dir":    make_dir,
    "delete_dir":  delete_dir,
    "file_info":   file_info,
    # Recherche
    "grep_search": grep_search,
    "find_files":  find_files,
    # Shell & SSH
    "run_shell":   run_shell,
    "ssh_exec":    ssh_exec,
    # Web
    "web_fetch":   web_fetch,
    "http_request": http_request,
    # Git
    "git_run":     git_run,
    # Système
    "get_cwd":     get_cwd,
    "change_dir":  change_dir,
    "system_info": system_info,
    "process_list": process_list,
    "env_get":     env_get,
    "env_set":     env_set,
}

def _fn(name: str, desc: str, props: dict, required: list = None):
    return {"type": "function", "function": {"name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required or []}}}

def _s(d): return {"type": "string",  "description": d}
def _i(d): return {"type": "integer", "description": d}
def _b(d): return {"type": "boolean", "description": d}
def _o(d): return {"type": "object",  "description": d}
def _a(d): return {"type": "array", "items": {"type": "string"}, "description": d}

SCHEMAS: list = [
    _fn("list_files",  "Liste les fichiers d'un dossier",          {"path": _s("dossier (défaut: .)")}),
    _fn("read_file",   "Lit un fichier texte",                      {"path": _s("chemin")}, ["path"]),
    _fn("write_file",  "Écrit/crée un fichier",                    {"path": _s("chemin"), "content": _s("contenu")}, ["path", "content"]),
    _fn("append_file", "Ajoute du texte à la fin d'un fichier",    {"path": _s("chemin"), "content": _s("texte à ajouter")}, ["path", "content"]),
    _fn("delete_file", "Supprime un fichier",                      {"path": _s("chemin")}, ["path"]),
    _fn("move_file",   "Déplace ou renomme un fichier",            {"src": _s("source"), "dst": _s("destination")}, ["src", "dst"]),
    _fn("copy_file",   "Copie un fichier",                         {"src": _s("source"), "dst": _s("destination")}, ["src", "dst"]),
    _fn("make_dir",    "Crée un dossier",                          {"path": _s("chemin")}, ["path"]),
    _fn("delete_dir",  "Supprime un dossier et son contenu",       {"path": _s("chemin")}, ["path"]),
    _fn("file_info",   "Métadonnées d'un fichier (taille, dates)", {"path": _s("chemin")}, ["path"]),
    _fn("grep_search", "Cherche un pattern dans les fichiers",     {"pattern": _s("texte ou regex"), "path": _s("dossier (défaut: .)"), "recursive": _b("récursif"), "case_sensitive": _b("casse")}, ["pattern"]),
    _fn("find_files",  "Trouve des fichiers par nom (glob *.py)",  {"name_pattern": _s("pattern glob"), "directory": _s("dossier"), "recursive": _b("récursif")}, ["name_pattern"]),
    _fn("run_shell",   "Exécute une commande bash",                {"command": _s("commande"), "timeout": _i("timeout s")}, ["command"]),
    _fn("ssh_exec",    "Exécute une commande SSH",                  {"host": _s("hôte"), "user": _s("user"), "command": _s("cmd"), "port": _i("port")}, ["host", "user", "command"]),
    _fn("web_fetch",   "Télécharge une URL en texte",              {"url": _s("URL"), "max_chars": _i("limite chars")}, ["url"]),
    _fn("http_request","Requête HTTP (GET/POST/…)",                {"url": _s("URL"), "method": _s("méthode"), "headers": _o("headers"), "body": _s("body")}, ["url"]),
    _fn("git_run",     "Commande git sans le mot 'git'",           {"command": _s("ex: status, log -5, diff HEAD")}, ["command"]),
    _fn("get_cwd",     "Répertoire courant",                       {}),
    _fn("change_dir",  "Change le répertoire courant",             {"path": _s("chemin")}, ["path"]),
    _fn("system_info", "Infos système (OS, CPU, RAM, disque)",     {}),
    _fn("process_list","Liste les processus",                      {"filter_name": _s("filtre nom")}),
    _fn("env_get",     "Lit une variable d'environnement",         {"name": _s("nom")}, ["name"]),
    _fn("env_set",     "Définit une variable d'environnement",     {"name": _s("nom"), "value": _s("valeur")}, ["name", "value"]),
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
