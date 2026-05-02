import os
import platform
import subprocess
import sys
import psutil
from config import MAX_OUTPUT_CHARS


def get_cwd() -> str:
    """Retourne le répertoire de travail courant."""
    return os.getcwd()


def change_dir(path: str) -> str:
    """Change le répertoire de travail courant de l'agent."""
    try:
        expanded = os.path.expanduser(path)
        if not os.path.isabs(expanded):
            expanded = os.path.join(os.getcwd(), expanded)
        os.chdir(expanded)
        return f"Répertoire courant: {os.getcwd()}"
    except FileNotFoundError:
        return f"[ERR] Répertoire introuvable: {path}"
    except Exception as e:
        return f"[ERR] {e}"


def system_info() -> str:
    """Retourne les informations système (OS, CPU, RAM, disque, Python)."""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    lines = [
        f"OS       : {platform.system()} {platform.release()} ({platform.machine()})",
        f"Hostname : {platform.node()}",
        f"CPU      : {platform.processor()} — {os.cpu_count()} cœurs",
        f"RAM      : {mem.total // 1024**2} Mo total  |  {mem.available // 1024**2} Mo disponible  ({mem.percent}% utilisé)",
        f"Disque / : {disk.total // 1024**3} Go total  |  {disk.free // 1024**3} Go libre",
        f"Python   : {sys.version.split()[0]}",
        f"CWD      : {os.getcwd()}",
    ]
    return "\n".join(lines)


def process_list(filter_name: str = "") -> str:
    """Liste les processus en cours. Filtre optionnel par nom."""
    procs = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status"]):
        try:
            info = p.info
            if filter_name and filter_name.lower() not in info["name"].lower():
                continue
            procs.append(
                f"{info['pid']:6d}  {info['name'][:28]:<28}  {info['status']:<10}"
                f"  CPU {info['cpu_percent']:5.1f}%  MEM {info['memory_percent']:4.1f}%"
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not procs:
        return f"Aucun processus trouvé{' pour ' + filter_name if filter_name else ''}"
    header = f"{'PID':>6}  {'NOM':<28}  {'ÉTAT':<10}  CPU       MEM"
    output = header + "\n" + "\n".join(procs[:60])
    if len(procs) > 60:
        output += f"\n... et {len(procs) - 60} autres"
    return output


def env_get(name: str) -> str:
    """Retourne la valeur d'une variable d'environnement."""
    val = os.environ.get(name)
    if val is None:
        return f"[ERR] Variable '{name}' non définie"
    return f"{name}={val}"


def env_set(name: str, value: str) -> str:
    """Définit une variable d'environnement pour la session courante."""
    os.environ[name] = value
    return f"Défini: {name}={value}"
