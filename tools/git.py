import os
import subprocess
from config import SHELL_TIMEOUT, MAX_OUTPUT_CHARS


def git_run(command: str) -> str:
    """Exécute une commande git (sans le mot 'git'). Ex: 'status', 'log --oneline -10', 'diff HEAD'."""
    try:
        result = subprocess.run(
            f"git {command}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=SHELL_TIMEOUT,
            cwd=os.getcwd(),
        )
        parts = []
        if result.stdout:
            parts.append(result.stdout.rstrip())
        if result.stderr:
            parts.append(f"[stderr] {result.stderr.rstrip()}")
        if result.returncode != 0:
            parts.append(f"[exit {result.returncode}]")
        output = "\n".join(parts)
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n... [tronqué]"
        return output or "[OK] Commande git exécutée (pas de sortie)"
    except subprocess.TimeoutExpired:
        return f"[ERR] Timeout après {SHELL_TIMEOUT}s"
    except Exception as e:
        return f"[ERR] {e}"
