import os
import subprocess
from config import SHELL_TIMEOUT, MAX_OUTPUT_CHARS


def run_shell(command: str, timeout: int = SHELL_TIMEOUT) -> str:
    """Exécute une commande shell (bash) sur la machine locale et retourne stdout + stderr."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
        )
        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"[STDERR]\n{result.stderr}")
        parts.append(f"[exit {result.returncode}]")
        output = "\n".join(parts)
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n... [tronqué]"
        return output
    except subprocess.TimeoutExpired:
        return f"[ERR] Timeout après {timeout}s"
    except Exception as e:
        return f"[ERR] {e}"
