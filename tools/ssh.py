import os
import paramiko


def ssh_exec(host: str, user: str, command: str, port: int = 22) -> str:
    """Exécute une commande sur un serveur distant via SSH (clé ~/.ssh/id_rsa)."""
    try:
        key_path = os.path.expanduser("~/.ssh/id_rsa")
        if not os.path.exists(key_path):
            return f"[ERR] Clé SSH introuvable: {key_path}"

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=host, port=port, username=user, key_filename=key_path)

        _, stdout, stderr = client.exec_command(command)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        client.close()

        parts = []
        if out:
            parts.append(out)
        if err:
            parts.append(f"[STDERR]\n{err}")
        return "\n".join(parts) or "[OK] Commande exécutée (pas de sortie)"
    except Exception as e:
        return f"[ERR] SSH {user}@{host}:{port} — {e}"
