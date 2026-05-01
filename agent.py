import ollama
import re
import os
import sys
import paramiko

MODEL = "mistral:latest"

# ---------- LLM ----------

def ollama_chat(messages):
    """Send messages to Ollama using Python API and return clean response"""
    try:
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            stream=False,
            options={
                'temperature': 0.0,  # Deterministic for strict tool use (Mistral 7B best practice)
                'num_predict': 2048,
                'stop': ['[INST]', '[/INST]']  # Mistral 7B v0.3 stop tokens
            }
        )
        
        if 'message' in response and 'content' in response['message']:
            return response['message']['content']
        elif 'response' in response:
            return response['response']
        else:
            return f"[ERR] Réponse inattendue d'Ollama: {response}"
            
    except Exception as e:
        return f"[ERR] Erreur Ollama: {e}"

# ---------- TOOLS FS ----------

def file_list(path):
    """List files in a directory"""
    try:
        expanded_path = os.path.expanduser(path)
        if not os.path.isabs(expanded_path):
            expanded_path = os.path.join(os.getcwd(), expanded_path)
        
        if not os.path.isdir(expanded_path):
            return f"[ERR] {expanded_path} n'est pas un répertoire"
        
        files = os.listdir(expanded_path)
        return f"[OK] Contenu de {expanded_path}:\n" + "\n".join(f"- {f}" for f in sorted(files))
    except PermissionError:
        return f"[ERR] Permission refusée pour {expanded_path}"
    except Exception as e:
        return f"[ERR] Impossible de lister {expanded_path}: {e}"

def file_read(path):
    """Read file content"""
    try:
        # Expand ~ and relative paths
        expanded_path = os.path.expanduser(path)
        if not os.path.isabs(expanded_path):
            expanded_path = os.path.join(os.getcwd(), expanded_path)
        
        with open(expanded_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Truncate very large files
        if len(content) > 10000:
            content = content[:10000] + f"\n\n... [fichier tronqué, taille totale: {len(content)} chars]"
        return f"[OK] Lecture de {expanded_path}:\n" + content
    except FileNotFoundError:
        return f"[ERR] Fichier introuvable: {path}"
    except PermissionError:
        return f"[ERR] Permission refusée pour {path}"
    except Exception as e:
        return f"[ERR] Impossible de lire {path}: {e}"

def file_write(path, content):
    """Write content to file"""
    try:
        expanded_path = os.path.expanduser(path)
        if not os.path.isabs(expanded_path):
            expanded_path = os.path.join(os.getcwd(), expanded_path)
        
        os.makedirs(os.path.dirname(expanded_path), exist_ok=True)
        with open(expanded_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[OK] Fichier écrit: {expanded_path}"
    except Exception as e:
        return f"[ERR] Impossible d'écrire {path}: {e}"

def file_append(path, content):
    """Append content to file"""
    try:
        expanded_path = os.path.expanduser(path)
        if not os.path.isabs(expanded_path):
            expanded_path = os.path.join(os.getcwd(), expanded_path)
        
        with open(expanded_path, "a", encoding="utf-8") as f:
            f.write(content)
        return f"[OK] Contenu ajouté à: {expanded_path}"
    except Exception as e:
        return f"[ERR] Impossible d'ajouter à {path}: {e}"

# ---------- TOOL SSH ----------

def ssh_exec(host, user, command, port=22, key_path=None, password=None):
    """Execute SSH command"""
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        if key_path:
            key_path = os.path.expanduser(key_path)
            key = paramiko.RSAKey.from_private_key_file(key_path)
            client.connect(hostname=host, port=port, username=user, pkey=key)
        else:
            client.connect(hostname=host, port=port, username=user, password=password)

        stdin, stdout, stderr = client.exec_command(command)
        out = stdout.read().decode('utf-8', errors='replace')
        err = stderr.read().decode('utf-8', errors='replace')
        client.close()

        return f"[SSH OUT]\n{out}\n[SSH ERR]\n{err}"
    except Exception as e:
        return f"[ERR] SSH vers {user}@{host}: {e}"

# ---------- ACTION PARSING ----------

def detect_action(text):
    """Detect and parse action tags from model response
    
    Accepts ONLY exact XML tags (no surrounding text).
    Tolerates self-closing tags for write/append by converting to empty content.
    """
    text = text.strip()

    # Vérifie que c'est EXACTEMENT une balise (pas de texte avant/après)
    if not (text.startswith('<') and text.endswith('>')):
        return None

    # <list path="..." />
    m = re.fullmatch(r'<list\s+path="(.*?)"\s*/>', text)
    if m:
        return ("list", m.group(1))

    # <read path="..." />
    m = re.fullmatch(r'<read\s+path="(.*?)"\s*/>', text)
    if m:
        return ("read", m.group(1))

    # <write path="...">content</write> OR <write path="..." /> (tolerated -> empty content)
    m = re.fullmatch(r'<write\s+path="(.*?)"\s*/>', text)
    if m:
        return ("write", m.group(1), "")  # Tolerate self-closing: treat as empty file
    m = re.fullmatch(r'<write\s+path="(.*?)">(.*?)</write>', text, re.S)
    if m:
        return ("write", m.group(1), m.group(2))

    # <append path="...">content</append> OR <append path="..." /> (tolerated -> empty)
    m = re.fullmatch(r'<append\s+path="(.*?)"\s*/>', text)
    if m:
        return ("append", m.group(1), "")  # Tolerate self-closing: treat as empty append
    m = re.fullmatch(r'<append\s+path="(.*?)">(.*?)</append>', text, re.S)
    if m:
        return ("append", m.group(1), m.group(2))

    # <ssh host="x" user="y" port="22">command</ssh>
    m = re.fullmatch(r'<ssh\s+host="(.*?)"\s+user="(.*?)"(?:\s+port="(.*?)")?>(.*?)</ssh>', text, re.S)
    if m:
        host, user, port, cmd = m.group(1), m.group(2), m.group(3), m.group(4).strip()
        port = int(port) if port else 22
        return ("ssh", host, user, port, cmd)

    return None

# ---------- MAIN LOOP ----------

def main():
    system_prompt = """Tu es un assistant CLI.

Règles STRICTES (une SEULE chose par réponse):
1. Si demande de LISTER un répertoire → Réponds UNIQUEMENT: <list path="chemin" />
2. Si demande de LIR un fichier → Réponds UNIQUEMENT: <read path="chemin" />
3. Si demande d'ÉCRIRE un fichier → Réponds UNIQUEMENT: <write path="chemin">contenu</write>
4. Si demande d'AJOUTER à un fichier → Réponds UNIQUEMENT: <append path="chemin">texte</append>
5. Si demande SSH → Réponds UNIQUEMENT: <ssh host="ip" user="u" port="p">cmd</ssh>
6. SINON → Réponds UNIQUEMENT en français naturel, SANS AUCUNE balise XML

INTERDIT ABSOLUMENT:
- Mélanger texte et balise dans une même réponse
- Mentionner les balises XML dans les réponses naturelles
- Suggérer des commandes shell
- Utiliser balises auto-fermantes pour write/append

Utilise EXACTEMENT le chemin fourni."""
    messages = [{"role": "system", "content": system_prompt}]

    print(f"Agent local ({MODEL}) prêt. Tape une commande ou 'quit' pour sortir.\n")
    print(f"Répertoire courant: {os.getcwd()}\n")

    while True:
        try:
            user_input = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\n[EXIT]")
            break

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("[EXIT]")
            break

        messages.append({"role": "user", "content": user_input})
        
        print("\n[PENSER...]")
        response = ollama_chat(messages)
        print(f"\n[MODEL]\n{response}\n")

        action = detect_action(response)

        if action:
            kind = action[0]

            if kind == "list":
                path = action[1]
                result = file_list(path)
                print(f"[ACTION]\n{result}\n")
                messages.append({"role": "assistant", "content": result})

            elif kind == "read":
                path = action[1]
                result = file_read(path)
                print(f"[ACTION]\n{result}\n")
                messages.append({"role": "assistant", "content": result})

            elif kind == "write":
                path, content = action[1], action[2]
                result = file_write(path, content)
                print(f"[ACTION]\n{result}\n")
                messages.append({"role": "assistant", "content": result})

            elif kind == "append":
                path, content = action[1], action[2]
                result = file_append(path, content)
                print(f"[ACTION]\n{result}\n")
                messages.append({"role": "assistant", "content": result})

            elif kind == "ssh":
                host, user, port, cmd = action[1], action[2], action[3], action[4]
                default_key = os.path.expanduser("~/.ssh/id_rsa")
                if os.path.exists(default_key):
                    result = ssh_exec(host, user, cmd, port=port, key_path=default_key)
                else:
                    result = f"[ERR] Clé SSH par défaut introuvable: {default_key}"
                print(f"[ACTION]\n{result}\n")
                messages.append({"role": "assistant", "content": result})
        else:
            messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
