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
                'temperature': 0.1,  # More deterministic for tool use
                'num_predict': 2048
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
    """Detect and parse action tags from model response"""
    text = text.strip()

    # Try to find action tags anywhere in the text (not just full match)
    # <read path="/tmp/test.py" />
    m = re.search(r'<read\s+path="(.*?)"\s*/>', text, re.S)
    if m:
        return ("read", m.group(1))

    # <write path="/tmp/test.py">content</write>
    m = re.search(r'<write\s+path="(.*?)">(.*?)</write>', text, re.S)
    if m:
        return ("write", m.group(1), m.group(2))

    # <append path="/tmp/test.py">content</append>
    m = re.search(r'<append\s+path="(.*?)">(.*?)</append>', text, re.S)
    if m:
        return ("append", m.group(1), m.group(2))

    # <ssh host="x" user="y" port="22">command</ssh>
    m = re.search(r'<ssh\s+host="(.*?)"\s+user="(.*?)"(?:\s+port="(.*?)")?>(.*?)</ssh>', text, re.S)
    if m:
        host = m.group(1)
        user = m.group(2)
        port = int(m.group(3)) if m.group(3) else 22
        cmd = m.group(4).strip()
        return ("ssh", host, user, port, cmd)

    return None

# ---------- MAIN LOOP ----------

def main():
    system_prompt = """Tu es un agent CLI intelligent. Tu as DEUX modes de réponse:

MODE 1 - ACTIONS FILESYSTEM/SSH: Si la demande est explicitement une action sur le filesystem ou SSH, réponds UNIQUEMENT avec la balise XML correspondante. NE JAMAIS ajouter de texte.

Actions valides et leurs balises:
- Lire un fichier: <read path="chemin/fichier" />
- Écrire un fichier: <write path="chemin/fichier">contenu exact</write>
- Ajouter à un fichier: <append path="chemin/fichier">texte à ajouter</append>
- Commande SSH: <ssh host="IP" user="user" port="22">commande</ssh>

MODE 2 - QUESTIONS NORMALES: Pour TOUTE autre demande (questions, discussions, etc.), réponds normalement en français sans aucune balise XML.

EXEMPLES PRÉCIS:
--- ACTIONS (Mode 1) ---
Q: "Lis le fichier test.py" → R: <read path="test.py" />
Q: "Crée un fichier hello.txt avec Bonjour" → R: <write path="hello.txt">Bonjour</write>
Q: "Ajoute la ligne print(x) à script.py" → R: <append path="script.py">print(x)</append>
Q: "Fais un ls sur 192.168.1.1 avec user admin" → R: <ssh host="192.168.1.1" user="admin" port="22">ls</ssh>

--- QUESTIONS NORMALES (Mode 2) ---
Q: "salut" → R: "Bonjour ! Comment puis-je vous aider ?"
Q: "qui es-tu ?" → R: "Je suis un agent CLI qui peut interagir avec le filesystem et SSH."
Q: "Quelle heure est-il ?" → R: "Je ne peux pas accéder à l'heure système, mais..."
Q: "Explique moi comment fonctionne Python" → R: "Python est un langage de programmation..."

RÈGLES STRICTES:
1. Si la demande contient un verbe d'action (lis, crée, écris, ajoute, exécute, fais, SSH) + un fichier/serveur → Mode 1 (balise XML SEULE)
2. Sinon → Mode 2 (réponse normale)
3. NE JAMAIS mélanger les modes
4. NE JAMAIS inventer des fichiers ou des actions
5. Utilise chemins relatifs/absolus, ~ autorisé"""

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

            if kind == "read":
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
