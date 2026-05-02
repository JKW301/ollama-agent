import functions.execute_code as execute_code

def help():
    print("La commande /help affiche cette aide.")

# Écrire le code dans le fichier main.py
execute_code.write_file({
  "path": "/Users/julien/Git_Repos/ollama-agent/main.py",
  "content": "def help():\n    print(\"La commande /help affiche cette aide.\")\n"
})