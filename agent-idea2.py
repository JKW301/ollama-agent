import ollama
import json

def run_agent(query: str, max_steps=10):
    messages = [{"role": "user", "content": query}]
    
    for _ in range(max_steps):
        response = ollama.chat(
            model='mistral:latest',
            messages=messages,
            tools=[{
                "type": "function",
                "function": {
                    "name": name,
                    "description": func.__doc__,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Chemin du fichier"},
                            "content": {"type": "string", "description": "Contenu à écrire"},
                            "directory": {"type": "string", "description": "Dossier à lister"}
                        },
                        "required": ["path"] if name in ["read_file", "write_file"] else []
                    }
                }
            } for name, func in available_tools.items() if name != "list_files"] + [{
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "Liste les fichiers d'un dossier",
                    "parameters": {
                        "type": "object",
                        "properties": {"directory": {"type": "string"}}
                    }
                }
            }]
        )
        
        messages.append(response['message'])
        
        # Si le modèle veut appeler un tool
        if response['message'].get('tool_calls'):
            for tool_call in response['message']['tool_calls']:
                func_name = tool_call['function']['name']
                args = json.loads(tool_call['function']['arguments'])
                
                if func_name in available_tools:
                    result = available_tools[func_name](**args)
                    messages.append({
                        "role": "tool",
                        "content": str(result),
                        "name": func_name
                    })
        else:
            # Réponse finale du modèle
            print(response['message']['content'])
            break

# Exemple d'utilisation
run_agent("Crée un fichier test.py avec une fonction qui dit bonjour, puis lis-le.")