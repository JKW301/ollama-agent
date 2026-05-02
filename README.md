# Ollama CLI Agent

Un agent CLI puissant basé sur Ollama pour interagir avec votre système local.

## Description

Ollama Agent est un assistant développeur en ligne de commande qui permet d'exécuter des tâches systèmes complexes via une interface conversationnelle. Il prend en charge :

- **Gestion de fichiers** : lecture, écriture, suppression, copie, déplacement
- **Exécution de commandes shell** : lancement de processus, scripts, compilation
- **Recherche dans le code** : grep, recherche de fichiers
- **Gestion Git** : commits, push, pull, gestion de dépôts
- **Gestion de l'environnement** : variables d'environnement, informations système

## Prérequis

- Python 3.10+
- [Ollama](https://ollama.ai/) installé et fonctionnel
- Un modèle compatible chargé (par défaut : `mistral-nemo:latest`)

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/JKW301/ollama-agent.git
cd ollama-agent

# Installer les dépendances
pip install -r requirements.txt
```

## Dépendances

```
ollama
rich
```

## Configuration

Modifiez `config.py` pour adapter les paramètres à votre environnement :

```python
MODEL = "mistral-nemo:latest"      # Modèle Ollama à utiliser
MAX_STEPS = 15                     # Nombre maximum d'étapes par requête
TEMPERATURE = 0.0                 # Températura de génération (0.0 = déterministe)
NUM_PREDICT = 128                 # Nombre de tokens générés par requête
SHELL_TIMEOUT = 30                # Timeout pour les commandes shell (secondes)
MAX_OUTPUT_CHARS = 8000           # Longueur maximale de la sortie
```

## Utilisation

```bash
python main.py
```

### Commandes spéciales

- `reset` : Réinitialise la session (efface le contexte)
- `quit` / `exit` / `q` : Quitte l'application
- `Ctrl+C` : Interrompt la génération en cours

### Outils disponibles

| Outil | Description |
|-------|-------------|
| `list_files` | Liste les fichiers d'un répertoire |
| `read_file` | Lit le contenu d'un fichier |
| `write_file` | Écrit dans un fichier |
| `append_file` | Ajoute du contenu à un fichier |
| `delete_file` | Supprime un fichier |
| `move_file` | Déplace un fichier |
| `copy_file` | Copie un fichier |
| `make_dir` | Crée un répertoire |
| `delete_dir` | Supprime un répertoire |
| `file_info` | Affiche les informations d'un fichier |
| `grep_search` | Recherche un motif dans des fichiers |
| `find_files` | Trouve des fichiers par nom |
| `run_shell` | Exécute une commande shell |
| `ssh_exec` | Exécute une commande via SSH |
| `web_fetch` | Récupère le contenu d'une URL |
| `http_request` | Effectue une requête HTTP |
| `git_run` | Exécute une commande Git |
| `get_cwd` | Récupère le répertoire courant |
| `change_dir` | Change de répertoire |
| `system_info` | Affiche les informations système |
| `process_list` | Liste les processus en cours |
| `env_get` | Récupère une variable d'environnement |
| `env_set` | Définit une variable d'environnement |
| `ask_user` | Pose une question à l'utilisateur |

### Exemples

```
>>> Liste les fichiers dans le répertoire courant
>>> Crée un fichier test.py avec du code Python
>>> Exécute la commande git status
>>> Cherche tous les fichiers .py dans src/
```

## Personnalisation

### Ajouter un outil

1. Définissez votre outil dans `tools.py` avec son schéma
2. Implémentez la fonction correspondante dans le dossier `tools/`
3. Ajoutez-le à la liste `SCHEMAS`

### Confirmer certains outils

Dans `config.py`, ajoutez le nom de l'outil à `CONFIRM_TOOLS` pour demander une confirmation avant exécution :

```python
CONFIRM_TOOLS = {"run_shell", "ssh_exec", "delete_file", "delete_dir", "mon_outil"}
```

## Avertissements

⚠️ **Cet agent a un accès complet à votre système.**

- Il peut supprimer, modifier ou créer des fichiers
- Il peut exécuter des commandes shell arbitraires
- Il peut accéder à votre réseau
- **Utilisez à vos propres risques**

L'auteur décline toute responsabilité en cas de dommages, perte de données ou autres problèmes résultant de l'utilisation de cet outil.

## Auteur

**JKW301**

## Licence

Ce projet est protégé par une licence **Tous droits réservés**.
Voir le fichier [LICENSE](LICENSE) pour plus de détails.
