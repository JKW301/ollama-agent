import os

MODEL = "mistral-nemo:latest"
MAX_STEPS = 15
TEMPERATURE = 0.0
NUM_PREDICT = 128
SHELL_TIMEOUT = 30
MAX_OUTPUT_CHARS = 8000
TOOL_RETRY_MAX_ATTEMPTS = 2
TOOL_RETRY_DELAY_SECONDS = 0.35
SESSION_DIR = ".agent_sessions"
LOG_DIR = ".agent_logs"
AUTO_SAVE_SESSION = True

# Mode de sécurité de l'agent: strict | balanced | open
# - strict: blocage des refus et narrations activé
# - balanced: blocage des refus activé, narration plus tolérée
# - open: aucun blocage forcé, laisse le modèle répondre librement
AGENT_SAFETY_MODE = os.getenv("AGENT_SAFETY_MODE", "open").strip().lower()
if AGENT_SAFETY_MODE not in {"strict", "balanced", "open"}:
    AGENT_SAFETY_MODE = "balanced"

# Tools qui demandent une confirmation avant exécution
CONFIRM_TOOLS = {"run_shell", "ssh_exec", "delete_file", "delete_dir"}
