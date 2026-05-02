MODEL = "mistral-nemo:latest"
MAX_STEPS = 15
TEMPERATURE = 0.0
NUM_PREDICT = 2048
SHELL_TIMEOUT = 30
MAX_OUTPUT_CHARS = 8000

# Tools qui demandent une confirmation avant exécution
CONFIRM_TOOLS = {"run_shell", "ssh_exec"}
