import os
import sys
import time
import threading
import argparse
import json
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion

from agent import Agent
from config import (
    MODEL,
    CONFIRM_TOOLS,
    AGENT_SAFETY_MODE,
    SESSION_DIR,
    LOG_DIR,
    AUTO_SAVE_SESSION,
)

MOIS = ["janvier","février","mars","avril","mai","juin",
        "juillet","août","septembre","octobre","novembre","décembre"]

def now_fr() -> str:
    d = datetime.now()
    return f"{d.day:02d} {MOIS[d.month-1]} {d.year}  {d.hour:02d}h{d.minute:02d}"

console = Console()
SLASH_COMMANDS = [
    ("/stats", "Afficher les stats de la session courante"),
    ("/stats all", "Afficher les stats globales (tous les logs)"),
    ("/reset", "Réinitialiser la session"),
    ("/quit", "Quitter l'application"),
]


class SlashCompleter(Completer):
    def __init__(self, commands: list[tuple[str, str]]):
        self.commands = commands

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.strip()
        if not text.startswith("/"):
            return
        for cmd, desc in self.commands:
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=cmd,
                    display_meta=desc,
                )

def render_banner(safety_mode: str) -> str:
    return f"""[cyan]              NNNN[black]..............[/black]NN[black]......[/black]NN[/cyan]
[cyan]            NN[black]....[/black]NN[black]..........[/black]NN[black]..[/black]NN[black]..[/black]NN[black]..[/black]NN[/cyan]
[cyan]          cNN..NNNN[black]............[/black]NN[black]....[/black]NN[black]....[/black]NN[/cyan]
[cyan]        XNO..NN[black]........[/black]NNNNNN..NN[black]..........[/black]NN[/cyan]
[cyan]        XNO.:NN[black]......[/black]NN[black]......[/black]NNNN..NN[black]..[/black]NN[black]..[/black]NN[/cyan]
[cyan]        XNO.cNN....NN[black]........[/black]NN[black]......[/black]NN[black]......[/black]NN[/cyan]
[cyan]        XNO...,NNNNNN....NN[black]....[/black]NNNNNN[black]....[/black]NNNNNN[/cyan]
[cyan]          cNNNc...,NN[black]......[/black]NN[black]........[/black]NN..NN[/cyan]
[cyan]            ,NNNNNN[black]........[/black]NN[black]........[/black]NN..NN[/cyan]
[cyan]                  NNNNNNNNNNNNNNNNNNNNNN[/cyan]

[cyan]  Mistral CLI Agent  ·  {MODEL}[/cyan]
[cyan]  Safety mode        ·  {safety_mode}[/cyan]"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mistral CLI Agent")
    parser.add_argument(
        "-s",
        "--safety-mode",
        choices=["strict", "balanced", "open"],
        default=AGENT_SAFETY_MODE,
        help="Override du mode de sécurité (défaut: AGENT_SAFETY_MODE/config).",
    )
    parser.add_argument(
        "--session-file",
        default="",
        help="Chemin de session JSON (sauvegarde auto dans ce fichier).",
    )
    parser.add_argument(
        "--resume-session",
        default="",
        help="Charge une session JSON existante avant de commencer.",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Chemin du log JSONL (sinon auto dans .agent_logs/).",
    )
    parser.add_argument(
        "--no-session-save",
        action="store_true",
        help="Désactive la sauvegarde auto de session.",
    )
    return parser.parse_args()


def _default_log_path() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(LOG_DIR, f"run-{ts}.jsonl")


def _append_jsonl(path: str, payload: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def display_tool_call(name: str, args: dict, result: str):
    args_text = "  ".join(f"[dim]{k}=[/]{str(v)[:80]!r}" for k, v in args.items())
    console.print(f"  [bold yellow]⚙[/] [yellow]{name}[/]  {args_text}")
    if result:
        first_line = result.strip().split("\n")[0][:200]
        color = "red" if result.startswith("[ERR]") else "green"
        console.print(f"    [dim]↳[/] [{color}]{first_line}[/{color}]")


def run():
    args = parse_args()
    safety_mode = args.safety_mode
    session_file = args.session_file.strip() or None
    resume_session = args.resume_session.strip() or None
    log_file = args.log_file.strip() or _default_log_path()
    auto_save = AUTO_SAVE_SESSION and not args.no_session_save

    console.print(render_banner(safety_mode))
    console.print(f"[dim]Répertoire: {os.getcwd()}[/]")
    console.print(f"[dim]Logs JSONL: {log_file}[/]")
    console.print("[dim]Commandes: 'reset' pour nouvelle session · 'quit' pour quitter · Ctrl+C pour interrompre[/]\n")

    if session_file:
        os.makedirs(os.path.dirname(session_file) or ".", exist_ok=True)
    else:
        os.makedirs(SESSION_DIR, exist_ok=True)

    agent = Agent(safety_mode=safety_mode, session_path=session_file)

    if resume_session:
        try:
            agent.load_session(resume_session)
            console.print(f"[dim]Session chargée: {resume_session}[/]")
        except Exception as e:
            console.print(f"[bold red][ERR][/bold red] Impossible de charger la session: {e}")
    _append_jsonl(log_file, {
        "type": "session_start",
        "at": datetime.now().isoformat(timespec="seconds"),
        "cwd": os.getcwd(),
        "safety_mode": agent.safety_mode,
        "resume_session": resume_session or "",
    })

    while True:
        try:
            user_input = prompt_session.prompt(">>> ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input == "/":
            console.print("[dim]Utilise les flèches ↑/↓ pour choisir une commande slash, puis Entrée.[/]")
            console.print("[dim]Exemples: /stats, /stats all, /reset, /quit[/]\n")
            continue
        if user_input.lower() == "/quit":
            user_input = "quit"
        elif user_input.lower() == "/reset":
            user_input = "reset"

        sys.stdout.write("\033[1A\033[2K\r")
        sys.stdout.flush()
        console.print(Panel(
            user_input,
            title=f"[cyan]Vous[/]  [dim]{now_fr()}[/]",
            border_style="cyan",
            padding=(0, 1),
        ))

        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Bye.[/]")
            break

        if user_input.lower() == "reset":
            agent.reset()
            console.print("[dim]Session réinitialisée.[/]\n")
            if auto_save:
                path = agent.save_session()
                console.print(f"[dim]Session sauvée: {path}[/]")
            continue

        t0 = time.time()
        _stop_timer = threading.Event()

        with console.status("", spinner="dots") as status:

            def _run_timer():
                while not _stop_timer.is_set():
                    elapsed = time.time() - t0
                    status.update(f"[bold cyan]Réflexion...  {elapsed:.1f}s[/]")
                    time.sleep(0.1)

            _timer_thread = threading.Thread(target=_run_timer, daemon=True)
            _timer_thread.start()

            def on_tool(name: str, args: dict, result: str):
                status.stop()
                display_tool_call(name, args, result)
                status.start()

            def confirm(name: str, args: dict) -> bool:
                if name not in CONFIRM_TOOLS:
                    return True
                status.stop()
                args_str = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in args.items())
                console.print(f"\n[bold red]⚠ Confirmation[/] — [yellow]{name}[/]({args_str})")
                answer = Prompt.ask("Exécuter ?", choices=["o", "n"], default="n")
                status.start()
                return answer == "o"

            def user_choice(question: str, options: list[str]) -> str:
                status.stop()
                console.print(f"\n  [bold cyan]❓ {question}[/]")
                for i, opt in enumerate(options, 1):
                    console.print(f"     [cyan]{i}[/]  {opt}")
                choices = [str(i) for i in range(1, len(options) + 1)]
                idx = Prompt.ask("  Votre choix", choices=choices)
                selected = options[int(idx) - 1]
                console.print(f"  [dim]→ {selected}[/]\n")
                status.start()
                return selected

            try:
                response = agent.run(
                    user_input,
                    on_tool_call=on_tool,
                    confirm_tool=confirm,
                    on_user_choice=user_choice,
                    on_event=lambda evt: _append_jsonl(log_file, evt),
                )
            except KeyboardInterrupt:
                agent.reset()
                _stop_timer.set()
                _timer_thread.join(timeout=0.3)
                console.print("\n[bold yellow]⏹ Interrompu (Ctrl+C)[/]")
                continue

            _stop_timer.set()
            _timer_thread.join(timeout=0.3)

        total = time.time() - t0
        console.print(f"  [dim]↳ {total:.1f}s[/]")

        console.print(Panel(
            response,
            title=f"[green]Agent[/]  [dim]{now_fr()}[/]",
            border_style="green",
            padding=(0, 1),
        ))
        console.print()
        if auto_save:
            path = agent.save_session()
            console.print(f"[dim]Session sauvée: {path}[/]")


if __name__ == "__main__":
    run()
