import os
import sys
import time
import threading
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from agent import Agent
from config import MODEL, CONFIRM_TOOLS

MOIS = ["janvier","février","mars","avril","mai","juin",
        "juillet","août","septembre","octobre","novembre","décembre"]

def now_fr() -> str:
    d = datetime.now()
    return f"{d.day:02d} {MOIS[d.month-1]} {d.year}  {d.hour:02d}h{d.minute:02d}"

console = Console()

BANNER = f"""[cyan]              NNNN[black]..............[/black]NN[black]......[/black]NN[/cyan]
[cyan]            NN[black]....[/black]NN[black]..........[/black]NN[black]..[/black]NN[black]..[/black]NN[black]..[/black]NN[/cyan]
[cyan]          cNN..NNNN[black]............[/black]NN[black]....[/black]NN[black]....[/black]NN[/cyan]
[cyan]        XNO..NN[black]........[/black]NNNNNN..NN[black]..........[/black]NN[/cyan]
[cyan]        XNO.:NN[black]......[/black]NN[black]......[/black]NNNN..NN[black]..[/black]NN[black]..[/black]NN[/cyan]
[cyan]        XNO.cNN....NN[black]........[/black]NN[black]......[/black]NN[black]......[/black]NN[/cyan]
[cyan]        XNO...,NNNNNN....NN[black]....[/black]NNNNNN[black]....[/black]NNNNNN[/cyan]
[cyan]          cNNNc...,NN[black]......[/black]NN[black]........[/black]NN..NN[/cyan]
[cyan]            ,NNNNNN[black]........[/black]NN[black]........[/black]NN..NN[/cyan]
[cyan]                  NNNNNNNNNNNNNNNNNNNNNN[/cyan]

[cyan]  Mistral CLI Agent  ·  {MODEL}[/cyan]"""


def display_tool_call(name: str, args: dict, result: str):
    args_text = "  ".join(f"[dim]{k}=[/]{str(v)[:80]!r}" for k, v in args.items())
    console.print(f"  [bold yellow]⚙[/] [yellow]{name}[/]  {args_text}")
    if result:
        first_line = result.strip().split("\n")[0][:200]
        color = "red" if result.startswith("[ERR]") else "green"
        console.print(f"    [dim]↳[/] [{color}]{first_line}[/{color}]")


def run():
    console.print(BANNER)
    console.print(f"[dim]Répertoire: {os.getcwd()}[/]")
    console.print("[dim]Commandes: 'reset' pour nouvelle session · 'quit' pour quitter · Ctrl+C pour interrompre[/]\n")

    agent = Agent()

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]>>>[/]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

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


if __name__ == "__main__":
    run()
