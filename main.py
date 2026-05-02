import os
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.columns import Columns
from rich.text import Text

from agent import Agent
from config import MODEL, CONFIRM_TOOLS

console = Console()

BANNER = f"""[bold cyan]
  ╔══════════════════════════════════════╗
  ║   Ollama Agent  ·  {MODEL:<18} ║
  ╚══════════════════════════════════════╝
[/]"""


def display_tool_call(name: str, args: dict, result: str):
    args_text = "  ".join(f"[dim]{k}=[/]{str(v)[:120]!r}" for k, v in args.items())
    console.print(f"  [bold yellow]⚙[/] [yellow]{name}[/]  {args_text}")


def run():
    console.print(BANNER)
    console.print(f"[dim]Répertoire: {os.getcwd()}[/]")
    console.print("[dim]Commandes: 'reset' pour nouvelle session · 'quit' pour quitter[/]\n")

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

        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Bye.[/]")
            break

        if user_input.lower() == "reset":
            agent.reset()
            console.print("[dim]Session réinitialisée.[/]\n")
            continue

        with console.status("[bold cyan]Réflexion...[/]", spinner="dots") as status:

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

            response = agent.run(user_input, on_tool_call=on_tool, confirm_tool=confirm)

        console.print(Panel(response, title="[green]Agent[/]", border_style="green", padding=(0, 1)))
        console.print()


if __name__ == "__main__":
    run()
