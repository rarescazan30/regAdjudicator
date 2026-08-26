"""
Interactive CLI Demonstration for regAdjudicator
Allows users to select from curated regulatory divergence scenarios
or input custom cross-jurisdictional queries with live trace visualization.
"""

import sys
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.markdown import Markdown

from src.agent import RegulatoryAgent
from src.config import GEMINI_AGENT_MODEL, GEMINI_VERIFIER_MODEL


console = Console()

SAMPLE_QUERIES = {
    "1": (
        "Lecanemab (Leqembi) Divergence",
        "Compare FDA and EMA approval scopes for Lecanemab (Leqembi). Did both agencies approve it and what restrictions exist regarding ApoE genotypes?",
    ),
    "2": (
        "Aducanumab (Aduhelm) Divergence",
        "Is Aducanumab (Aduhelm) authorized in both the US and EU? Explain why the two regulatory bodies reached different decisions.",
    ),
    "3": (
        "Bevacizumab (Avastin) Revocation",
        "What is the current regulatory status of Bevacizumab (Avastin) for metastatic breast cancer in the US versus the European Union?",
    ),
    "4": (
        "Olaparib (Lynparza) Indication Scope",
        "How do FDA and EMA indications differ for Olaparib (Lynparza) in metastatic castration-resistant prostate cancer (mCRPC)?",
    ),
    "5": (
        "Negative Control (Hallucination Test)",
        "What is the cross-jurisdictional regulatory approval status for the experimental oncological compound XYLOPHEN-99 for glioblastoma?",
    ),
}


def print_banner():
    """Prints the application header banner."""
    console.print(
        Panel.fit(
            "[bold cyan]regAdjudicator[/bold cyan] — [bold white]FDA/EMA Regulatory Divergence Detection Agent[/bold white]\n"
            f"[dim]Agent Model: [green]{GEMINI_AGENT_MODEL}[/green] | Verifier Model: [green]{GEMINI_VERIFIER_MODEL}[/green][/dim]\n"
            "[dim]Autonomous ReAct Loop • Two-Pass Grounding Verification • ChromaDB Vector RAG[/dim]",
            border_style="cyan",
        )
    )

def display_sample_query_menu():
    """Displays sample regulatory divergence scenarios."""
    table = Table(
        title="[bold cyan]Curated Regulatory Scenarios[/bold cyan]",
        border_style="cyan",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Key", style="bold yellow", justify="center", width=5)
    table.add_column("Scenario Focus", style="bold white", width=34)
    table.add_column("Query Preview", style="dim")

    for key, (title, query) in SAMPLE_QUERIES.items():
        table.add_row(f"[{key}]", title, query[:80] + ("..." if len(query) > 80 else ""))

    console.print("\n", table)


def display_menu():
    """Displays a clean, styled navigation banner."""
    console.print(
        Panel.fit(
            "💡 [bold white]Type your regulatory question[/bold white]  │  "
            "[[bold cyan]1-5[/bold cyan]] [dim]Instant Presets[/dim]  │  "
            "[[bold cyan]A[/bold cyan]] [dim]Browse Scenarios[/dim]  │  "
            "[[bold red]Q[/bold red]] [dim]Quit[/dim]",
            border_style="cyan",
            padding=(0, 2),
        )
    )


def run_interactive_demo():
    """Main interactive loop."""
    print_banner()

    try:
        agent = RegulatoryAgent()
    except Exception as e:
        console.print(f"[bold red]Error initializing agent:[/bold red] {e}")
        sys.exit(1)

    while True:
        display_menu()
        user_input = Prompt.ask("\n[bold cyan]Query / Option[/bold cyan]").strip()

        if not user_input:
            continue

        if user_input.upper() == "Q":
            console.print("\n[bold green]Thank you for using regAdjudicator. Goodbye![/bold green]\n")
            break

        query = ""
        # 1. Direct preset shortcut (typing 1-5 directly)
        if user_input in SAMPLE_QUERIES:
            query = SAMPLE_QUERIES[user_input][1]
            console.print(f"\n[dim]Selected preset [{user_input}]: {SAMPLE_QUERIES[user_input][0]}[/dim]")
            console.print(f"[bold cyan]Preset Query:[/bold cyan] {query}\n")
        # 2. Browse presets menu
        elif user_input.upper() == "A":
            display_sample_query_menu()
            question_choice = Prompt.ask("\n[bold cyan]Select Scenario (1-5)[/bold cyan]", default="1").strip()
            if question_choice not in SAMPLE_QUERIES:
                console.print("[red]Invalid selection. Returning to main menu.[/red]\n")
                continue
            query = SAMPLE_QUERIES[question_choice][1]
            console.print(f"\n[dim]Selected preset [{question_choice}]: {SAMPLE_QUERIES[question_choice][0]}[/dim]")
            console.print(f"[bold cyan]Preset Query:[/bold cyan] {query}\n")
        # 3. Custom free-form query
        else:
            query = user_input
            console.print()

        # Execute Agent with Live Status Spinner
        with console.status("[bold yellow]Executing Autonomous ReAct Loop & Two-Pass Verification...[/bold yellow]", spinner="dots"):
            t0 = time.time()
            try:
                result = agent.run(query)
                elapsed = time.time() - t0
            except Exception as e:
                console.print(f"[bold red]Execution error:[/bold red] {e}\n")
                continue

        # 1. Display Execution Metadata
        meta_table = Table(title="Agent Execution Summary", border_style="green")
        meta_table.add_column("Metric", style="bold")
        meta_table.add_column("Value")

        traj_str = " ➔ ".join(result.get("trajectory", [])) if result.get("trajectory") else "None (Direct Memory Guarded)"
        meta_table.add_row("Tool Trajectory", f"[cyan]{traj_str}[/cyan]")

        is_verified = result.get("is_verified", False)
        status_str = "[green]✓ PASSED (All Claims Grounded)[/green]" if is_verified else "[red]✗ FAILED (Verification Retry Exhausted)[/red]"
        meta_table.add_row("Citation Verification", status_str)
        meta_table.add_row("Self-Correction Retries", str(result.get("retries_used", 0)))
        meta_table.add_row("Execution Latency", f"{elapsed:.2f}s")
        meta_table.add_row("Retrieved Evidence Chunks", str(len(result.get("retrieved_chunks", {}))))

        console.print(meta_table)

        # 2. Display Final Verified Report
        final_report = result.get("final_report", "No report generated.")
        console.print("\n", Panel(
            Markdown(final_report),
            title="[bold green]Final Verified Regulatory Synthesis Report[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))

        Prompt.ask("\n[dim]Press Enter to return to main menu...[/dim]")
        console.print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    run_interactive_demo()
