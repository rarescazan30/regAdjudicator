"""
Evaluation Runner & Benchmarking Suite
Executes evals/evalset.json test cases against RegulatoryAgent,
calculates trajectory accuracy, classification precision, and verification rates,
and prints a rich summary table of performance metrics.
"""

import json
import time
from typing import Dict, Any, List
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.config import EVALS_DIR
from src.agent import RegulatoryAgent


console = Console()


def load_evalset() -> List[Dict[str, Any]]:
    """Loads benchmark test cases from evals/evalset.json."""
    evalset_path = EVALS_DIR / "evalset.json"
    if not evalset_path.exists():
        raise FileNotFoundError(f"Evalset file not found at {evalset_path}")
    with open(evalset_path, mode="r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_case(agent: RegulatoryAgent, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a single test case through the agent and evaluates all 4 metrics."""
    case_id = test_case["id"]
    query = test_case["query"]
    expected_postures = test_case.get("expected_postures", [test_case.get("expected_posture", "")])
    required_tools = test_case.get("required_tools", [])
    forbidden_tools = test_case.get("forbidden_tools", [])
    expected_keywords = test_case.get("expected_keywords", [])

    # Execute agent and measure execution latency
    t0 = time.time()
    result = agent.run(query)
    latency = time.time() - t0

    trajectory = result.get("trajectory", [])
    final_report = result.get("final_report", "")
    is_verified = result.get("is_verified", False)
    retries_used = result.get("retries_used", 0)

    # Metric 1: Trajectory Adherence
    required_passed = all(tool in trajectory for tool in required_tools)
    forbidden_passed = not any(tool in trajectory for tool in forbidden_tools)
    traj_pass = required_passed and forbidden_passed

    # Metric 2: Posture Classification Accuracy
    class_pass = any(posture.lower() in final_report.lower() for posture in expected_postures)

    # Metric 3: Factual Keyword Coverage (stem / substring matching)
    kw_passed = all(kw.lower() in final_report.lower() for kw in expected_keywords)

    # Metric 4: Two-Pass Citation Verification
    verify_pass = is_verified

    # Overall Case Pass: All metrics pass
    overall_pass = traj_pass and class_pass and verify_pass and kw_passed

    return {
        "id": case_id,
        "category": test_case.get("category", "general"),
        "latency": latency,
        "overall_pass": overall_pass,
        "traj_pass": traj_pass,
        "class_pass": class_pass,
        "verify_pass": verify_pass,
        "kw_pass": kw_passed,
        "retries_used": retries_used,
        "trajectory": " -> ".join(trajectory) if trajectory else "None",
    }


def run_benchmark():
    """Runs the full evaluation benchmark suite and prints the metrics report."""
    test_cases = load_evalset()
    agent = RegulatoryAgent()

    console.print(
        Panel.fit(
            f"[bold cyan]Starting regAdjudicator Benchmark Evaluation[/bold cyan]\n"
            f"Test Suite: [yellow]{len(test_cases)} cases[/yellow] | Model: [green]{agent.model_name}[/green]",
            border_style="cyan",
        )
    )

    results = []
    for i, case in enumerate(test_cases, 1):
        console.print(f"Running [{i}/{len(test_cases)}]: [bold]{case['id']}[/bold]...", end="")
        res = evaluate_case(agent, case)
        status_icon = "[green]✓ PASS[/green]" if res["overall_pass"] else "[red]✗ FAIL[/red]"
        console.print(f" {status_icon} ({res['latency']:.1f}s)")
        results.append(res)
        
        # Rate-limiting spacer to stay below 15 RPM
        if i < len(test_cases):
            time.sleep(5)

    # Calculate Aggregate Metrics
    total = len(results)
    passed_total = sum(1 for r in results if r["overall_pass"])
    traj_score = sum(1 for r in results if r["traj_pass"]) / total * 100
    class_score = sum(1 for r in results if r["class_pass"]) / total * 100
    verify_score = sum(1 for r in results if r["verify_pass"]) / total * 100
    kw_score = sum(1 for r in results if r["kw_pass"]) / total * 100
    avg_latency = sum(r["latency"] for r in results) / total

    # Print Results Table
    table = Table(title="\nDetailed Benchmark Results", border_style="cyan")
    table.add_column("Case ID", style="bold")
    table.add_column("Category")
    table.add_column("Trajectory", justify="center")
    table.add_column("Classification", justify="center")
    table.add_column("Verified", justify="center")
    table.add_column("Keywords", justify="center")
    table.add_column("Latency", justify="right")
    table.add_column("Outcome", justify="center")

    for r in results:
        table.add_row(
            r["id"],
            r["category"],
            "✓" if r["traj_pass"] else "✗",
            "✓" if r["class_pass"] else "✗",
            "✓" if r["verify_pass"] else "✗",
            "✓" if r["kw_pass"] else "✗",
            f"{r['latency']:.1f}s",
            "[green]PASS[/green]" if r["overall_pass"] else "[red]FAIL[/red]",
        )

    console.print(table)

    # Print Summary Metrics Panel
    console.print(
        Panel.fit(
            f"[bold]Aggregate Performance Metrics:[/bold]\n\n"
            f"• [bold]Overall Benchmark Pass Rate:[/bold] [green]{passed_total}/{total} ({passed_total/total*100:.1f}%)[/green]\n"
            f"• [bold]Trajectory Accuracy:[/bold] {traj_score:.1f}%\n"
            f"• [bold]Posture Classification Accuracy:[/bold] {class_score:.1f}%\n"
            f"• [bold]Citation Grounding / Verification Rate:[/bold] {verify_score:.1f}%\n"
            f"• [bold]Factual Keyword Recall:[/bold] {kw_score:.1f}%\n"
            f"• [bold]Average Latency per Query:[/bold] {avg_latency:.2f}s",
            title="Benchmark Summary",
            border_style="green" if passed_total == total else "yellow",
        )
    )


if __name__ == "__main__":
    run_benchmark()
