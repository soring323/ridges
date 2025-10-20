#!/usr/bin/env python3
"""
Simple local benchmark for your agent.
Tests agent on multiple problems without Docker/sandbox complexity.
"""
import os
import sys
import json
import time
import shutil
import tempfile
import argparse
import importlib.util
from pathlib import Path
from typing import List, Dict
from io import StringIO
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluator.problem_suites.problem_suite import ProblemSuite

console = Console()

# Default test problems - edit this list
DEFAULT_TEST_PROBLEMS = [
    #"affine-cipher",
    #"beer-song", 
    "react",
]


def run_agent_locally(agent_file_path: str, problem, suite, timeout: int) -> tuple[str, str, float, str]:
    """Run agent locally without Docker. Returns (patch, logs, elapsed, repo_dir)."""
    workspace_base = os.path.join(os.getcwd(), "result")
    os.makedirs(workspace_base, exist_ok=True)
    workspace_dir = tempfile.mkdtemp(prefix="benchmark_", dir=workspace_base)
    
    # Setup workspace
    repo_dir = os.path.join(workspace_dir, "repo")
    os.makedirs(repo_dir, exist_ok=True)
    
    agent_path = os.path.join(workspace_dir, "agent.py")
    shutil.copy2(agent_file_path, agent_path)
    
    # Copy problem files including tests
    suite.copy_problem_files_to_directory(problem, repo_dir, include_tests=True)
    
    input_data = {"problem_statement": problem.problem_statement}
    with open(os.path.join(workspace_dir, "input.json"), "w") as f:
        json.dump(input_data, f, indent=2)
    
    # Set environment
    os.environ["SANDBOX_PROXY_URL"] = "http://localhost:8000"
    
    original_cwd = os.getcwd()
    os.chdir(workspace_dir)
    sys.path.insert(0, workspace_dir)
    
    try:
        # Run agent
        spec = importlib.util.spec_from_file_location("agent", agent_path)
        agent_module = importlib.util.module_from_spec(spec)
        
        old_stdout, old_stderr = sys.stdout, sys.stderr
        captured = StringIO()
        sys.stdout = sys.stderr = captured
        
        try:
            spec.loader.exec_module(agent_module)
            
            if not hasattr(agent_module, "agent_main"):
                raise Exception("agent_main() not found")
            
            start_time = time.time()
            patch = agent_module.agent_main(input_data)
            elapsed = time.time() - start_time
            
            if not isinstance(patch, str):
                raise Exception(f"agent_main() must return string, got {type(patch)}")
            
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            logs = captured.getvalue()
        
        return patch, logs, elapsed, repo_dir
    finally:
        os.chdir(original_cwd)
        if workspace_dir in sys.path:
            sys.path.remove(workspace_dir)


def run_tests(repo_dir: str, problem_name: str, problem_dir: str) -> Dict:
    """Run tests from original dataset against generated main.py."""
    import subprocess
    import re
    
    # Check main.py exists
    main_py = os.path.join(repo_dir, "main.py")
    if not os.path.exists(main_py):
        return {
            "tests_passed": 0,
            "tests_total": 0,
            "tests_failed": 0,
            "test_output": "No main.py generated",
            "error": "No main.py generated"
        }
    
    # Get tests.py from original dataset
    dataset_test_file = os.path.join("evaluator/datasets/polyglot", problem_name, "tests.py")
    if not os.path.exists(dataset_test_file):
        return {
            "tests_passed": 0,
            "tests_total": 0,
            "tests_failed": 0,
            "test_output": f"No tests.py found at {dataset_test_file}",
            "error": "No tests.py found in dataset"
        }
    
    # Copy tests.py to repo for execution (tests import from main.py)
    repo_test_file = os.path.join(repo_dir, "tests.py")
    shutil.copy2(dataset_test_file, repo_test_file)
    
    try:
        # Run pytest with verbose output
        result = subprocess.run(
            [sys.executable, "-m", "pytest", repo_test_file, "-v", "--tb=short"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout + "\n" + result.stderr
        
        # Parse test results
        passed_match = re.search(r'(\d+) passed', output)
        failed_match = re.search(r'(\d+) failed', output)
        error_match = re.search(r'(\d+) error', output)
        
        tests_passed = int(passed_match.group(1)) if passed_match else 0
        tests_failed = int(failed_match.group(1)) if failed_match else 0
        tests_error = int(error_match.group(1)) if error_match else 0
        tests_total = tests_passed + tests_failed + tests_error
        
        # Save test output
        test_output_file = os.path.join(problem_dir, "test_output.txt")
        with open(test_output_file, "w") as f:
            f.write(output)
        
        console.print(f"  📊 Tests: {tests_passed}/{tests_total} passed", 
                     style="green" if tests_passed == tests_total else "yellow")
        
        return {
            "tests_passed": tests_passed,
            "tests_total": tests_total,
            "tests_failed": tests_failed,
            "test_output": output,
            "error": None if result.returncode == 0 else "Tests failed"
        }
    
    except subprocess.TimeoutExpired:
        return {
            "tests_passed": 0,
            "tests_total": 0,
            "tests_failed": 0,
            "test_output": "Test execution timeout",
            "error": "Test execution timeout"
        }
    except Exception as e:
        return {
            "tests_passed": 0,
            "tests_total": 0,
            "tests_failed": 0,
            "test_output": str(e),
            "error": str(e)
        }


def evaluate_problem(problem_name: str, agent_file: str, timeout: int, results_dir: str) -> Dict:
    """Evaluate agent on one problem."""
    workspace_dir = None
    try:
        # Find problem
        search_result = ProblemSuite.find_problem_in_suites(problem_name)
        if not search_result:
            return {
                "problem": problem_name,
                "score": 0.0,
                "tests_passed": 0,
                "tests_total": 0,
                "patch_generated": False,
                "error": f"Problem '{problem_name}' not found"
            }
        
        suite_name, suite = search_result
        problem = suite.get_problem(problem_name)
        
        # Run agent
        patch, logs, elapsed, repo_dir = run_agent_locally(agent_file, problem, suite, timeout)
        workspace_dir = os.path.dirname(repo_dir)
        
        # Save results
        problem_slug = problem_name.replace("/", "_").replace(" ", "_")
        problem_dir = os.path.join(results_dir, problem_slug)
        os.makedirs(problem_dir, exist_ok=True)
        
        # Save patch
        with open(os.path.join(problem_dir, "patch.diff"), "w") as f:
            f.write(patch if patch else "")
        
        # Save agent logs
        with open(os.path.join(problem_dir, "agent_logs.txt"), "w") as f:
            f.write(logs)
        
        # Copy generated main.py from workspace
        main_src = os.path.join(repo_dir, "main.py")
        if os.path.exists(main_src):
            main_dst = os.path.join(problem_dir, "main.py")
            shutil.copy2(main_src, main_dst)
        
        # Copy reference solution from original problem directory
        problem_dataset_dir = os.path.join("evaluator/datasets/polyglot", problem.name)
        reference_solution = os.path.join(problem_dataset_dir, "solution.py")
        if os.path.exists(reference_solution):
            reference_dst = os.path.join(problem_dir, "reference_solution.py")
            shutil.copy2(reference_solution, reference_dst)
        
        # Copy tests.py from dataset
        tests_src = os.path.join(problem_dataset_dir, "tests.py")
        if os.path.exists(tests_src):
            tests_dst = os.path.join(problem_dir, "tests.py")
            shutil.copy2(tests_src, tests_dst)
        
        # Run tests - this will also save generated code as solution.py
        test_results = {"tests_passed": 0, "tests_total": 0, "tests_failed": 0, "error": "No solution generated"}
        if os.path.exists(main_src):
            test_results = run_tests(repo_dir, problem.name, problem_dir)
        else:
            console.print(f"  ❌ No main.py generated", style="red")
        
        # Calculate score
        score = test_results["tests_passed"] / test_results["tests_total"] if test_results["tests_total"] > 0 else 0.0
        
        summary = {
            "problem": problem_name,
            "score": score,
            "tests_passed": test_results["tests_passed"],
            "tests_total": test_results["tests_total"],
            "tests_failed": test_results["tests_failed"],
            "patch_generated": bool(patch),
            "patch_length": len(patch) if patch else 0,
            "elapsed_time": elapsed,
            "exceeded_timeout": elapsed > timeout,
            "error": test_results.get("error")
        }
        
        with open(os.path.join(problem_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        # Display test results
        if test_results["tests_total"] > 0:
            console.print(f"  Tests: {test_results['tests_passed']}/{test_results['tests_total']} passed", style="cyan")
        
        return summary
        
    except Exception as e:
        import traceback
        return {
            "problem": problem_name,
            "score": 0.0,
            "tests_passed": 0,
            "tests_total": 0,
            "patch_generated": False,
            "error": f"{str(e)}\n{traceback.format_exc()}"
        }
    finally:
        # Cleanup workspace
        if workspace_dir and os.path.exists(workspace_dir):
            shutil.rmtree(workspace_dir, ignore_errors=True)


def benchmark_agent(agent_file: str, problems: List[str], timeout: int):
    """Run benchmark on all problems."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("result", f"benchmark_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    console.print("📡 Requires inference gateway at http://localhost:8000", style="cyan")
    console.print("💡 Start it: python -m inference_gateway.main\n", style="dim")
    
    console.print(Panel(
        f"[bold cyan]🏆 Agent Benchmark (Local Mode)[/bold cyan]\n\n"
        f"[yellow]Agent:[/yellow] {agent_file}\n"
        f"[yellow]Problems:[/yellow] {len(problems)}\n"
        f"[yellow]Timeout:[/yellow] {timeout}s per problem",
        title="🎯 Configuration",
        border_style="cyan"
    ))
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Running...", total=len(problems))
        
        for i, problem in enumerate(problems):
            console.print(f"\n[bold]Problem {i+1}/{len(problems)}:[/bold] {problem}")
            
            result = evaluate_problem(problem, agent_file, timeout, results_dir)
            results.append(result)
            
            if result["error"] and not result['patch_generated']:
                console.print(f"  [red]❌ {result['error'][:50]}[/red]")
            else:
                tests_total = result.get('tests_total', 0)
                tests_passed = result.get('tests_passed', 0)
                if tests_total > 0:
                    emoji = "🎉" if tests_passed == tests_total else "⚠️" if tests_passed > 0 else "❌"
                    console.print(f"  {emoji} Tests: {tests_passed}/{tests_total} passed ({result['score']*100:.1f}%)")
                else:
                    emoji = "🎉" if result['patch_generated'] else "❌"
                    console.print(f"  {emoji} Patch: {'Generated' if result['patch_generated'] else 'Failed'}")
            
            progress.update(task, advance=1)
    
    # Calculate scores
    valid_results = [r for r in results if r["error"] is None]
    final_score = sum(r['score'] for r in valid_results) / len(valid_results) if valid_results else 0.0
    
    # Display table
    console.print("\n" + "="*80)
    table = Table(title="📊 Results", show_header=True, header_style="bold magenta")
    table.add_column("Problem", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Tests", justify="center")
    table.add_column("Status")
    
    for result in results:
        score_str = f"{result['score']*100:.1f}%"
        tests_str = f"{result.get('tests_passed', 0)}/{result.get('tests_total', 0)}"
        
        if result['score'] == 1.0:
            status = "✅ All passed"
        elif result['score'] > 0:
            status = "⚠️ Partial"
        elif result.get('error'):
            status = f"❌ {result['error'][:20]}"
        else:
            status = "❌ Failed"
        
        table.add_row(result['problem'], score_str, tests_str, status)
    
    console.print(table)
    
    # Summary
    console.print("\n" + "="*80)
    total_tests = sum(r.get('tests_total', 0) for r in results)
    total_passed = sum(r.get('tests_passed', 0) for r in results)
    
    console.print(Panel(
        f"[bold cyan]Final Score:[/bold cyan] [bold yellow]{final_score*100:.1f}%[/bold yellow]\n\n"
        f"Total Tests: {total_passed}/{total_tests} passed\n"
        f"Problems: {sum(1 for r in results if r['score'] == 1.0)}/{len(results)} fully solved\n"
        f"Patches generated: {sum(1 for r in results if r['patch_generated'])}/{len(results)}",
        title="🏆 Summary",
        border_style="green" if final_score > 0.8 else "yellow"
    ))
    
    # Save summary
    summary_data = {
        "timestamp": timestamp,
        "agent_file": agent_file,
        "final_score": final_score,
        "problems_tested": len(problems),
        "patches_generated": sum(1 for r in results if r['patch_generated']),
        "total_tests": total_tests,
        "total_passed": total_passed,
        "results": results
    }
    
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)
    
    with open("result/latest_benchmark.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    
    # Create README
    readme_content = f"""# Benchmark Results - {timestamp}

## Summary
- **Agent**: {agent_file}
- **Final Score**: {final_score*100:.1f}%
- **Total Tests**: {total_passed}/{total_tests} passed
- **Problems**: {len(problems)} tested, {sum(1 for r in results if r['score'] == 1.0)} fully solved

## Results by Problem

"""
    for result in results:
        emoji = "🎉" if result['score'] == 1.0 else "⚠️" if result['score'] > 0 else "❌"
        readme_content += f"### {emoji} {result['problem']}\n"
        readme_content += f"- **Score**: {result['score']*100:.1f}%\n"
        readme_content += f"- **Tests**: {result.get('tests_passed', 0)}/{result.get('tests_total', 0)} passed\n"
        readme_content += f"- **Elapsed Time**: {result.get('elapsed_time', 0):.2f}s\n"
        if result.get('error'):
            readme_content += f"- **Error**: {result['error']}\n"
        readme_content += f"- **Files**: See `{result['problem'].replace('/', '_').replace(' ', '_')}/`\n\n"
    
    readme_content += """## Directory Structure

Each problem has its own subdirectory with:
- `main.py` - **🎯 Generated solution** by your agent (this is what was tested!)
- `reference_solution.py` - Correct reference solution from dataset
- `tests.py` - Test cases from dataset (executed against main.py)
- `test_output.txt` - **📊 Complete test execution logs** with all pass/fail details
- `agent_logs.txt` - Agent's internal execution logs
- `patch.diff` - Git diff format patch
- `summary.json` - Machine-readable summary

## Files in this Directory
- `summary.json` - Overall benchmark summary
- `README.md` - This file

## How to Review Results

1. Check `summary.json` for overall score
2. For each problem, review:
   - `main.py` to see what your agent generated
   - `reference_solution.py` to see the correct implementation
   - `test_output.txt` to see which tests passed/failed and error messages
   - `agent_logs.txt` to debug agent behavior
   
## Re-running Tests Manually

You can re-run tests for any problem:
```bash
cd result/benchmark_TIMESTAMP/PROBLEM_NAME/
python -m pytest tests.py -v
```

## Comparing Your Solution vs Reference

To see how your agent's solution differs from the reference:
```bash
cd result/benchmark_TIMESTAMP/PROBLEM_NAME/
diff main.py reference_solution.py
```

## Test Output Format

The `test_output.txt` files contain pytest output showing:
- ✅ Passed tests with names
- ❌ Failed tests with error messages
- Test execution time
- Coverage information (if available)
"""
    
    with open(os.path.join(results_dir, "README.md"), "w") as f:
        f.write(readme_content)
    
    console.print(f"\n💾 Results saved to: [cyan]{results_dir}/[/cyan]")
    console.print(f"   - [cyan]README.md[/cyan] - Human-readable summary")
    console.print(f"   - [cyan]summary.json[/cyan] - Machine-readable data")
    console.print(f"   - [cyan]<problem>/main.py[/cyan] - 🎯 Generated solution (tested)")
    console.print(f"   - [cyan]<problem>/reference_solution.py[/cyan] - Reference solution (correct)")
    console.print(f"   - [cyan]<problem>/tests.py[/cyan] - Test cases from dataset")
    console.print(f"   - [cyan]<problem>/test_output.txt[/cyan] - 📊 Complete test logs")
    console.print(f"\n   Quick access: [cyan]result/latest_benchmark.json[/cyan]")


def main():
    parser = argparse.ArgumentParser(description="Simple local agent benchmark")
    parser.add_argument("--agent-file", default="agent.py", help="Agent file (default: agent.py)")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout per problem (default: 30s)")
    
    args = parser.parse_args()
    
    if not Path(args.agent_file).exists():
        console.print(f"[bold red]❌ Agent file not found: {args.agent_file}[/bold red]")
        sys.exit(1)
    
    if not DEFAULT_TEST_PROBLEMS:
        console.print("[bold red]❌ No problems in DEFAULT_TEST_PROBLEMS[/bold red]")
        sys.exit(1)
    
    benchmark_agent(args.agent_file, DEFAULT_TEST_PROBLEMS, args.timeout)


if __name__ == "__main__":
    main()
