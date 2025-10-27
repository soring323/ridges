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
import subprocess
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
from evaluator.problem_suites.polyglot.polyglot_suite import PolyglotSuite
from evaluator.problem_suites.swebench_verified.swebench_verified_suite import SWEBenchVerifiedSuite

console = Console()

# Default test problems - edit this list
DEFAULT_TEST_PROBLEMS = [
    # "affine-cipher",
    "beer-song", 
    # "react",
    # "react",
    # "react",
    # "react",
    # "react",
    # "react",
    # "react",
    # "react",
    # "robot-name",
    # "rest-api",      # Test: JSON string format detection
    # "book-store",    # Test: Unit conversion (cents vs dollars)
    # "scale-generator",
    # "grep",
    # "pov",
    
    # SWE Benchmark problems
    # "astropy__astropy-14369",
    # "django__django-15629",
    # "django__django-15957",
    # "django__django-10554",
    # "django__django-12325",
    # "django__django-11400",
    # "django__django-16263",
    # "django__django-12708",
]

AGENT_FILE_NAME = "test_driven_agent_oop.py"


def get_problem_suites() -> List[ProblemSuite]:
    """Initialize and return all available problem suites."""
    datasets_path = Path(__file__).parent / "evaluator" / "datasets"
    
    polyglot_suite = PolyglotSuite(str(datasets_path / "polyglot"))
    swebench_suite = SWEBenchVerifiedSuite(str(datasets_path / "swebench_verified"))
    
    return [polyglot_suite, swebench_suite]


def run_agent_locally(agent_file_path: str, problem, suite, timeout: int) -> tuple[str, str, float, str]:
    """Run agent locally without Docker. Returns (patch, logs, elapsed, repo_dir)."""
    workspace_base = os.path.join(os.getcwd(), "result")
    os.makedirs(workspace_base, exist_ok=True)
    workspace_dir = tempfile.mkdtemp(prefix="benchmark_", dir=workspace_base)
    
    # Setup workspace
    repo_dir = os.path.join(workspace_dir, "repo")
    os.makedirs(repo_dir, exist_ok=True)

    
    
    agent_path = os.path.join(workspace_dir, AGENT_FILE_NAME)
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


def run_tests_polyglot(repo_dir: str, problem_name: str, problem_dir: str) -> Dict:
    """Run tests from original dataset against generated main.py (Polyglot problems only)."""
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


def setup_test_venv_for_repo(repo_name: str, repo_path: str, force_reinstall: bool = False) -> str:
    """Create or reuse a cached virtual environment for a repository."""
    venv_dir = os.path.join(os.getcwd(), ".testvenv", repo_name)
    venv_ready_marker = os.path.join(venv_dir, ".venv_ready")
    
    # Check if venv exists and is marked as ready
    if os.path.exists(venv_dir) and os.path.exists(venv_ready_marker) and not force_reinstall:
        console.print(f"  ♻️  Reusing cached venv for {repo_name}...", style="green")
        return venv_dir
    
    # If force reinstall, remove old venv
    if force_reinstall and os.path.exists(venv_dir):
        console.print(f"  🔄 Force reinstalling venv for {repo_name}...", style="yellow")
        shutil.rmtree(venv_dir, ignore_errors=True)
    
    console.print(f"  🔨 Creating new venv for {repo_name}...", style="cyan")
    os.makedirs(venv_dir, exist_ok=True)
    
    # Create virtual environment
    # For Python 3.12+, we need to use Python 3.11 for old packages (distutils compatibility)
    python_cmd = sys.executable
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    # If we're on Python 3.12+, try to use a compatible Python from a shared location
    if sys.version_info >= (3, 12):
        # Check for pre-installed Python 3.11 in a shared location
        shared_python_dir = os.path.join(os.getcwd(), ".python_versions")
        python311_path = os.path.join(shared_python_dir, "python3.11", "bin", "python3")
        
        if not os.path.exists(python311_path):
            console.print(f"  📦 Python 3.12+ detected, installing Python 3.11 for compatibility...", style="yellow")
            os.makedirs(shared_python_dir, exist_ok=True)
            
            # Use deadsnakes PPA or download portable Python
            install_result = subprocess.run(
                ["bash", "-c", """
                    # Try to install python3.11 via apt if available
                    if command -v apt-get &> /dev/null; then
                        sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
                        sudo apt-get update -qq 2>/dev/null || true
                        sudo apt-get install -y python3.11 python3.11-venv python3.11-dev 2>/dev/null || true
                    fi
                    
                    # Check if python3.11 is now available
                    if command -v python3.11 &> /dev/null; then
                        echo "python3.11"
                    else
                        echo "failed"
                    fi
                """],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if "python3.11" in install_result.stdout:
                python_cmd = "python3.11"
                console.print(f"  ✅ Installed Python 3.11 system-wide", style="green")
            else:
                console.print(f"  ⚠️  Could not install Python 3.11, using Python {current_version} (may fail)", style="yellow")
        else:
            python_cmd = python311_path
            console.print(f"  ✅ Using cached Python 3.11", style="green")
    
    # Try to find any available Python 3.11/3.10/3.9
    if python_cmd == sys.executable and sys.version_info >= (3, 12):
        for py_version in ["python3.11", "python3.10", "python3.9"]:
            try:
                result = subprocess.run([py_version, "--version"], capture_output=True, timeout=5)
                if result.returncode == 0:
                    python_cmd = py_version
                    console.print(f"  📍 Found {result.stdout.decode().strip()}", style="dim")
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
    
    version_check = subprocess.run([python_cmd, "--version"], capture_output=True, text=True)
    console.print(f"  📍 Using {version_check.stdout.strip()} for venv", style="dim")
    
    subprocess.run([python_cmd, "-m", "venv", venv_dir], check=True)
    
    # Get pip path
    pip_path = os.path.join(venv_dir, "bin", "pip")
    
    # Upgrade pip
    console.print(f"  📦 Upgrading pip...", style="cyan")
    subprocess.run([pip_path, "install", "--upgrade", "pip"], 
                   capture_output=True, check=True)
    
    # Install repository dependencies from the checked-out commit
    console.print(f"  📦 Installing {repo_name} dependencies...", style="cyan")
    try:
        # CRITICAL: Install build dependencies FIRST before trying to install the package
        # For Python 3.12+, we need to use a modern setuptools OR install distutils separately
        python_path = os.path.join(venv_dir, "bin", "python")
        
        # Check Python version
        py_version_check = subprocess.run(
            [python_path, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True, text=True, timeout=10
        )
        py_version = py_version_check.stdout.strip() if py_version_check.returncode == 0 else "3.12"
        
        console.print(f"  📦 Installing build dependencies for Python {py_version}...", style="dim")
        
        # For Python 3.12+, install setuptools with distutils compatibility
        if py_version >= "3.12":
            # Install setuptools that works without stdlib distutils
            subprocess.run(
                [pip_path, "install", "setuptools>=65.5.0", "wheel", "setuptools_scm", "extension-helpers", "cython", "numpy<2.0"],
                capture_output=True, text=True, timeout=120
            )
            # Install setuptools-distutils compatibility shim
            console.print(f"  📦 Installing distutils compatibility for Python 3.12+...", style="dim")
            subprocess.run(
                [pip_path, "install", "setuptools"],  # Modern setuptools includes _distutils_hack
                capture_output=True, text=True, timeout=60
            )
        else:
            # For Python <3.12, use older setuptools
            subprocess.run(
                [pip_path, "install", "setuptools<58", "wheel", "setuptools_scm<7", "extension-helpers", "cython", "numpy<2.0"],
                capture_output=True, text=True, timeout=120
            )
        
        # Install the package (NOT editable) - this properly builds C extensions
        # Don't use -e because old codebases have build issues with editable mode
        installed = False
        # Prefer using our preinstalled build deps by disabling build isolation
        for install_cmd in [
            [pip_path, "install", "--no-build-isolation", ".[test]"],
            [pip_path, "install", "--no-build-isolation", ".[dev]"],
            [pip_path, "install", "--no-build-isolation", ".[tests]"],
            [pip_path, "install", "--no-build-isolation", "."],
        ]:
            env = os.environ.copy()
            # Ensure distutils from stdlib is used by old setuptools
            env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
            # Ensure pip uses our preinstalled build deps
            env["PIP_NO_BUILD_ISOLATION"] = "1"
            result = subprocess.run(
                install_cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=300,
                env=env
            )
            if result.returncode == 0:
                console.print(f"  ✅ Installed with {' '.join(install_cmd[2:])}", style="dim")
                installed = True
                break
        
        # If install failed, show the error
        if not installed:
            console.print(f"  ⚠️  All install attempts failed. Last error:", style="yellow")
            # Print more of the error to aid debugging
            console.print(f"  {result.stderr[-1200:] if result and result.stderr else '(no stderr)'}", style="dim")

            # Fallback: try legacy setup.py installers (common for very old repos)
            python_path = os.path.join(venv_dir, "bin", "python")
            legacy_env = os.environ.copy()
            legacy_env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
            legacy_env["PIP_NO_BUILD_ISOLATION"] = "1"

            console.print("  🔄 Trying legacy setup.py install (no PEP517)...", style="cyan")
            legacy_install = subprocess.run(
                [python_path, "setup.py", "install"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=420,
                env=legacy_env
            )
            if legacy_install.returncode == 0:
                console.print("  ✅ setup.py install succeeded", style="green")
                installed = True
            else:
                # As a last resort, try building extensions in-place then install
                console.print("  🔄 Trying setup.py build_ext --inplace ...", style="cyan")
                build_ext = subprocess.run(
                    [python_path, "setup.py", "build_ext", "--inplace"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=420,
                    env=legacy_env
                )
                if build_ext.returncode == 0:
                    console.print("  ✅ setup.py build_ext succeeded; attempting pip install . (no isolation)", style="green")
                    pip_install_after_build = subprocess.run(
                        [pip_path, "install", "--no-build-isolation", "."],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=300,
                        env=legacy_env
                    )
                    if pip_install_after_build.returncode == 0:
                        installed = True
                    else:
                        console.print(f"  ❌ pip install after build failed: {pip_install_after_build.stderr[-800:]}" , style="red")
                else:
                    console.print(f"  ❌ setup.py build_ext failed: {build_ext.stderr[-800:]}" , style="red")
                    
    except Exception as e:
        console.print(f"  ⚠️  Warning during dependency installation: {e}", style="yellow")
    
    # Always install pytest and common build dependencies
    console.print(f"  📦 Installing pytest and build tools...", style="cyan")
    subprocess.run([pip_path, "install", "pytest", "pytest-xdist", "setuptools_scm", "wheel"],
                   capture_output=True)
    
    # Mark venv as successfully set up
    with open(venv_ready_marker, "w") as f:
        f.write(f"Created: {time.time()}\n")
    
    console.print(f"  ✅ Venv ready for {repo_name}", style="green")
    return venv_dir


def convert_swebench_test_to_pytest(test_name: str) -> str:
    """Convert SWEBench test format to pytest format.
    
    SWEBench format: 'test_name (module.file.Class)'
    Pytest format: 'module/file.py::Class::test_name'
    
    If already in pytest format (contains ::), return as-is.
    """
    import re
    
    # Already in pytest format
    if "::" in test_name:
        return test_name
    
    # Match SWEBench format: test_name (module.file.Class)
    # Use non-greedy match and capture everything before the last dot as module path
    match = re.match(r'^([\w_]+)\s*\((.+)\.(\w+)\)$', test_name)
    if match:
        test_func = match.group(1)
        module_path = match.group(2)  # Everything before last dot
        class_name = match.group(3)   # Last component after dot
        
        # Convert module.file to module/file.py
        file_path = module_path.replace('.', '/') + '.py'
        
        # Build pytest format: file.py::Class::test_func
        return f"{file_path}::{class_name}::{test_func}"
    
    # If no match, return as-is (might already be in correct format)
    return test_name


def run_tests_swebench_local(problem, patch: str, problem_dir: str) -> Dict:
    """Run tests for SWEBench problems locally without Docker."""
    import re
    from utils.git import clone_local_repo_at_commit
    
    try:
        console.print(f"  📁 Testing SWEBench problem locally (no Docker)...", style="cyan")
        
        # Get repo info from problem userdata
        swebench_data = problem.userdata
        repo_name = swebench_data.get("repo", "").replace('/', '_')
        base_commit = swebench_data.get("base_commit")
        repo_path = os.path.join("evaluator/datasets/swebench_verified/repos", repo_name)
        
        if not os.path.exists(repo_path):
            raise Exception(f"Repository not found: {repo_path}")
        
        # Create temp workspace
        with tempfile.TemporaryDirectory(prefix="swebench_test_") as temp_dir:
            # Clone repo at the exact base_commit (this handles checkout automatically)
            temp_repo = os.path.join(temp_dir, "repo")
            console.print(f"  📌 Cloning repo at commit: {base_commit[:8]}...", style="cyan")
            clone_local_repo_at_commit(repo_path, base_commit, temp_repo)
            
            # Setup venv for this specific commit (cache by repo + commit hash)
            venv_cache_name = f"{repo_name}_{base_commit[:8]}" if base_commit else repo_name
            venv_dir = setup_test_venv_for_repo(venv_cache_name, temp_repo, force_reinstall=False)
            python_path = os.path.join(venv_dir, "bin", "python")
            
            # Apply patch
            if patch and patch.strip():
                patch_file = os.path.join(temp_dir, "patch.diff")
                with open(patch_file, "w") as f:
                    f.write(patch)
                
                console.print(f"  🔧 Applying patch...", style="cyan")
                result = subprocess.run(
                    ["git", "apply", "--whitespace=fix", patch_file],
                    cwd=temp_repo,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    console.print(f"  ⚠️ Patch application warning: {result.stderr}", style="yellow")
                
                # Check if patch modifies any C/Cython files that require rebuild
                needs_rebuild = False
                if patch and patch.strip():
                    c_extensions = ['.c', '.cpp', '.pyx', '.pxd', '.h', '.hpp']
                    for line in patch.split('\n'):
                        if line.startswith('diff --git') or line.startswith('+++'):
                            if any(ext in line for ext in c_extensions):
                                needs_rebuild = True
                                break
                
                # Only rebuild if C extensions were modified
                if not needs_rebuild:
                    console.print(f"  ✅ Patch only modifies Python files, skipping rebuild", style="green")
                else:
                    # Rebuild package after applying patch (needed for C extensions like astropy)
                    console.print(f"  🔨 Rebuilding package with patch applied...", style="cyan")
                    pip_path = os.path.join(venv_dir, "bin", "pip")
                    python_path = os.path.join(venv_dir, "bin", "python")
                    
                    # First, ensure build dependencies are installed
                    console.print(f"  📦 Ensuring build dependencies...", style="dim")
                    subprocess.run(
                        [pip_path, "install", "-q", "setuptools", "wheel", "extension-helpers", "setuptools_scm", "cython"],
                        capture_output=True,
                        timeout=60
                    )
                    
                    # Try multiple rebuild strategies
                    rebuild_success = False
                    
                    # Strategy 1: Full reinstall with dependencies (most reliable for C extensions)
                    console.print(f"  🔄 Strategy 1: Full reinstall...", style="dim")
                    rebuild_result = subprocess.run(
                        [pip_path, "install", "--force-reinstall", "--no-build-isolation", "."],
                        cwd=temp_repo,
                        capture_output=True,
                        text=True,
                        timeout=240
                    )
                    
                    if rebuild_result.returncode == 0:
                        rebuild_success = True
                        console.print(f"  ✅ Rebuilt successfully", style="green")
                    else:
                        # Strategy 2: Try setup.py develop (editable install with build)
                        setup_py = os.path.join(temp_repo, "setup.py")
                        if os.path.exists(setup_py):
                            console.print(f"  🔄 Strategy 2: setup.py develop...", style="dim")
                            rebuild_result = subprocess.run(
                                [python_path, "setup.py", "develop"],
                                cwd=temp_repo,
                                capture_output=True,
                                text=True,
                                timeout=240
                            )
                            if rebuild_result.returncode == 0:
                                rebuild_success = True
                                console.print(f"  ✅ Built successfully", style="green")
                    
                    if not rebuild_success:
                        console.print(f"  ⚠️ Rebuild failed, tests may not run properly", style="yellow")
                        console.print(f"  Error: {rebuild_result.stderr[:300]}", style="dim")
            
            # Parse test cases from problem userdata
            fail_to_pass = json.loads(swebench_data.get("FAIL_TO_PASS", "[]"))
            pass_to_pass = json.loads(swebench_data.get("PASS_TO_PASS", "[]"))
            all_tests = fail_to_pass + pass_to_pass
            
            if not all_tests:
                return {
                    "tests_passed": 0,
                    "tests_total": 0,
                    "tests_failed": 0,
                    "test_output": "No test cases found in problem definition",
                    "error": "No tests"
                }
            
            # Check if Django's runtests.py exists (Django uses custom test runner)
            django_runtests = os.path.join(temp_repo, "tests", "runtests.py")
            use_django_runner = os.path.exists(django_runtests)
            
            console.print(f"  🧪 Running {len(all_tests)} tests...", style="cyan")
            
            if use_django_runner:
                # Django format: "test_name (module.file.Class)" → "module.file.Class.test_name"
                console.print(f"  🐍 Using Django's test runner (runtests.py)", style="cyan")
                console.print(f"problem_statement: {problem.userdata['problem_statement']}")
                # django_tests = []
                # for test in all_tests:
                    # Format: module.file.Class.test_name
                    # match = re.match(r'^([\w_]+)\s*\((.+)\)$', test)
                    # if match:
                    #     test_func = match.group(1)
                    #     module_class_path = match.group(2)
                    #     # Format: module.file.Class.test_name
                    #     django_tests.append(f"{module_class_path}.{test_func}")
                    # else:
                    #     # Skip test descriptions that don't match the expected format
                    #     # These are likely test docstrings, not actual test paths
                    #     console.print(f"  ⚠️  Skipping invalid test label: {test[:50]}...", style="dim")
                    #     continue
                
                # Run tests serially to avoid multiprocessing pickle errors
                test_args = [python_path, django_runtests, "--verbosity=2", "--parallel=1"] + all_tests
                result = subprocess.run(
                    test_args,
                    cwd=temp_repo,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                output = result.stdout + "\n" + result.stderr
            else:
                # Convert SWEBench test format to pytest format
                all_tests = [convert_swebench_test_to_pytest(t) for t in all_tests]
                
                # Check if repo has a tests/ directory and prepend it if needed
                tests_dir = os.path.join(temp_repo, "tests")
                if os.path.exists(tests_dir) and os.path.isdir(tests_dir):
                    all_tests = [
                        f"tests/{t}" if not t.startswith("tests/") else t
                        for t in all_tests
                    ]
                
                # Try running specific tests first
                # CRITICAL: Use --pyargs to import from installed package, not source directory
                # This tells pytest to import the package, not use file paths
                test_run_dir = os.path.join(temp_dir, "test_runner")
                os.makedirs(test_run_dir, exist_ok=True)
                
                # Convert file paths to module paths for --pyargs
                # e.g., "astropy/units/tests/test_format.py::test_func" -> "astropy.units.tests.test_format::test_func"
                pyargs_tests = []
                for test in all_tests:
                    # Remove .py extension and convert slashes to dots
                    test = test.replace("/", ".").replace(".py", "")
                    pyargs_tests.append(test)
                
                test_args = [python_path, "-m", "pytest", "-xvs", "--pyargs"] + pyargs_tests
                
                # Set environment to prevent importing from source
                test_env = os.environ.copy()
                test_env["PYTHONDONTWRITEBYTECODE"] = "1"
                test_env.pop("PYTHONPATH", None)
                
                result = subprocess.run(
                    test_args,
                    cwd=test_run_dir,  # Run from empty dir
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=test_env
                )
                
                output = result.stdout + "\n" + result.stderr
                
                # If pytest can't find the tests (common with SWEBench), try running specific test functions
                if "ERROR: not found:" in output and len(all_tests) > 0:
                    console.print(f"  ⚠️  Specific parameterized test IDs not found (version mismatch)", style="yellow")
                    
                    # Extract unique test function names (for -k keyword matching)
                    # e.g., "file.py::test_func[param]" -> "test_func"
                    test_function_names = set()
                    test_file = None
                    for test in all_tests:
                        if "::" in test:
                            parts = test.split("::")
                            # Get the file path
                            if not test_file:
                                test_file = parts[0]
                            # Get the test function name (last part, without parameters)
                            last_part = parts[-1].split("[")[0]
                            test_function_names.add(last_part)
                    
                    if test_function_names and test_file:
                        # Use -k for keyword matching of test names with --pyargs
                        keyword_expr = " or ".join(sorted(test_function_names))
                        console.print(f"  🔄 Running {len(test_function_names)} test functions with -k (keyword match)", style="cyan")
                        
                        # Convert file path to module path
                        test_module = test_file.replace("/", ".").replace(".py", "")
                        
                        test_args = [python_path, "-m", "pytest", "-xvs", "--pyargs", test_module, "-k", keyword_expr]
                        result = subprocess.run(
                            test_args,
                            cwd=test_run_dir,  # Run from empty dir, not repo
                            capture_output=True,
                            text=True,
                            timeout=300,
                            env=test_env
                        )
                        output = result.stdout + "\n" + result.stderr
            
            # Check for import/module errors that indicate missing dependencies
            has_import_error = (
                "ModuleNotFoundError" in output or 
                "ImportError" in output or
                "No module named" in output
            )
            
            # If we hit dependency issues, install the missing module and retry once
            if has_import_error:
                import re as regex_module
                missing_match = regex_module.search(r"No module named ['\"]([^'\"]+)['\"]", output)
                if missing_match:
                    missing_module = missing_match.group(1)
                    # Validate that this is actually a valid module name (not an error message)
                    # Module names should only contain alphanumeric, dots, and underscores
                    if not regex_module.match(r'^[a-zA-Z0-9_\.]+$', missing_module):
                        console.print(f"  ⚠️  Ignoring invalid module name: {missing_module[:50]}", style="dim")
                    else:
                        console.print(f"  ⚠️  Missing test dependency: {missing_module}", style="yellow")
                        
                        # Common package name mappings (import name -> pip package name)
                        package_mappings = {
                            'erfa': 'pyerfa',
                            'yaml': 'pyyaml',
                            'PIL': 'pillow',
                            'cv2': 'opencv-python',
                            'distutils': None,  # distutils is part of setuptools, already installed
                        }
                        
                        # Get the correct pip package name
                        pip_package = package_mappings.get(missing_module, missing_module)
                        
                        # Skip if package is None (already provided by other packages)
                        if pip_package is not None:
                            # Install the missing module
                            pip_path = os.path.join(venv_dir, "bin", "pip")
                            console.print(f"  📦 Installing {pip_package}...", style="cyan")
                            install_result = subprocess.run(
                                [pip_path, "install", pip_package],
                                capture_output=True,
                                text=True,
                                timeout=120
                            )
                            
                            if install_result.returncode == 0:
                                console.print(f"  ✅ Installed {pip_package}, retrying tests...", style="green")
                                
                                # Retry the test execution with the same test_args
                                result = subprocess.run(
                                    test_args,
                                    cwd=temp_repo,
                                    capture_output=True,
                                    text=True,
                                    timeout=300
                                )
                                output = result.stdout + "\n" + result.stderr
                            else:
                                console.print(f"  ❌ Failed to install {pip_package}: {install_result.stderr[:100]}", style="red")
                        else:
                            console.print(f"  ℹ️  {missing_module} is provided by setuptools, skipping...", style="dim")
            
            # Parse test results (different formats for Django vs pytest)
            if use_django_runner:
                # Django format: "Ran X tests in Y.YYs" and "OK" or "FAILED (failures=X, errors=Y)"
                ran_match = re.search(r'Ran (\d+) tests? in', output)
                ok_match = re.search(r'\nOK\s*$', output, re.MULTILINE)
                failed_match = re.search(r'FAILED.*failures=(\d+)', output)
                error_match = re.search(r'FAILED.*errors=(\d+)', output)
                
                tests_total = int(ran_match.group(1)) if ran_match else 0
                tests_failed = int(failed_match.group(1)) if failed_match else 0
                tests_error = int(error_match.group(1)) if error_match else 0
                tests_passed = tests_total - tests_failed - tests_error if tests_total > 0 else 0
                
                # If no "Ran X tests" but we see "Found X test(s)", use that
                if tests_total == 0:
                    found_match = re.search(r'Found (\d+) test\(s\)', output)
                    if found_match:
                        tests_total = int(found_match.group(1))
                        # If it didn't finish, assume all are errors
                        if 'TypeError' in output or 'AttributeError' in output:
                            tests_error = tests_total
                            tests_passed = 0
            else:
                # Pytest format
                passed_match = re.search(r'(\d+) passed', output)
                failed_match = re.search(r'(\d+) failed', output)
                error_match = re.search(r'(\d+) error', output)
                
                tests_passed = int(passed_match.group(1)) if passed_match else 0
                tests_failed = int(failed_match.group(1)) if failed_match else 0
                tests_error = int(error_match.group(1)) if error_match else 0
                tests_total = tests_passed + tests_failed + tests_error
            
            # Format test output
            test_output = f"SWEBench Local Test Results:\n"
            test_output += f"Total: {tests_total}, Passed: {tests_passed}, Failed: {tests_failed}\n\n"
            test_output += output
            
            # Save test output
            with open(os.path.join(problem_dir, "test_output.txt"), "w") as f:
                f.write(test_output)
            
            console.print(f"  📊 Tests: {tests_passed}/{tests_total} passed", 
                         style="green" if tests_passed == tests_total else "yellow")
            
            return {
                "tests_passed": tests_passed,
                "tests_total": tests_total,
                "tests_failed": tests_failed,
                "test_output": test_output,
                "error": None if tests_passed == tests_total else "Some tests failed"
            }
    
    except subprocess.TimeoutExpired:
        return {
            "tests_passed": 0,
            "tests_total": 0,
            "tests_failed": 0,
            "test_output": "Test execution timeout",
            "error": "Timeout"
        }
    except Exception as e:
        import traceback
        error_msg = f"Error running SWEBench local test: {str(e)}\n{traceback.format_exc()}"
        console.print(f"  ❌ {error_msg}", style="red")
        return {
            "tests_passed": 0,
            "tests_total": 0,
            "tests_failed": 0,
            "test_output": error_msg,
            "error": str(e)
        }


def evaluate_problem(problem_name: str, agent_file: str, timeout: int, results_dir: str, reuse_existing: bool = False) -> Dict:
    """Evaluate agent on one problem.
    
    Args:
        problem_name: Name of the problem to evaluate
        agent_file: Path to agent file
        timeout: Timeout in seconds
        results_dir: Directory to save results
        reuse_existing: If True, reuse existing patch and only re-run tests
    """
    workspace_dir = None
    try:
        # Find problem across all suites
        problem_suites = get_problem_suites()
        suite = next((s for s in problem_suites if s.has_problem_name(problem_name)), None)
        
        if suite is None:
            return {
                "problem": problem_name,
                "score": 0.0,
                "tests_passed": 0,
                "tests_total": 0,
                "patch_generated": False,
                "error": f"Problem '{problem_name}' not found"
            }
        
        problem = suite.get_problem(problem_name)
        is_swebench = isinstance(suite, SWEBenchVerifiedSuite)
        
        problem_slug = problem_name.replace("/", "_").replace(" ", "_")
        problem_dir = os.path.join(results_dir, problem_slug)
        
        # Check if we should reuse existing results
        existing_patch_file = os.path.join(problem_dir, "patch.diff")
        if reuse_existing and os.path.exists(existing_patch_file):
            console.print(f"  ♻️  Reusing existing patch, re-running tests only...", style="cyan")
            
            # Load existing patch
            with open(existing_patch_file, "r") as f:
                patch = f.read()
            
            # Create temporary workspace for testing
            workspace_base = os.path.join(os.getcwd(), "result")
            os.makedirs(workspace_base, exist_ok=True)
            workspace_dir = tempfile.mkdtemp(prefix="retest_", dir=workspace_base)
            repo_dir = os.path.join(workspace_dir, "repo")
            os.makedirs(repo_dir, exist_ok=True)
            
            # Copy problem files
            suite.copy_problem_files_to_directory(problem, repo_dir, include_tests=True)
            
            logs = f"[REUSE MODE] Using existing patch from {existing_patch_file}\n"
            elapsed = 0.0
        else:
            # Run agent normally
            patch, logs, elapsed, repo_dir = run_agent_locally(agent_file, problem, suite, timeout)
            workspace_dir = os.path.dirname(repo_dir)
        
        # Save results with counter to avoid overwriting
        problem_slug = problem_name.replace("/", "_").replace(" ", "_")
        
        # Find next available counter for this problem
        counter = 1
        while True:
            if counter == 1:
                problem_dir = os.path.join(results_dir, problem_slug)
            else:
                problem_dir = os.path.join(results_dir, f"{problem_slug}_{counter}")
            
            if not os.path.exists(problem_dir):
                break
            counter += 1
        
        os.makedirs(problem_dir, exist_ok=True)
        
        if counter > 1:
            console.print(f"  💾 Saving as attempt #{counter} (previous attempts exist)", style="dim")
        
        # Save patch
        with open(os.path.join(problem_dir, "patch.diff"), "w") as f:
            f.write(patch if patch else "")
        
        # Save agent logs
        with open(os.path.join(problem_dir, "agent_logs.txt"), "w") as f:
            f.write(logs)
        
        # Run tests based on problem type
        if is_swebench:
            # SWEBench: Run tests locally without Docker
            test_results = run_tests_swebench_local(problem, patch, problem_dir)
        else:
            # Polyglot: Apply patch and run local pytest
            problem_dataset_dir = os.path.join("evaluator/datasets/polyglot", problem.name)
            
            # Apply patch if generated
            if patch and patch.strip():
                patch_file = os.path.join(repo_dir, "agent.patch")
                with open(patch_file, "w") as f:
                    f.write(patch)
                
                console.print(f"  🔧 Applying patch...", style="cyan")
                result = subprocess.run(
                    ["git", "apply", "--whitespace=fix", patch_file],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    console.print(f"  ⚠️ Patch failed: {result.stderr}", style="yellow")
                else:
                    console.print(f"  ✅ Patch applied", style="green")
            
            # Copy tests.py from dataset
            tests_src = os.path.join(problem_dataset_dir, "tests.py")
            if os.path.exists(tests_src):
                tests_dst = os.path.join(problem_dir, "tests.py")
                shutil.copy2(tests_src, tests_dst)
            
            # Run tests
            main_src = os.path.join(repo_dir, "main.py")
            test_results = {"tests_passed": 0, "tests_total": 0, "tests_failed": 0, "error": "No solution generated"}
            if os.path.exists(main_src):
                test_results = run_tests_polyglot(repo_dir, problem.name, problem_dir)
            else:
                console.print(f"  ❌ No main.py generated", style="red")
        
        # Copy final main.py AFTER patch application (for polyglot)
        main_src = os.path.join(repo_dir, "main.py")
        if os.path.exists(main_src):
            main_dst = os.path.join(problem_dir, "main.py")
            shutil.copy2(main_src, main_dst)
        
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


def benchmark_agent(agent_file: str, problems: List[str], timeout: int, reuse_existing: bool = False):
    """Run benchmark on all problems.
    
    Args:
        agent_file: Path to agent file
        problems: List of problem names to test
        timeout: Timeout per problem in seconds
        reuse_existing: If True, reuse existing patches and only re-run tests
    """
    # Use timestamp-based directory for this benchmark run
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("result", f"benchmark_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    console.print("📡 Requires inference gateway at http://localhost:8000", style="cyan")
    console.print("💡 Start it: python -m inference_gateway.main", style="dim")
    console.print("🎉 Testing locally - NO Docker required!", style="green")
    console.print("📦 Using cached venvs in .testvenv/ (first run will install deps)\n", style="dim")
    
    console.print(Panel(
        f"[bold cyan]🏆 Agent Benchmark (Local Mode)[/bold cyan]\n\n"
        f"[yellow]Agent:[/yellow] {agent_file}\n"
        f"[yellow]Problems:[/yellow] {len(problems)}\n"
        f"[yellow]Timeout:[/yellow] {timeout}s per problem",
        title="🎯 Configuration",
        border_style="cyan"
    ))
    
    results = []
    
    # Suppress suite loading logs during benchmark (they clutter progress bar)
    import logging
    suite_logger = logging.getLogger('polyglot_suite')
    swe_logger = logging.getLogger('swebench_verified_suite')
    original_level_poly = suite_logger.level
    original_level_swe = swe_logger.level
    suite_logger.setLevel(logging.WARNING)
    swe_logger.setLevel(logging.WARNING)
    
    try:
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
                
                result = evaluate_problem(problem, agent_file, timeout, results_dir, reuse_existing=reuse_existing)
                results.append(result)
                
                if result["error"] and not result['patch_generated']:
                    console.print(f"  [red]❌ {result['error'][:50]}[/red]")
                else:
                    tests_total = result.get('tests_total', 0)
                    tests_passed = result.get('tests_passed', 0)
                    elapsed = result.get('elapsed_time', 0)
                    time_str = f" [dim]({elapsed:.1f}s)[/dim]" if elapsed > 0 else ""
                    if tests_total > 0:
                        emoji = "🎉" if tests_passed == tests_total else "⚠️" if tests_passed > 0 else "❌"
                        console.print(f"  {emoji} Tests: {tests_passed}/{tests_total} passed ({result['score']*100:.1f}%){time_str}")
                    else:
                        emoji = "🎉" if result['patch_generated'] else "❌"
                        console.print(f"  {emoji} Patch: {'Generated' if result['patch_generated'] else 'Failed'}{time_str}")
                
                progress.update(task, advance=1)
    finally:
        # Restore original logging levels
        suite_logger.setLevel(original_level_poly)
        swe_logger.setLevel(original_level_swe)
    
    # Calculate scores
    valid_results = [r for r in results if r["error"] is None]
    final_score = sum(r['score'] for r in valid_results) / len(valid_results) if valid_results else 0.0
    
    # Display table
    console.print("\n" + "="*80)
    table = Table(title="📊 Results", show_header=True, header_style="bold magenta")
    table.add_column("Problem", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Tests", justify="center")
    table.add_column("Time", justify="right")
    table.add_column("Status")
    
    for result in results:
        score_str = f"{result['score']*100:.1f}%"
        tests_str = f"{result.get('tests_passed', 0)}/{result.get('tests_total', 0)}"
        elapsed = result.get('elapsed_time', 0)
        time_str = f"{elapsed:.1f}s" if elapsed > 0 else "N/A"
        
        if result['score'] == 1.0:
            status = "✅ All passed"
        elif result['score'] > 0:
            status = "⚠️ Partial"
        elif result.get('error'):
            status = f"❌ {result['error'][:20]}"
        else:
            status = "❌ Failed"
        
        table.add_row(result['problem'], score_str, tests_str, time_str, status)
    
    console.print(table)
    
    # Summary
    console.print("\n" + "="*80)
    total_tests = sum(r.get('tests_total', 0) for r in results)
    total_passed = sum(r.get('tests_passed', 0) for r in results)
    total_time = sum(r.get('elapsed_time', 0) for r in results)
    avg_time = total_time / len(results) if results else 0
    
    console.print(Panel(
        f"[bold cyan]Final Score:[/bold cyan] [bold yellow]{final_score*100:.1f}%[/bold yellow]\n\n"
        f"Total Tests: {total_passed}/{total_tests} passed\n"
        f"Problems: {sum(1 for r in results if r['score'] == 1.0)}/{len(results)} fully solved\n"
        f"Patches generated: {sum(1 for r in results if r['patch_generated'])}/{len(results)}\n"
        f"Total Time: {total_time:.1f}s | Avg: {avg_time:.1f}s per problem",
        title="🏆 Summary",
        border_style="green" if final_score > 0.8 else "yellow"
    ))
    
    # Save summary
    summary_data = {

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
    readme_content = f"""# Benchmark Results:

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
   - `test_output.txt` to see which tests passed/failed and error messages
   - `agent_logs.txt` to debug agent behavior
   
## Re-running Tests Manually

You can re-run tests for any problem:
```bash
cd result/benchmark_<problem-name>/<PROBLEM_NAME>/
python -m pytest tests.py -v
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
    console.print("   - [cyan]README.md[/cyan] - Human-readable summary")
    console.print("   - [cyan]summary.json[/cyan] - Machine-readable data")
    console.print("   - [cyan]<problem>/main.py[/cyan] - 🎯 Generated solution (tested)")
    console.print("   - [cyan]<problem>/tests.py[/cyan] - Test cases from dataset")
    console.print("   - [cyan]<problem>/test_output.txt[/cyan] - 📊 Complete test logs")
    console.print("\n   Quick access: [cyan]result/latest_benchmark.json[/cyan]")


def test_existing_patch(problem_name: str, patch_path: str) -> Dict:
    """Test an existing patch without regenerating it."""
    console.print(f"\n[bold cyan]Testing existing patch for: {problem_name}[/bold cyan]")
    
    # Find problem across all suites
    problem_suites = get_problem_suites()
    suite = next((s for s in problem_suites if s.has_problem_name(problem_name)), None)
    
    if suite is None:
        console.print(f"[bold red]❌ Problem not found: {problem_name}[/bold red]")
        return None
    
    problem = suite.get_problem(problem_name)
    is_swebench = isinstance(suite, SWEBenchVerifiedSuite)
    
    # Read existing patch
    if not os.path.exists(patch_path):
        console.print(f"[bold red]❌ Patch file not found: {patch_path}[/bold red]")
        return None
    
    with open(patch_path, 'r') as f:
        patch = f.read()
    
    console.print(f"  📄 Loaded patch: {len(patch)} chars")
    
    # Create results directory
    problem_slug = problem_name.replace("/", "_").replace(" ", "_")
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    problem_dir = os.path.join("result", f"retest_{timestamp}_{problem_slug}")
    os.makedirs(problem_dir, exist_ok=True)
    
    # Run tests with timing
    import time
    start_time = time.time()
    
    if is_swebench:
        test_results = run_tests_swebench_local(problem, patch, problem_dir)
    else:
        console.print(f"[yellow]⚠️  Polyglot testing not yet supported for --test-only[/yellow]")
        return None
    
    elapsed_time = time.time() - start_time
    
    # Calculate score
    score = (test_results["tests_passed"] / test_results["tests_total"] * 100) if test_results["tests_total"] > 0 else 0
    
    # Display results
    console.print(f"\n[bold]{'='*80}[/bold]")
    console.print("[bold cyan]                      📊 Test Results[/bold cyan]")
    console.print(f"[bold cyan]Problem:[/bold cyan] {problem_name}")
    console.print(f"[bold cyan]Score:[/bold cyan] {score:.1f}%")
    console.print(f"[bold cyan]Tests:[/bold cyan] {test_results['tests_passed']}/{test_results['tests_total']} passed")
    console.print(f"[bold cyan]Time:[/bold cyan] {elapsed_time:.1f}s")
    if test_results.get('error'):
        console.print(f"[bold red]Error:[/bold red] {test_results['error']}")
    console.print(f"[bold]{'='*80}[/bold]")
    
    return test_results


def main():
    parser = argparse.ArgumentParser(description="Simple local agent benchmark")
    parser.add_argument("--agent-file", default= AGENT_FILE_NAME, help="Agent file (default: ${AGENT_FILE_NAME})")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout per problem (default: 30s)")
    parser.add_argument("--reuse", action="store_true", help="Reuse existing patches if available, only re-run tests")
    parser.add_argument("--test-only", type=str, help="Test existing patch. Format: problem_name[:patch_path]. If patch_path omitted, auto-finds latest.")
    parser.add_argument("--force-rebuild-venv", action="store_true", help="Force rebuild of cached venv (useful if venv is broken)")
    parser.add_argument("problem", nargs="?", help="Problem name to test (shorthand for --test-only)")
    
    args = parser.parse_args()
    
    # Handle --test-only mode or positional problem argument
    test_problem = args.test_only or args.problem
    
    if test_problem:
        # Parse problem_name and optional patch_path
        if ":" in test_problem:
            problem_name, patch_path = test_problem.split(":", 1)
        else:
            problem_name = test_problem
            # Auto-find patch in result directories (search for most recent)
            patch_path = None
            problem_slug = problem_name.replace("/", "_").replace(" ", "_")
            
            # Search for most recent benchmark/retest directories
            result_base = Path("result")
            if result_base.exists():
                # Find all matching directories (benchmark_*, retest_*)
                candidates = []
                for dir_path in result_base.iterdir():
                    if dir_path.is_dir():
                        # Check for patch in problem subdirectory
                        patch_file = dir_path / problem_slug / "patch.diff"
                        if patch_file.exists():
                            candidates.append((dir_path.stat().st_mtime, str(patch_file)))
                
                # Use most recent
                if candidates:
                    candidates.sort(reverse=True)  # Most recent first
                    patch_path = candidates[0][1]
                    console.print(f"[dim]📁 Auto-found patch: {patch_path}[/dim]")
            
            if not patch_path:
                console.print(f"[bold red]❌ No patch found for {problem_name}[/bold red]")
                console.print(f"[yellow]Searched in: result/*/{problem_slug}/patch.diff[/yellow]")
                console.print(f"\n[yellow]Specify path explicitly: --test-only {problem_name}:path/to/patch.diff[/yellow]")
                sys.exit(1)
        
        test_existing_patch(problem_name, patch_path)
        return
    
    # Normal benchmark mode
    if not Path(args.agent_file).exists():
        console.print(f"[bold red]❌ Agent file not found: {args.agent_file}[/bold red]")
        sys.exit(1)
    
    if not DEFAULT_TEST_PROBLEMS:
        console.print("[bold red]❌ No problems in DEFAULT_TEST_PROBLEMS[/bold red]")
        sys.exit(1)
    
    benchmark_agent(args.agent_file, DEFAULT_TEST_PROBLEMS, args.timeout, reuse_existing=args.reuse)


if __name__ == "__main__":
    main()
