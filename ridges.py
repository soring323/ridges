#!.venv/bin/python3

"""
Ridges CLI - Elegant command-line interface for managing Ridges miners and validators
"""

import hashlib
import time
from fiber.chain.chain_utils import load_hotkey_keypair
import httpx
import os
import subprocess
from typing import Optional
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt

console = Console()
DEFAULT_API_BASE_URL = "https://platform-v2.ridges.ai"
# CONFIG_FILE = "miner/.env"

# load_dotenv(CONFIG_FILE)
# load_dotenv(".env")

validator_tracing = False
if os.getenv("DD_API_KEY") and os.getenv("DD_APP_KEY") and os.getenv("DD_HOSTNAME") and os.getenv("DD_SITE") and os.getenv("DD_ENV") and os.getenv("DD_SERVICE"):
    validator_tracing = True

def run_cmd(cmd: str, capture: bool = True) -> tuple[int, str, str]:
    """Run command and return (code, stdout, stderr)"""
    try:
        if capture:
            result = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
            return result.returncode, result.stdout, result.stderr
        else:
            # For non-captured commands, use Popen for better KeyboardInterrupt handling
            process = subprocess.Popen(cmd, shell=True)
            try:
                return_code = process.wait()
                return return_code, "", ""
            except KeyboardInterrupt:
                # Properly terminate the subprocess
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
    except KeyboardInterrupt:
        # Forward KeyboardInterrupt to subprocess by killing it
        # This ensures proper cleanup when user presses Ctrl+C
        raise
run_cmd("uv add click")
import click

def check_docker(image: str) -> bool:
    """Check if Docker image exists"""
    code, output, _ = run_cmd(f"docker images -q {image}")
    return code == 0 and output.strip() != ""

def build_docker(path: str, tag: str) -> bool:
    """Build Docker image"""
    console.print(f"🔨 Building {tag}...", style="yellow")
    return run_cmd(f"docker build -t {tag} {path}", capture=False)[0] == 0

def check_pm2(process: str = "ridges-validator") -> tuple[bool, str]:
    """Check if PM2 process is running"""
    code, output, _ = run_cmd("pm2 list")
    if code != 0:
        return False, ""
    
    # Parse PM2 list output to find exact process name
    lines = output.strip().split('\n')
    for line in lines:
        if '│' in line and process in line:
            # Split by │ and get the name column (index 2, after id and │)
            parts = line.split('│')
            if len(parts) >= 3:
                process_name = parts[2].strip()
                if process_name == process:
                    return True, "running"
    return False, ""

def get_logs(process: str = "ridges-validator", lines: int = 15) -> str:
    """Get PM2 logs"""
    return run_cmd(f"pm2 logs {process} --lines {lines} --nostream")[1]

class Config:
    def __init__(self):
        self.file = CONFIG_FILE
        if os.path.exists(self.file):
            load_dotenv(self.file)
    
    def save(self, key: str, value: str):
        lines = []
        if os.path.exists(self.file):
            with open(self.file, 'r') as f:
                lines = f.readlines()
        
        found = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{key}="):
                lines[i] = f"{key}={value}\n"
                found = True
                break
        
        if not found:
            lines.append(f"{key}={value}\n")
        
        with open(self.file, 'w') as f:
            f.writelines(lines)
        os.environ[key] = value
    
    def get_or_prompt(self, key: str, prompt: str, default: Optional[str] = None) -> str:
        value = os.getenv(key)
        if not value:
            value = Prompt.ask(f"🎯 {prompt}", default=default) if default else Prompt.ask(f"🎯 {prompt}")
            self.save(key, value)
        return value

class RidgesCLI:
    def __init__(self, api_url: Optional[str] = None):
        self.api_url = api_url or DEFAULT_API_BASE_URL
        self.config = Config()
    
    def get_keypair(self, coldkey_name: Optional[str] = None, hotkey_name: Optional[str] = None):
        coldkey = coldkey_name or self.config.get_or_prompt("RIDGES_COLDKEY_NAME", "Enter your coldkey name", "miner")
        hotkey = hotkey_name or self.config.get_or_prompt("RIDGES_HOTKEY_NAME", "Enter your hotkey name", "default")
        return load_hotkey_keypair(coldkey, hotkey)
    
    def get_agent_path(self) -> str:
        return self.config.get_or_prompt("RIDGES_AGENT_FILE", "Enter the path to your agent.py file", "miner/agent.py")

@click.group()
@click.version_option(version="1.0.0")
@click.option("--url", help=f"Custom API URL (default: {DEFAULT_API_BASE_URL})")
@click.pass_context
def cli(ctx, url):
    """Ridges CLI - Manage your Ridges miners and validators"""
    ctx.ensure_object(dict)
    ctx.obj['url'] = url

@cli.command()
@click.option("--file", help="Path to agent.py file")
@click.option("--coldkey-name", help="Coldkey name")
@click.option("--hotkey-name", help="Hotkey name")
@click.pass_context
def upload(ctx, hotkey_name: Optional[str], file: Optional[str], coldkey_name: Optional[str]):
    """Upload a miner agent to the Ridges API."""
    ridges = RidgesCLI(ctx.obj.get('url'))
    
    file = file or ridges.get_agent_path()
    if not os.path.exists(file) or os.path.basename(file) != "agent.py":
        console.print("💥 File must be named 'agent.py' and exist", style="bold red")
        return
    
    console.print(Panel(f"[bold cyan] Uploading Agent[/bold cyan]\n[yellow]File:[/yellow] {file}\n[yellow]API:[/yellow] {ridges.api_url}", title="🚀 Upload", border_style="cyan"))
    
    try:
        with open(file, 'rb') as f:
            files = {'agent_file': ('agent.py', f, 'text/plain')}
            content_hash = hashlib.sha256(f.read()).hexdigest()
            keypair = ridges.get_keypair(coldkey_name, hotkey_name)
            public_key = keypair.public_key.hex()
            
            name_and_prev_version = get_name_and_prev_version(ridges.api_url, keypair.ss58_address)
            if name_and_prev_version is None:
                name = Prompt.ask("Enter a name for your miner agent")
                version_num = -1
            else:
                name, prev_version_num = name_and_prev_version
                version_num = prev_version_num + 1

            file_info = f"{keypair.ss58_address}:{content_hash}:{version_num}"
            signature = keypair.sign(file_info).hex()
            payload = {'public_key': public_key, 'file_info': file_info, 'signature': signature, 'name': name}

            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
                progress.add_task("🔐 Signing and uploading...", total=None)

                with httpx.Client() as client:
                    response = client.post(f"{ridges.api_url}/upload/agent", files=files, data=payload, timeout=120)
                
                if response.status_code == 200:
                    console.print(Panel(f"[bold green]🎉 Upload Complete[/bold green]\n[cyan]Miner '{name}' uploaded successfully![/cyan]", title="✨ Success", border_style="green"))
                else:
                    error = response.json().get('detail', 'Unknown error') if response.headers.get('content-type', '').startswith('application/json') else response.text
                    console.print(f"💥 Upload failed: {error}", style="bold red")
                    
    except Exception as e:
        console.print(f"💥 Error: {e}", style="bold red")

def get_name_and_prev_version(url: str, miner_hotkey: str) -> Optional[tuple[str, int]]:
    try:
        with httpx.Client() as client:
            response = client.get(f"{url}/retrieval/agent-by-hotkey?miner_hotkey={miner_hotkey}")
            if response.status_code == 404:
                return None
            if response.status_code == 200:
                latest_agent = response.json()
                if latest_agent:
                    return latest_agent.get("name"), latest_agent.get("version_num")
    except Exception as e:
        console.print(f"💥 Error: {e}", style="bold red")
        exit(1)



@cli.group()
def validator():
    """Manage Ridges validators"""
    pass

@validator.command()
@click.option("--no-auto-update", is_flag=True, help="Run validator directly in foreground")
@click.option("--no-follow", is_flag=True, help="Start validator but don't follow logs")
def run(no_auto_update: bool, no_follow: bool):
    """Run the Ridges validator."""

    if no_auto_update:
        console.print("🚀 Starting validator...", style="yellow")
        if validator_tracing:
            run_cmd("uv run -m validator.main", capture=False)
        else:
            run_cmd("uv run -m validator.main", capture=False)
        return
    
    # Check if already running
    is_running, _ = check_pm2("ridges-validator-updater")
    if is_running:
        console.print("✅ Auto-updater already running!", style="green")
        if not no_follow:
            console.print(" Showing validator logs...", style="cyan")
            run_cmd("pm2 logs ridges-validator ridges-validator-updater", capture=False)
        return
    
    # Start validator
    console.print("🚀 Starting validator...", style="yellow")
    run_cmd("uv pip install -e .", capture=False)
    run_cmd("pm2 start 'uv run -m validator.main' --name ridges-validator", capture=False)
    
    # Start auto-updater in background
    if run_cmd(f"pm2 start './ridges.py validator update --every 5' --name ridges-validator-updater", capture=False)[0] == 0:
        console.print(Panel(f"[bold green] Auto-updater started![/bold green]\n[cyan]Validator running with auto-updates every 5 minutes.[/cyan]", title="✨ Success", border_style="green"))
        if not no_follow:
            console.print(" Showing validator logs...", style="cyan")
            run_cmd("pm2 logs ridges-validator ridges-validator-updater", capture=False)
    else:
        console.print("💥 Failed to start validator", style="red")

@validator.command()
def stop():
    """Stop the Ridges validator."""
    stopped = [p for p in ["ridges-validator", "ridges-validator-updater"] if run_cmd(f"pm2 delete {p}")[0] == 0]
    if stopped:
        console.print(Panel(f"[bold green] Stopped:[/bold green]\n[cyan]{', '.join(stopped)}[/cyan]", title="✨ Stop Complete", border_style="green"))
    else:
        console.print("  No validator processes running", style="cyan")

@validator.command()
def logs():
    """Show validator logs."""
    console.print("📋 Showing validator logs...", style="cyan")
    run_cmd("pm2 logs ridges-validator ridges-validator-updater", capture=False)

@validator.command()
@click.option("--every", default=None, type=int, help="Run in loop every N minutes (default: update once and exit)")
def update(every: Optional[int]):
    """Update validator code and restart."""
    while True:
        # Get current commit and pull updates
        code, current_commit, _ = run_cmd("git rev-parse HEAD")
        if code != 0 or run_cmd("git pull --autostash")[0] != 0:
            console.print("💥 Git operation failed", style="red")
            console.print("Resetting with `git reset --hard HEAD`")
            if run_cmd("git reset --hard HEAD")[0] != 0:
                console.print("💥 Git reset failed", style="red")
                return
        
        # Check if updates applied
        code, new_commit, _ = run_cmd("git rev-parse HEAD")
        if code != 0 or current_commit.strip() == new_commit.strip():
            console.print("No updates available")
            if not every:
                break
            console.print(f"Sleeping for {every} minutes...")
            time.sleep(every * 60)
            continue
        
        # Update deps and restart validator
        console.print("✨ Updates found! Restarting validator...", style="green")
        run_cmd("uv pip install -e .")
        is_running, _ = check_pm2("ridges-validator")
        run_cmd("pm2 restart ridges-validator" if is_running else "pm2 start 'uv run -m validator.main' --name ridges-validator")
        console.print("Validator updated!")
        
        if not every:
            break
            
        console.print(f"Sleeping for {every} minutes...")
        time.sleep(every * 60)

@cli.group()
def platform():
    """Manage Ridges API platform"""
    pass

@platform.command()
@click.option("--no-auto-update", is_flag=True, help="Run platform directly in foreground")
def run(no_auto_update: bool):
    """Run the Ridges API platform."""
    console.print(Panel(f"[bold cyan]🚀 Starting Platform[/bold cyan]", title="🌐 Platform", border_style="cyan"))
    
    # Check if running
    is_running, _ = check_pm2("ridges-api-platform")
    if is_running:
        console.print(Panel("[bold yellow]⚠️  Platform already running![/bold yellow]", title="🔄 Status", border_style="yellow"))
        return
    
    if no_auto_update:
        console.print(" Starting platform...", style="yellow")
        run_cmd("uv run -m api.src.main", capture=False)
        return

    # Remove old venv, create new venv, activate new venv, download dependencies
    if run_cmd("rm -rf .venv")[0] == 0:
        console.print("🔄 Removed old venv", style="yellow")
    else:
        console.print("💥 Failed to remove old venv", style="red")
        return
    if run_cmd("uv venv .venv")[0] == 0:
        console.print("🔄 Created new venv", style="yellow")
    else:
        console.print("💥 Failed to create new venv", style="red")
        return
    if run_cmd(". .venv/bin/activate")[0] == 0:
        console.print("🔄 Activated new venv", style="yellow")
    else:
        console.print("💥 Failed to activate new venv", style="red")
        return
    if run_cmd("uv pip install -e .")[0] == 0:
        console.print("🔄 Downloaded dependencies", style="yellow")
    else:
        console.print("💥 Failed to download dependencies", style="red")
        return
    
    # Start platform
    if run_cmd(f"pm2 start 'uv run -m api.src.main' --name ridges-api-platform", capture=False)[0] == 0:
        console.print(Panel(f"[bold green] Platform started![/bold green] Running on 0.0.0.0:8000", title="✨ Success", border_style="green"))
        console.print(" Showing platform logs...", style="cyan")
        run_cmd("pm2 logs ridges-api-platform", capture=False)
    else:
        console.print("💥 Failed to start platform", style="red")

@platform.command()
def stop():
    """Stop the Ridges API platform."""
    if run_cmd("pm2 delete ridges-api-platform")[0] == 0:
        console.print(Panel("[bold green] Platform stopped![/bold green]", title="✨ Stop Complete", border_style="green"))
    else:
        console.print("⚠️  Platform not running", style="yellow")

@platform.command()
def logs():
    """Show platform logs."""
    console.print(" Showing platform logs...", style="cyan")
    run_cmd("pm2 logs ridges-api-platform", capture=False)

@platform.command()
def update():
    """Update platform code and restart."""
    # Get current commit and pull updates
    code, current_commit, _ = run_cmd("git rev-parse HEAD")
    if code != 0:
        console.print(" Failed to get commit hash", style="red")
        return
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
        progress.add_task("📥 Pulling changes...", total=None)
        if run_cmd("git pull")[0] != 0:
            console.print("💥 Git pull failed", style="red")
            return
    
    # Check if updates applied
    code, new_commit, _ = run_cmd("git rev-parse HEAD")
    if code != 0 or current_commit.strip() == new_commit.strip():
        console.print(Panel("[bold yellow] No updates available[/bold yellow]", title="📋 Status", border_style="yellow"))
        return 0
    
    # Update deps and restart
    console.print(Panel("[bold green]✨ Updates found![/bold green]\n[cyan]Updating deps and restarting...[/cyan]", title="🔄 Update", border_style="green"))
    run_cmd("uv pip install -e .")
    run_cmd("pm2 restart ridges-api-platform")
    console.print(Panel("[bold green]🎉 Platform updated![/bold green]", title="✨ Complete", border_style="green"))

@cli.group()
def proxy():
    """Manage Ridges proxy server"""
    pass

@proxy.command()
@click.option("--no-auto-update", is_flag=True, help="Run proxy directly in foreground")
def run(no_auto_update: bool):
    """Run the Ridges proxy server."""
    
    # Check if running
    is_running, _ = check_pm2("ridges-proxy")
    if is_running:
        console.print(Panel("[bold yellow]⚠️  Proxy already running![/bold yellow]", title="🔄 Status", border_style="yellow"))
        return
    
    if no_auto_update:
        console.print(" Starting proxy server...", style="yellow")
        run_cmd("python -m inference_gateway.main", capture=False)
        return

    # Get port from inference gateway config
    try:
        import inference_gateway.config as gateway_config
        port = gateway_config.PORT
        port_msg = f"Running on port {port}"
    except Exception:
        port_msg = "Running"


    # Start proxy with PM2
    if run_cmd(f"pm2 start 'python -m inference_gateway.main' --name ridges-proxy", capture=False)[0] == 0:
        console.print(Panel(f"[bold green]🎉 Proxy started![/bold green] {port_msg}", title="✨ Success", border_style="green"))
        console.print(" Showing proxy logs...", style="cyan")
        run_cmd("pm2 logs ridges-proxy", capture=False)
    else:
        console.print(" Failed to start proxy", style="red")

@proxy.command()
def stop():
    """Stop the Ridges proxy server."""
    if run_cmd("pm2 delete ridges-proxy")[0] == 0:
        console.print(Panel("[bold green] Proxy stopped![/bold green]", title="✨ Stop Complete", border_style="green"))
    else:
        console.print("⚠️  Proxy not running", style="yellow")

@proxy.command()
def logs():
    """Show proxy logs."""
    console.print(" Showing proxy logs...", style="cyan")
    run_cmd("pm2 logs ridges-proxy", capture=False)

@cli.group()
def local():
    """Manage all Ridges services locally"""
    pass

@local.command()
def run():
    """Run all Ridges services (platform, proxy, validator) in the background with PM2."""
    console.print(Panel(f"[bold cyan] Starting All Services[/bold cyan]", title=" Local Environment", border_style="cyan"))
    
    services_started = []
    services_already_running = []
    services_failed = []
    
    # Start Platform
    console.print(" Starting platform...", style="yellow")
    is_running, _ = check_pm2("ridges-api-platform")
    if is_running:
        services_already_running.append("Platform")
        console.print("✅ Platform already running!", style="green")
    else:
        # Install dependencies and start platform
        run_cmd("uv pip install -e .", capture=False)
        if run_cmd(f"pm2 start 'uv run -m api.src.main' --name ridges-api-platform", capture=False)[0] == 0:
            services_started.append("Platform")
            console.print("✅ Platform started!", style="green")
        else:
            services_failed.append("Platform")
            console.print(" Failed to start platform", style="red")
    
    # Start Proxy
    console.print("🔗 Starting proxy...", style="yellow")
    is_running, _ = check_pm2("ridges-proxy")
    if is_running:
        services_already_running.append("Proxy")
        console.print("✅ Proxy already running!", style="green")
    else:
        if run_cmd(f"pm2 start 'uv run -m proxy.main' --name ridges-proxy", capture=False)[0] == 0:
            services_started.append("Proxy")
            console.print("✅ Proxy started!", style="green")
        else:
            services_failed.append("Proxy")
            console.print(" Failed to start proxy", style="red")
    
    # Start Validator
    console.print("🔍 Starting validator...", style="yellow")
    is_running, _ = check_pm2("ridges-validator")
    if is_running:
        services_already_running.append("Validator")
        console.print("✅ Validator already running!", style="green")
    else:
        if run_cmd("pm2 start 'uv run -m validator.main' --name ridges-validator", capture=False)[0] == 0:
            services_started.append("Validator")
            console.print("✅ Validator started!", style="green")
        else:
            services_failed.append("Validator")
            console.print(" Failed to start validator", style="red")
    
    # Summary
    console.print("\n" + "="*50)
    if services_started:
        console.print(f"[bold green]🎉 Started:[/bold green] {', '.join(services_started)}")
    if services_already_running:
        console.print(f"[bold yellow]⚠️  Already Running:[/bold yellow] {', '.join(services_already_running)}")
    if services_failed:
        console.print(f"[bold red] Failed:[/bold red] {', '.join(services_failed)}")
    
    # Show final status
    total_services = len(services_started) + len(services_already_running)
    if total_services == 3:
        console.print(Panel(f"[bold green] All services are running![/bold green]\n[cyan]Platform, Proxy, and Validator are active[/cyan]", title="✨ Success", border_style="green"))
        console.print(" Use 'pm2 logs' to view all logs or './ridges.py local logs' for combined logs", style="cyan")
    else:
        console.print(Panel(f"[bold yellow]⚠️  {total_services}/3 services running[/bold yellow]", title="🔄 Status", border_style="yellow"))

@local.command()
def stop():
    """Stop all Ridges services."""
    services = ["ridges-api-platform", "ridges-proxy", "ridges-validator"]
    stopped = []
    
    for service in services:
        if run_cmd(f"pm2 delete {service}")[0] == 0:
            stopped.append(service.replace("ridges-", "").replace("-", " ").title())
    
    if stopped:
        console.print(Panel(f"[bold green]🎉 Stopped:[/bold green]\n[cyan]{', '.join(stopped)}[/cyan]", title="✨ Stop Complete", border_style="green"))
    else:
        console.print(" No services running", style="cyan")

@local.command()
def logs():
    """Show logs for all services."""
    console.print("📋 Showing logs for all services...", style="cyan")
    run_cmd("pm2 logs ridges-api-platform ridges-proxy ridges-validator", capture=False)

@local.command()
def status():
    """Show status of all services."""
    console.print(Panel(f"[bold cyan] Service Status[/bold cyan]", title="🔍 Status Check", border_style="cyan"))
    
    services = [
        ("ridges-api-platform", "Platform"),
        ("ridges-proxy", "Proxy"),
        ("ridges-validator", "Validator")
    ]
    
    for pm2_name, display_name in services:
        is_running, status = check_pm2(pm2_name)
        if is_running:
            console.print(f"✅ {display_name}: [bold green]Running[/bold green]")
        else:
            console.print(f"❌ {display_name}: [bold red]Stopped[/bold red]")


@cli.command()
@click.argument("problem_name")
@click.argument("agent_file")
@click.option("--local", is_flag=True, help="Run agent locally without Docker (no sandbox isolation)")
@click.option("--log-docker-to-stdout", is_flag=True, help="Print Docker container logs to stdout in real-time")
@click.option("--include-solution", is_flag=True, help="Expose the solution to the agent at /sandbox/solution.diff")
@click.option("--verbose", is_flag=True, help="Enable verbose (debug) logging")
@click.option("--timeout", default=10, type=int, help="Timeout in seconds for sandbox execution (default: 10)")
@click.option("--cleanup", is_flag=True, default=True, help="Clean up containers after test")
@click.option("--gateway-url", help="URL for the gateway (overrides RIDGES_PROXY_URL)")
def test_agent(
    problem_name: str,
    agent_file: str,
    local: bool,
    log_docker_to_stdout: bool,
    include_solution: bool,
    verbose: bool,
    timeout: int,
    cleanup: bool,
    gateway_url: str
):
    """Test your agent locally with full SWE-bench evaluation.
#
    This command runs a single agent against a specific problem. It automatically searches
    all available problem suites (Polyglot and SWE-bench Verified) to find the problem,
    handles sandbox creation (Docker or local), and provides detailed output about the agent's performance.
#
#
    Examples:
        # Start inference gateway first (required for both modes):
        python -m inference_gateway.main
        
        # Run locally without Docker (faster, no isolation):
        ./ridges.py test-agent affine-cipher miner/agent.py --local
        ./ridges.py test-agent affine-cipher miner/agent.py --local --include-solution
        
        # Run with Docker (full isolation):
        ./ridges.py test-agent affine-cipher miner/agent.py
        ./ridges.py test-agent django__django-12308 miner/agent.py --include-solution --verbose
#
    Note:
        - Both modes require inference gateway running at http://localhost:8000
        - Local mode (--local): Runs agent directly, no Docker/sandbox required, faster
        - Docker mode (default): Full isolation with Docker sandbox containers
        - Problems with >150 tests are rejected to prevent excessive resource usage
    """
#
    import os
    import sys
    import time
    import uuid
    import json
    import shutil
    import tempfile
    import traceback
    import subprocess
    import socket
    import importlib.util
    from pathlib import Path
#
    def get_local_ip():
        """Get the local IP address for the inference gateway."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                # Connect to a remote address (doesn't actually send data)
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception as e:
            console.print(f"⚠️ Could not determine local IP: {e}", style="yellow")
            console.print("   You can manually specify with --gateway-url http://YOUR_IP:8000", style="dim")
            return "192.168.1.100"  # Fallback IP
    
    def run_agent_locally(agent_file_path: str, problem, suite, include_solution: bool, timeout: int, verbose: bool):
        """Run agent locally without Docker."""
        # Create temporary workspace in project directory
        workspace_base = os.path.join(os.getcwd(), "result")
        os.makedirs(workspace_base, exist_ok=True)
        workspace_dir = tempfile.mkdtemp(prefix="local_", dir=workspace_base)
        console.print(f"📁 Created local workspace: {workspace_dir}", style="dim")
        
        try:
            # Create directory structure
            repo_dir = os.path.join(workspace_dir, "repo")
            os.makedirs(repo_dir, exist_ok=True)
            
            # Copy agent file
            agent_path = os.path.join(workspace_dir, "agent.py")
            shutil.copy2(agent_file_path, agent_path)
            console.print(f"✓ Copied agent to workspace", style="dim")
            
            # Copy problem files
            suite.copy_problem_files_to_directory(problem, repo_dir)
            console.print(f"✓ Copied problem files to {repo_dir}", style="dim")
            
            # Copy solution if requested
            if include_solution:
                solution_path = os.path.join(workspace_dir, "solution.diff")
                with open(solution_path, "w") as f:
                    f.write(problem.solution_diff)
                console.print(f"✓ Copied solution to workspace", style="dim")
            
            # Create input data
            input_data = {"problem_statement": problem.problem_statement}
            input_path = os.path.join(workspace_dir, "input.json")
            with open(input_path, "w") as f:
                json.dump(input_data, f, indent=2)
            console.print(f"✓ Created input.json", style="dim")
            
            # Load and run agent
            console.print(f"\n▶️  Running agent (timeout: {timeout}s)...", style="yellow")
            
            # Set environment variable for inference gateway (local mode uses localhost)
            # The agent expects SANDBOX_PROXY_URL to point to the inference gateway
            os.environ["SANDBOX_PROXY_URL"] = "http://localhost:8000"
            console.print(f"✓ Set SANDBOX_PROXY_URL to http://localhost:8000", style="dim")
            
            # Change to workspace directory for agent execution
            original_cwd = os.getcwd()
            os.chdir(workspace_dir)
            
            # Add workspace to Python path
            sys.path.insert(0, workspace_dir)
            
            try:
                # Import agent module
                spec = importlib.util.spec_from_file_location("agent", agent_path)
                agent_module = importlib.util.module_from_spec(spec)
                
                # Capture stdout/stderr
                from io import StringIO
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                captured_output = StringIO()
                sys.stdout = captured_output
                sys.stderr = captured_output
                
                try:
                    # Execute agent module
                    spec.loader.exec_module(agent_module)
                    
                    # Check for agent_main function
                    if not hasattr(agent_module, "agent_main"):
                        raise Exception("agent_main() function not found in agent.py")
                    
                    # Call agent_main
                    start_time = time.time()
                    patch = agent_module.agent_main(input_data)
                    elapsed_time = time.time() - start_time
                    
                    if elapsed_time > timeout:
                        console.print(f"⚠️  Agent exceeded timeout ({elapsed_time:.1f}s > {timeout}s)", style="yellow")
                    
                    # Validate return value
                    if not isinstance(patch, str):
                        raise Exception(f"agent_main() must return a string, got {type(patch)}")
                    
                finally:
                    # Restore stdout/stderr
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    agent_logs = captured_output.getvalue()
                
                # Save results before cleanup
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                problem_slug = problem.name.replace("/", "_").replace(" ", "_")
                results_dir = os.path.join(workspace_base, f"{timestamp}_{problem_slug}")
                os.makedirs(results_dir, exist_ok=True)
                
                # Save patch
                patch_file = os.path.join(results_dir, "patch.diff")
                with open(patch_file, "w") as f:
                    f.write(patch if patch else "")
                
                # Save logs
                logs_file = os.path.join(results_dir, "agent_logs.txt")
                with open(logs_file, "w") as f:
                    f.write(agent_logs)
                
                # Save problem statement
                problem_file = os.path.join(results_dir, "problem_statement.txt")
                with open(problem_file, "w") as f:
                    f.write(problem.problem_statement)
                
                # Copy the entire repo directory (contains test files and generated code)
                results_repo_dir = os.path.join(results_dir, "repo")
                if os.path.exists(repo_dir):
                    shutil.copytree(repo_dir, results_repo_dir)
                
                # Create summary JSON
                summary = {
                    "problem_name": problem.name,
                    "timestamp": timestamp,
                    "timeout_seconds": timeout,
                    "elapsed_time": elapsed_time,
                    "exceeded_timeout": elapsed_time > timeout,
                    "patch_generated": bool(patch),
                    "patch_length": len(patch) if patch else 0,
                    "agent_file": agent_file_path,
                    "include_solution": include_solution,
                    "workspace_dir": results_dir
                }
                summary_file = os.path.join(results_dir, "summary.json")
                with open(summary_file, "w") as f:
                    json.dump(summary, f, indent=2)
                
                # Create README explaining the structure
                readme_content = f"""# Test Results for {problem.name}

## Execution Summary
- **Timestamp**: {timestamp}
- **Problem**: {problem.name}
- **Agent**: {agent_file_path}
- **Timeout**: {timeout}s
- **Elapsed Time**: {elapsed_time:.2f}s
- **Exceeded Timeout**: {'Yes' if elapsed_time > timeout else 'No'}
- **Patch Generated**: {'Yes' if patch else 'No'}

## Directory Structure
- `patch.diff` - Generated solution patch
- `agent_logs.txt` - Complete agent execution logs (includes test results)
- `problem_statement.txt` - Original problem statement
- `repo/` - Complete repository state with generated code and test files
- `summary.json` - Machine-readable summary
- `README.md` - This file

## Test Results
Check `agent_logs.txt` for detailed test execution output including:
- Test cases executed
- Passed/failed test counts
- Error messages and stack traces

## Generated Code
All generated/modified files are in the `repo/` directory.
"""
                readme_file = os.path.join(results_dir, "README.md")
                with open(readme_file, "w") as f:
                    f.write(readme_content)
                
                console.print(f"💾 Saved results to: {results_dir}", style="green")
                
                return patch, agent_logs, results_dir
                
            finally:
                # Restore original directory and path
                os.chdir(original_cwd)
                if workspace_dir in sys.path:
                    sys.path.remove(workspace_dir)
        
        finally:
            # Cleanup temporary workspace
            if os.path.exists(workspace_dir):
                shutil.rmtree(workspace_dir)
                console.print(f"✓ Cleaned up temporary workspace", style="dim")
#
    # Only check gateway config if not running in local mode
    if not local:
        inference_env_path = Path("inference_gateway/.env")
        inference_env_example_path = Path("inference_gateway/.env.example")
#
        if not inference_env_path.exists():
            if inference_env_example_path.exists():
                console.print("📋 No inference_gateway/.env file found, copying from .env.example...", style="yellow")
                shutil.copy(inference_env_example_path, inference_env_path)
                console.print("✅ Created inference_gateway/.env from inference_gateway/.env.example", style="green")
            else:
                console.print("❌ No inference_gateway/.env.example file found! This is required for setup.", style="bold red")
                return
#
        # Check for required Chutes API key
        if os.path.exists("inference_gateway/.env"):
            with open("inference_gateway/.env", "r") as f:
                env_content = f.read()
#
            if "PROXY_CHUTES_API_KEY=" in env_content:
                # Check if it's just empty or still has placeholder
                import re
                api_key_match = re.search(r'PROXY_CHUTES_API_KEY=(.*)$', env_content, re.MULTILINE)
                if not api_key_match or not api_key_match.group(1).strip():
                    console.print("❌ PROXY_CHUTES_API_KEY is required in inference_gateway/.env", style="bold red")
                    console.print("   Please get your API key from https://chutes.ai and update inference_gateway/.env", style="yellow")
                    return
#
    # Load environment variables
    try:
        from dotenv import load_dotenv
        validator_env = Path("validator/.env")
        if validator_env.exists():
            load_dotenv(validator_env)
            console.print("Loaded configuration from validator/.env", style="green")
        else:
            console.print("No validator/.env found, using defaults", style="yellow")
    except ImportError as e:
        console.print(f"❌ Failed to load environment setup: {e}", style="bold red")
        return
#
    if verbose:
        from validator.utils.logger import enable_verbose
        enable_verbose()
        console.print("🔧 Verbose logging enabled", style="dim")
#
    mode_str = "Local Mode (No Docker)" if local else "Docker Mode"
    console.print(Panel(f"[bold cyan]🧪 Testing Agent[/bold cyan]\n"
                        f"[yellow]Mode:[/yellow] {mode_str}\n"
                        f"[yellow]Problem:[/yellow] {problem_name}\n"
                        f"[yellow]Agent:[/yellow] {agent_file}\n"
                        f"[yellow]Timeout:[/yellow] {timeout}s",
                        title=" Test Agent", border_style="cyan"))
#
    # Remind about inference gateway requirement
    if local:
        console.print(f"📡 Local mode requires inference gateway at http://localhost:8000", style="cyan")
        console.print(f"💡 Start it in another terminal: python -m inference_gateway.main", style="dim")
#
    # Validate agent file exists
    if not Path(agent_file).exists():
        console.print(f" Agent file not found: {agent_file}", style="bold red")
        return
#
    # Set gateway URL (only needed for Docker mode)
    if not local:
        local_ip = get_local_ip()
        if gateway_url:
            gateway_full_url = gateway_url
        else:
            gateway_full_url = f"http://{local_ip}:8000"
        gateway_url = f"http://0.0.0.0:8000"
        console.print(f"📡 Using inference gateway at: {gateway_full_url}", style="cyan")
        console.print(f"💡 Make sure to start it manually: python -m inference_gateway.main", style="dim")
        
        from evaluator.sandbox.sandbox_manager import SandboxManager
    
    from evaluator.problem_suites.problem_suite import ProblemSuite

    console.print(f"🔍 Searching for problem '{problem_name}' in all suites...", style="yellow")
    search_result = ProblemSuite.find_problem_in_suites(problem_name)
#
    if search_result is None:
        console.print(f" Problem '{problem_name}' not found in any suite", style="bold red")
        console.print("Available suites: polyglot, swebench_verified", style="yellow")
        return
#
    suite_name, suite = search_result
    console.print(f"✅ Found problem '{problem_name}' in '{suite_name}' suite", style="green")
#
    test_count = suite.get_problem_test_count(problem_name)
    if test_count > 150:
        console.print(f" Problem {problem_name} has {test_count} tests (>150)", style="bold red")
        return
#
    console.print(f"Problem {problem_name} has {test_count} tests", style="cyan")
#
    problem = suite.get_problem(problem_name)
    
    results_dir = None
    try:
        if local:
            # Run locally without Docker
            patch, agent_logs, results_dir = run_agent_locally(
                agent_file_path=agent_file,
                problem=problem,
                suite=suite,
                include_solution=include_solution,
                timeout=timeout,
                verbose=verbose
            )
        else:
            # Run with Docker
            sandbox_manager = SandboxManager(gateway_full_url)
            
            with open(agent_file, "r") as f:
                agent_source_code = f.read()
            
            run_id = uuid.uuid4()
            
            console.print(f"🚀 Creating agent sandbox...", style="yellow")
            
            # Initialize agent sandbox
            agent_sandbox = suite.initialize_agent_sandbox(
                sandbox_manager=sandbox_manager,
                problem=problem,
                evaluation_run_id=run_id,
                agent_code=agent_source_code,
                include_solution=include_solution
            )
            
            console.print(f"▶️  Running agent (timeout: {timeout}s)...", style="yellow")
            
            # Run the agent
            patch, agent_logs = suite.run_agent_sandbox(
                sandbox_manager=sandbox_manager,
                agent_sandbox=agent_sandbox,
                timeout_seconds=timeout
            )
        
        console.print("\n" + "=" * 80, style="cyan")
        console.print(" AGENT RESULTS", style="bold cyan")
        console.print("=" * 80 + "\n", style="cyan")
        
        if patch:
            console.print("📝 Generated patch:", style="bold green")
            print(patch)
        else:
            console.print("⚠️  No patch generated", style="yellow")
        
        if agent_logs:
            console.print("\n📋 Agent logs:", style="bold blue")
            print(agent_logs)
        
        console.print("\n✅ Agent execution completed successfully", style="bold green")
        
        if results_dir and local:
            console.print(f"\n📂 Full results saved to: {results_dir}", style="bold cyan")
            console.print(f"   - patch.diff - Generated solution", style="dim")
            console.print(f"   - agent_logs.txt - Complete logs with test results", style="dim")
            console.print(f"   - repo/ - All generated code and test files", style="dim")
            console.print(f"   - summary.json - Machine-readable summary", style="dim")
    
    except KeyboardInterrupt:
        console.print("\n🛑 Test interrupted by user", style="yellow")
    except Exception as e:
        console.print(f" Test failed: {e}", style="bold red")
        if verbose:
            console.print(traceback.format_exc(), style="dim")
    finally:
        if not local:
            console.print("🧹 Cleaning up...", style="dim")
            cleanup_tasks = []
#
            # Clean up Docker containers if cleanup flag is enabled
            if cleanup and 'sandbox_manager' in locals():
                try:
                    cleanup_tasks.append("sandboxes")
                    sandbox_manager.cleanup()
                    console.print("✓ Cleaned up sandboxes", style="dim")
                except Exception as e:
                    if verbose:
                        console.print(f"⚠️  Could not clean up containers: {e}", style="yellow")
#
            # Show what was cleaned up
            if cleanup_tasks:
                console.print(f"✅ Cleaned up {', '.join(cleanup_tasks)}", style="dim")
            else:
                console.print("✅ Nothing to clean up", style="dim")


if __name__ == "__main__":
    run_cmd(". .venv/bin/activate")
    cli() 