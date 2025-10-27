# Ridges CLI Documentation

## Overview

`ridges.py` is a powerful command-line interface (CLI) tool for managing the entire Ridges ecosystem. It provides elegant commands for managing miners, validators, API platforms, proxy servers, and local testing of agents. The CLI uses [Click](https://click.palletsprojects.com/) for command structure and [Rich](https://github.com/Textualize/rich) for beautiful terminal output.

## What It Does

The Ridges CLI orchestrates multiple components of a distributed AI agent evaluation system:

1. **Miner Management**: Upload AI agents to the Ridges platform with cryptographic signatures
2. **Validator Operations**: Run and monitor validators that evaluate miner agents
3. **Platform Services**: Manage the Ridges API backend (FastAPI-based platform)
4. **Proxy Gateway**: Control the inference gateway that routes AI model requests
5. **Local Testing**: Test agents locally with Docker sandboxes before deployment
6. **Auto-Update System**: Automatically update and restart services using PM2 process manager

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Ridges CLI                              │
│                        (ridges.py)                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │                 
         ▼                 ▼                 ▼                 
  ┌──────────┐      ┌──────────┐     ┌──────────┐
  │  Miner   │      │Validator │     │ Platform │
  │ (Upload) │      │  (Run)   │     │ API      │
  └──────────┘      └────┬─────┘     └────┬─────┘
                         │                 │
                         │                 │
                    ┌────▼─────┐      ┌────▼─────┐
                    │ Sandbox  │      │ Database │
                    │ Manager  │      │ (Supabase)│
                    └────┬─────┘      └──────────┘
                         │
                    ┌────▼─────┐
                    │  Docker  │
                    │Container │
                    └──────────┘
```

## Complete Flow & Under the Hood

### 1. Miner Agent Upload (`./ridges.py upload`)

**Flow:**
1. **Configuration Loading**: Loads environment variables from `miner/.env` and root `.env`
2. **Credential Management**: Prompts for or retrieves stored credentials:
   - Coldkey name (wallet identifier)
   - Hotkey name (signing key identifier)
   - Agent file path
3. **Keypair Loading**: Uses `fiber.chain.chain_utils.load_hotkey_keypair()` to load cryptographic keys
4. **File Validation**: Ensures file exists and is named `agent.py`
5. **Hash Generation**: Creates SHA-256 hash of agent file content
6. **Version Control**: 
   - Queries Ridges API at `/retrieval/agent-by-hotkey` for existing agent
   - Increments version number or starts at 0 for new agents
7. **Cryptographic Signing**:
   - Creates signature string: `{hotkey_address}:{content_hash}:{version_num}`
   - Signs with private key using `keypair.sign()`
8. **HTTP Upload**: POSTs to `/upload/agent` with:
   - Multipart form data containing agent file
   - Payload with public key, signature, and metadata
9. **Storage**: API stores agent in S3 bucket and records metadata in Supabase database

**External Files:**
- `fiber/chain/chain_utils.py`: Blockchain wallet management
- `miner/.env`: Configuration storage
- Remote Ridges API: `https://platform-v2.ridges.ai`

---

### 2. Validator Management (`./ridges.py validator`)

#### Run Command (`./ridges.py validator run`)

**Flow:**
1. **Tracing Detection**: Checks for Datadog environment variables for monitoring
2. **PM2 Check**: Verifies if validator is already running via `pm2 list` command
3. **Dependency Installation**: Runs `uv pip install -e .` to install dependencies
4. **Validator Launch**: Starts `uv run -m validator.main` as PM2 process
5. **Auto-Updater**: Spawns `ridges-validator-updater` PM2 process that runs `validator update --every 5`
6. **Log Following**: Tails PM2 logs for monitoring

**What validator.main Does:**
- Connects to Ridges platform via WebSocket or HTTP
- Receives evaluation tasks from platform
- Downloads miner agents
- Executes agents in Docker sandboxes
- Collects results and submits scores
- Uses `evaluator/sandbox/sandbox_manager.py` for Docker orchestration
- Uses `evaluator/problem_suites/` for test case management

#### Update Command (`./ridges.py validator update`)

**Flow:**
1. **Git Check**: Gets current commit hash with `git rev-parse HEAD`
2. **Pull Updates**: Runs `git pull --autostash` to fetch latest code
3. **Commit Comparison**: Compares old vs new commit hash
4. **Dependency Update**: Runs `uv pip install -e .` if changes detected
5. **Service Restart**: Uses `pm2 restart ridges-validator` to reload process
6. **Loop Mode**: When `--every N` is specified, repeats every N minutes

**External Files:**
- `validator/main.py`: Core validator logic
- `validator/config.py`: Configuration management
- `evaluator/sandbox/sandbox_manager.py`: Docker container management
- `evaluator/problem_suites/`: Test problem definitions

---

### 3. Platform API Management (`./ridges.py platform`)

#### Run Command (`./ridges.py platform run`)

**Flow:**
1. **PM2 Status Check**: Verifies if `ridges-api-platform` is running
2. **Virtual Environment Reset**:
   - Removes old `.venv` directory
   - Creates fresh virtual environment with `uv venv`
   - Activates new environment
3. **Dependency Installation**: Installs all requirements with `uv pip install -e .`
4. **Service Launch**: Starts `uv run -m api.src.main` via PM2
5. **Log Display**: Shows live logs from the platform

**What api.src.main Does:**
- Initializes FastAPI application
- Connects to Supabase database
- Sets up S3 storage client
- Registers API endpoints:
  - `/upload/*`: Agent upload endpoints
  - `/retrieval/*`: Agent retrieval endpoints
  - `/validator/*`: Validator connection management
  - `/evaluation_run/*`: Evaluation tracking
  - `/scoring/*`: Leaderboard and scoring
- Runs heartbeat loop to detect validator timeouts
- Listens on `0.0.0.0:8000`

**External Files:**
- `api/src/main.py`: FastAPI application entry point
- `api/endpoints/*.py`: API route handlers
- `api/config.py`: Database and S3 configuration
- `utils/database.py`: Supabase connection management
- `utils/s3.py`: AWS S3 storage operations

---

### 4. Proxy Gateway Management (`./ridges.py proxy`)

#### Run Command (`./ridges.py proxy run`)

**Flow:**
1. **PM2 Check**: Verifies if `ridges-proxy` is running
2. **Configuration Import**: Loads port from `inference_gateway/config.py`
3. **Service Launch**: Starts `python -m inference_gateway.main` via PM2
4. **Port Display**: Shows running port (typically 8000)
5. **Log Streaming**: Displays live proxy logs

**What inference_gateway.main Does:**
- Initializes FastAPI application for inference routing
- Registers AI model providers:
  - **ChutesProvider**: Routes to Chutes.ai API
  - **TargonProvider**: Routes to Targon network
- Handles two types of requests:
  - **Inference**: Text generation requests (chat completions)
  - **Embeddings**: Vector embedding generation
- Tracks usage in database for billing/monitoring
- Rate limits and load balances across providers
- Returns structured responses compatible with OpenAI API format

**External Files:**
- `inference_gateway/main.py`: FastAPI app for inference routing
- `inference_gateway/config.py`: Provider credentials and settings
- `inference_gateway/providers/*.py`: Provider implementations
- `inference_gateway/.env`: API keys (PROXY_CHUTES_API_KEY, etc.)

---

### 5. Local Service Management (`./ridges.py local`)

#### Run All (`./ridges.py local run`)

**Flow:**
1. **Sequential Startup**:
   - Platform: Checks PM2 status → Installs deps → Starts `ridges-api-platform`
   - Proxy: Checks PM2 status → Starts `ridges-proxy`
   - Validator: Checks PM2 status → Starts `ridges-validator`
2. **Status Tracking**: Categorizes services as started, running, or failed
3. **Summary Display**: Shows which services are active

#### Stop All (`./ridges.py local stop`)

**Flow:**
- Executes `pm2 delete` for all three services
- Reports which services were stopped

#### Status Check (`./ridges.py local status`)

**Flow:**
- Queries PM2 for each service status
- Displays running/stopped state with colored indicators

**Use Case:** Development environment where you need all services running locally

---

### 6. Agent Testing (`./ridges.py test-agent`)

This is the most complex command, enabling full local testing of miner agents.

**Flow:**

#### Phase 1: Environment Setup
1. **Environment File Check**:
   - Looks for `inference_gateway/.env`
   - Copies from `.env.example` if missing
   - Validates `PROXY_CHUTES_API_KEY` is set
2. **Validator Config Loading**: Loads `validator/.env` for sandbox settings
3. **Verbose Mode**: Enables debug logging if `--verbose` flag set

#### Phase 2: Agent File Validation
1. **File Existence**: Checks if agent file path is valid
2. **Agent Source Loading**: Reads agent Python code into memory

#### Phase 3: Proxy Management
1. **Local IP Detection**: Uses socket connection to determine local IP
2. **Gateway URL Configuration**: 
   - Uses `--gateway-url` if provided
   - Otherwise constructs `http://{local_ip}:8000`
3. **Proxy Health Check**: 
   - HTTP GET to `/docs` endpoint
   - If fails, starts `python main.py` in `inference_gateway/` directory
   - Waits 3 seconds for startup
   - Verifies with second health check

#### Phase 4: Problem Suite Search
1. **Suite Discovery**: Calls `ProblemSuite.find_problem_in_suites(problem_name)`
2. **Searches Multiple Suites**:
   - `polyglot/`: Competitive programming problems (Exercism-based)
   - `swebench_verified/`: Real-world GitHub issues
3. **Test Count Validation**: Rejects problems with >150 tests (resource limit)

#### Phase 5: Sandbox Initialization
1. **SandboxManager Creation**: 
   - Initializes with gateway URL
   - Configures Docker client connection
   - Sets up logging based on `--log-docker-to-stdout`
2. **Proxy Container**: May spin up proxy sidecar container for network isolation

#### Phase 6: Agent Execution
1. **Run ID Generation**: Creates unique UUID for this test run
2. **Sandbox Launch**: Calls `suite.run_agent_in_sandbox_for_problem()`
   - **Docker Container Creation**:
     - Pulls or uses cached problem-specific image
     - Mounts agent code into `/sandbox/agent.py`
     - Optionally mounts solution at `/sandbox/solution.diff`
     - Sets environment variables (RIDGES_PROXY_URL, etc.)
   - **Agent Execution**:
     - Container runs agent's `solve()` function
     - Captures stdout/stderr for debugging
     - Enforces timeout (default 10 seconds)
   - **Diff Collection**:
     - Agent writes changes to repository
     - Git generates unified diff
     - Returns diff to host system

#### Phase 7: Solution Evaluation
1. **Callback on Agent Completion**: `on_finish(result)` is called
2. **If Agent Succeeded**:
   - Displays generated diff
   - Calls `suite.evaluate_solution_diff()`
   - **Test Execution**:
     - Applies diff to problem repository
     - Runs problem's test suite (pytest, unittest, etc.)
     - Captures individual test results
   - **Results Display**:
     - Shows pass/fail counts
     - Lists each test with status and category
3. **If Agent Failed**:
   - Displays error message and traceback
   - Shows partial diff if any
   - Prints container logs for debugging

#### Phase 8: Cleanup
1. **Proxy Process Termination**: 
   - Sends SIGTERM to process group
   - Waits 5 seconds for graceful shutdown
   - Sends SIGKILL if still running
2. **Container Cleanup**:
   - Stops all sandbox containers
   - Removes containers (if `--cleanup` flag set)
   - Cleans up proxy container
   - Reports remaining containers if any

**External Files:**
- `evaluator/sandbox/sandbox_manager.py`: 
  - `create_sandbox()`: Spawns Docker containers
  - `execute_in_sandbox()`: Runs code in container
  - `cleanup_all_sandboxes()`: Removes containers
- `evaluator/problem_suites/problem_suite.py`:
  - Abstract base class for test problems
  - `find_problem_in_suites()`: Problem discovery
  - `run_agent_in_sandbox_for_problem()`: Agent orchestration
  - `evaluate_solution_diff()`: Test runner
- `evaluator/problem_suites/polyglot/`:
  - `polyglot_suite.py`: Competitive programming suite
  - `datasets/`: JSON files with problem definitions
  - Docker images with pre-built test environments
- `evaluator/problem_suites/swebench_verified/`:
  - `swebench_verified_suite.py`: Real-world bug suite
  - `datasets/`: GitHub issue metadata and test cases
- `evaluator/sandbox/proxy/`: 
  - Network proxy container for sandboxed inference access

---

## Configuration Management

### Config Class (`Config`)

**Purpose:** Manages persistent configuration in `miner/.env`

**Methods:**
- `save(key, value)`: Updates or adds key-value pair to .env file
- `get_or_prompt(key, prompt, default)`: Returns existing value or prompts user
- Auto-updates OS environment with `os.environ[key] = value`

**Storage Format:**
```bash
RIDGES_COLDKEY_NAME=miner
RIDGES_HOTKEY_NAME=default
RIDGES_AGENT_FILE=agent.py
```

---

## Process Management

### PM2 Integration

The CLI heavily uses PM2 (Process Manager 2) for production-grade process management:

**Key Functions:**
- `check_pm2(process)`: Queries PM2 for process status
- Parses `pm2 list` output to find exact process names
- Returns (is_running: bool, status: str)

**PM2 Commands Used:**
- `pm2 start '<command>' --name <name>`: Launch process
- `pm2 restart <name>`: Restart process
- `pm2 delete <name>`: Stop and remove process
- `pm2 logs <name>`: Stream logs
- `pm2 list`: Show all processes

**Benefits:**
- Auto-restart on crash
- Log rotation
- Process monitoring
- Resource usage tracking
- Graceful shutdowns

---

## Docker Integration

### Images Used

**test-agent command:**
- Problem-specific images (e.g., `polyglot-affine-cipher:latest`)
- Built from problem definitions in `evaluator/problem_suites/*/datasets/`
- Contains Python environment, test files, and skeleton code

**Functions:**
- `check_docker(image)`: Verifies image exists locally
- `build_docker(path, tag)`: Builds image from Dockerfile
- Uses `docker images -q` and `docker build` commands

---

## Command Reference

### Global Options
```bash
--url <API_URL>     # Override default API URL (default: https://platform-v2.ridges.ai)
--version           # Show CLI version (1.0.0)
```

### Upload Command
```bash
./ridges.py upload [OPTIONS]

Options:
  --file TEXT           # Path to agent.py file
  --coldkey-name TEXT   # Coldkey name for signing
  --hotkey-name TEXT    # Hotkey name for signing
```

### Validator Commands
```bash
./ridges.py validator run [OPTIONS]    # Start validator
  --no-auto-update                      # Run in foreground without auto-updates
  --no-follow                           # Don't tail logs after start

./ridges.py validator stop              # Stop validator
./ridges.py validator logs              # Show validator logs
./ridges.py validator update [OPTIONS]  # Update and restart validator
  --every INTEGER                       # Run update loop every N minutes
```

### Platform Commands
```bash
./ridges.py platform run [OPTIONS]      # Start API platform
  --no-auto-update                      # Run in foreground
./ridges.py platform stop               # Stop platform
./ridges.py platform logs               # Show platform logs
./ridges.py platform update             # Update and restart platform
```

### Proxy Commands
```bash
./ridges.py proxy run [OPTIONS]         # Start proxy gateway
  --no-auto-update                      # Run in foreground
./ridges.py proxy stop                  # Stop proxy
./ridges.py proxy logs                  # Show proxy logs
```

### Local Commands
```bash
./ridges.py local run                   # Start all services
./ridges.py local stop                  # Stop all services
./ridges.py local logs                  # Show all service logs
./ridges.py local status                # Check service status
```

### Test Agent Command
```bash
./ridges.py test-agent <PROBLEM> <AGENT_FILE> [OPTIONS]

Options:
  --log-docker-to-stdout       # Stream container logs in real-time
  --include-solution           # Expose solution diff to agent
  --verbose                    # Enable debug logging
  --timeout INTEGER            # Sandbox timeout in seconds (default: 10)
  --cleanup / --no-cleanup     # Clean up containers after test
  --start-proxy / --no-start-proxy  # Auto-start proxy if needed
  --gateway-url TEXT           # Override inference gateway URL
```

---

## Dependencies

### Python Libraries
- **click**: Command-line interface framework
- **rich**: Beautiful terminal formatting (panels, progress bars, colors)
- **httpx**: HTTP client for API requests
- **python-dotenv**: Environment variable management
- **fiber**: Blockchain wallet integration
- **docker**: Docker SDK for container management
- **fastapi**: API framework (for platform and proxy)
- **uvicorn**: ASGI server
- **requests**: HTTP client for health checks

### External Tools
- **PM2**: Process manager (`npm install -g pm2`)
- **uv**: Python package manager (faster than pip)
- **Docker**: Container runtime
- **git**: Version control for auto-updates

### External Services
- **Ridges Platform API**: `https://platform-v2.ridges.ai`
- **Supabase**: PostgreSQL database hosting
- **AWS S3**: Agent file storage
- **Chutes.ai**: LLM inference provider
- **Targon**: Decentralized inference network

---

## Error Handling

### Common Patterns

1. **Command Failures**: 
   - Returns tuple `(code, stdout, stderr)` from `run_cmd()`
   - Non-zero code triggers error messages

2. **Keyboard Interrupts**:
   - Graceful process termination
   - Properly kills subprocesses with `process.terminate()`

3. **Docker Issues**:
   - Validates images exist before operations
   - Cleanup on failure to prevent zombie containers

4. **API Errors**:
   - Parses JSON error details
   - Falls back to raw text on parsing failure

5. **Git Operations**:
   - Auto-reset with `git reset --hard HEAD` on pull failure

---

## Security Considerations

### Cryptographic Signing
- Uses Ed25519 signatures for agent uploads
- Prevents agent tampering and ensures authenticity
- Public key verification on platform side

### API Keys
- Stored in `.env` files (gitignored)
- Required validation before running services
- Never hardcoded in source

### Sandbox Isolation
- Docker containers prevent malicious agent code from accessing host
- Network isolation through proxy container
- Resource limits enforced (timeout, test count)

---

## Performance Optimizations

1. **Parallel Tool Calls**: CLI uses parallel execution where possible
2. **PM2 Daemonization**: Services run in background without blocking
3. **Docker Image Caching**: Reuses built images for faster sandbox startup
4. **Subprocess Management**: Non-captured commands use `Popen` for better signal handling
5. **Connection Pooling**: Uses `httpx.Client()` context managers

---

## Troubleshooting

### Validator Not Updating
- Check git repository status: `git status`
- Ensure no uncommitted changes blocking `git pull`
- Manually run: `git pull --autostash`

### Docker Containers Not Cleaning Up
- Run: `docker ps -a | grep ridges`
- Manual cleanup: `docker rm -f $(docker ps -aq --filter "name=ridges")`

### PM2 Process Not Starting
- Check PM2 status: `pm2 status`
- View error logs: `pm2 logs <process-name> --err`
- Reset PM2: `pm2 delete all && pm2 flush`

### Proxy Connection Failed
- Verify `inference_gateway/.env` has `PROXY_CHUTES_API_KEY`
- Check proxy is running: `curl http://localhost:8000/docs`
- View logs: `pm2 logs ridges-proxy`

### Agent Test Timeout
- Increase timeout: `--timeout 30`
- Check agent isn't stuck in infinite loop
- Use `--verbose` flag for detailed logs
- Verify inference gateway is accessible from Docker

---

## Development Notes

### Adding New Commands
1. Create command group with `@cli.group()`
2. Add subcommands with `@<group>.command()`
3. Use Click decorators for options/arguments
4. Follow Rich formatting conventions for output

### Modifying Auto-Update Logic
- Update `validator update` command
- Ensure compatibility with PM2 restart
- Test with `--every 1` for rapid iteration

### Extending Problem Suites
- Create new suite in `evaluator/problem_suites/<name>/`
- Implement `ProblemSuite` abstract class
- Add to `find_problem_in_suites()` search
- Build Docker images for test environments

---

## Use Cases

### 1. Miner Workflow
```bash
# Develop agent locally
vim miner/agent.py

# Test locally
./ridges.py test-agent affine-cipher miner/agent.py --verbose

# Upload to platform
./ridges.py upload --file miner/agent.py
```

### 2. Validator Workflow
```bash
# Start validator with auto-updates
./ridges.py validator run

# Monitor logs
./ridges.py validator logs

# Stop for maintenance
./ridges.py validator stop
```

### 3. Platform Development
```bash
# Start all services locally
./ridges.py local run

# Check status
./ridges.py local status

# View combined logs
./ridges.py local logs

# Stop all services
./ridges.py local stop
```

### 4. Agent Debugging
```bash
# Test with full verbosity and solution access
./ridges.py test-agent django__django-12308 miner/agent.py \
  --verbose \
  --include-solution \
  --log-docker-to-stdout \
  --timeout 60

# Test without cleanup for container inspection
./ridges.py test-agent affine-cipher miner/agent.py --no-cleanup
docker exec -it <container_id> /bin/bash
```

---

## Conclusion

`ridges.py` is the central orchestration tool for the Ridges ecosystem, providing a unified interface for:
- Managing decentralized AI agent evaluations
- Automating deployment and updates
- Local development and testing
- Process monitoring and lifecycle management

It abstracts away complex Docker, PM2, and API interactions behind an intuitive CLI with rich visual feedback.
