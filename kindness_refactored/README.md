# Kindness Refactored

A modular AI agent framework for code generation and fixing, refactored from the original `Kindness.py` file for better maintainability and organization.

## Structure

The refactored codebase is organized into the following modules:

### Core Modules

- **`__init__.py`** - Package initialization with exports
- **`constants.py`** - Configuration constants and model names
- **`prompts.py`** - All prompt templates and system messages
- **`network.py`** - LLM communication and network handling
- **`utils.py`** - Utility functions and helper classes
- **`visitors.py`** - AST visitors for code analysis

### Agent Components

- **`enhanced_cot.py`** - Chain of Thought management (updated with imports)
- **`enhanced_tool_manager.py`** - Base tool manager (updated with imports)
- **`fix_tool_manager.py`** - Specialized tool manager for fix tasks

### Workflows

- **`workflows.py`** - Main workflow functions and task processing
- **`main.py`** - Entry points and high-level functions

## Key Features

### 1. Modular Design
- Each component has a single responsibility
- Clear separation of concerns
- Easy to maintain and extend

### 2. Enhanced Tool Management
- Comprehensive tool framework
- Specialized tool managers for different task types
- Robust error handling and validation

### 3. Advanced Chain of Thought
- Sophisticated thought management
- Action tracking and repetition detection
- Context-aware conversation handling

### 4. Network Communication
- Robust LLM communication
- Retry logic and error handling
- Multiple model support

### 5. Workflow Management
- Task-specific workflows
- Fix and create task handling
- Comprehensive logging and monitoring

## Usage

### Basic Usage

```python
from kindness_refactored import main, fix_task, create_task

# For general tasks
result = main({"problem_statement": "Your problem here"})

# For fix tasks specifically
patch = fix_task({"problem_statement": "Fix this bug..."})

# For create tasks specifically
solution = create_task({"problem_statement": "Create this feature..."})
```

### Advanced Usage

```python
from kindness_refactored import (
    EnhancedCOT, EnhancedToolManager, FixTaskEnhancedToolManager,
    EnhancedNetwork, Utils
)

# Use individual components
cot = EnhancedCOT()
tool_manager = FixTaskEnhancedToolManager()
network = EnhancedNetwork()
```

## Configuration

The framework uses environment variables for configuration:

- `SANDBOX_PROXY_URL` - Proxy URL for LLM requests
- `AGENT_TIMEOUT` - Timeout for agent execution
- `REPO_PATH` - Path to repository being worked on
- `RUN_ID` - Unique identifier for the run

## Model Support

The framework supports multiple LLM models:

- GLM-4.5-FP8
- Kimi-K2-Instruct  
- DeepSeek-V3-0324
- Qwen3-Coder-480B-A35B-Instruct-FP8

## Error Handling

Comprehensive error handling with specific error types:

- Syntax errors
- Runtime errors
- Timeout errors
- File not found errors
- Network errors
- Invalid tool calls

## Logging

Detailed logging throughout the framework:

- Debug information
- Error tracking
- Performance monitoring
- Workflow execution tracking

## Benefits of Refactoring

1. **Maintainability** - Each module has a clear purpose
2. **Testability** - Individual components can be tested in isolation
3. **Extensibility** - Easy to add new features and tools
4. **Readability** - Code is organized and well-documented
5. **Reusability** - Components can be reused across different workflows

## Migration from Original

The refactored version maintains full compatibility with the original `Kindness.py` while providing:

- Better organization
- Improved error handling
- Enhanced logging
- Modular architecture
- Easier maintenance

All original functionality is preserved and can be accessed through the same interfaces.
