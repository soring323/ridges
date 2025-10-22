"""
Kindness Refactored - A modular AI agent framework for code generation and fixing.

This package provides a comprehensive framework for AI agents that can:
- Generate code solutions for programming problems
- Fix bugs in existing codebases
- Run and validate tests
- Generate patches and solutions

Main Components:
- EnhancedCOT: Chain of Thought management
- EnhancedToolManager: Tool execution framework
- EnhancedNetwork: LLM communication
- Workflows: High-level task processing
"""

from __future__ import annotations

# Core classes
from kindness_refactored.enhanced_cot import EnhancedCOT
from kindness_refactored.enhanced_tool_manager import EnhancedToolManager
from kindness_refactored.network import EnhancedNetwork
from kindness_refactored.utils import Utils

# Specialized tool managers
from kindness_refactored.fix_tool_manager import FixTaskEnhancedToolManager

# Workflow functions
from kindness_refactored.workflows import (
    process_fix_task,
    process_create_task,
    agent_main,
    fix_task_solve_for_fixing_workflow,
    fix_task_solve_workflow
)

# Constants and configuration
from kindness_refactored.constants import (
    DEFAULT_PROXY_URL,
    DEFAULT_TIMEOUT,
    PROBLEM_TYPE_CREATE,
    PROBLEM_TYPE_FIX,
    GLM_MODEL_NAME,
    KIMI_MODEL_NAME,
    DEEPSEEK_MODEL_NAME,
    QWEN_MODEL_NAME,
    AGENT_MODELS,
    MAX_FIX_TASK_STEPS
)

# Utility functions
from kindness_refactored.utils import (
    ensure_git_initialized,
    set_env_for_agent,
    get_directory_tree,
    get_code_skeleton
)

__version__ = "2.0.0"
__author__ = "Kindness AI Team"

__all__ = [
    # Core classes
    "EnhancedCOT",
    "EnhancedToolManager", 
    "EnhancedNetwork",
    "Utils",
    "FixTaskEnhancedToolManager",
    
    # Workflow functions
    "process_fix_task",
    "process_create_task", 
    "agent_main",
    "fix_task_solve_for_fixing_workflow",
    "fix_task_solve_workflow",
    
    # Constants
    "DEFAULT_PROXY_URL",
    "DEFAULT_TIMEOUT",
    "PROBLEM_TYPE_CREATE",
    "PROBLEM_TYPE_FIX",
    "GLM_MODEL_NAME",
    "KIMI_MODEL_NAME", 
    "DEEPSEEK_MODEL_NAME",
    "QWEN_MODEL_NAME",
    "AGENT_MODELS",
    "MAX_FIX_TASK_STEPS",
    
    # Utility functions
    "ensure_git_initialized",
    "set_env_for_agent",
    "get_directory_tree",
    "get_code_skeleton"
]
