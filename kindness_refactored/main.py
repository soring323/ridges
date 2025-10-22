"""
Main entry points for the Kindness AI agent framework.
"""

import logging
import os
import sys
from typing import Any, Dict

from kindness_refactored.workflows import agent_main, process_fix_task, process_create_task
from kindness_refactored.utils import ensure_git_initialized, set_env_for_agent

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

for h in list(logger.handlers):
    logger.removeHandler(h)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def main(input_dict: Dict[str, Any], repo_dir: str = "repo"):
    """
    Main entry point for the Kindness AI agent.
    
    Args:
        input_dict: Dictionary containing problem statement and other parameters
        repo_dir: Directory containing the repository to work on
        
    Returns:
        Generated patch or solution
    """
    return agent_main(input_dict, repo_dir)


def fix_task(input_dict: Dict[str, Any]):
    """
    Entry point for fix tasks.
    
    Args:
        input_dict: Dictionary containing problem statement
        
    Returns:
        Generated patch for the fix
    """
    return process_fix_task(input_dict)


def create_task(input_dict: Dict[str, Any]):
    """
    Entry point for create tasks.
    
    Args:
        input_dict: Dictionary containing problem statement
        
    Returns:
        Generated solution for the create task
    """
    return process_create_task(input_dict)


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python -m kindness_refactored.main <problem_statement>")
        sys.exit(1)
    
    problem_statement = sys.argv[1]
    input_dict = {"problem_statement": problem_statement}
    
    result = main(input_dict)
    print(f"Result: {result}")
