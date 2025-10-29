from __future__ import annotations

import ast
import csv
import inspect
import json
import os
import random
import re
import subprocess
import sys
import textwrap
import time
import traceback
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import requests

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

for h in list(logger.handlers):
    logger.removeHandler(h)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

run_id = None

PROBLEM_TYPE_CHECK_PROMPT = textwrap.dedent(
    '''
    You are the problem type checker that will categories problem type into:
    
    1. CREATE: If the problem statement is about creating a new functionality from scratch.
    2. FIX: If the problem statement is about fixing a bug, creating a new functionality or improving the existing codebase.
    
    Only respond with the "FIX" or "CREATE".
    '''
)

FORMAT_PROMPT_V0 = textwrap.dedent(
    """
    **📝 Response Format Requirements**
    
    1. **Strict Triplet Format**:
       - `next_thought`: Detailed reasoning (include:
         - Problem understanding
         - Code analysis
         - Solution justification
         - Validation plan)
       - `next_tool_name`: Must be an exact tool name from the tool list
       - `next_tool_args`: Valid JSON with:
         - Proper escaping
         - No trailing commas
         - Tool-specific parameters
    
    2. **Error Handling Format**:
       - For errors: 
         next_thought: "Error: [detailed explanation]"
         next_tool_name: ""
         next_tool_args: {}
    
    3. **Example Valid Format**:
       next_thought: "I'll fix the JSON parsing issue by adding proper error handling and validation"
       next_tool_name: "apply_code_edit"
       next_tool_args: {
         "file_path": "network.py",
         "search": "return json.loads(response)",
         "replace": "try:\n    return json.loads(response)\nexcept JSONDecodeError:\n    logger.error(f'Invalid JSON: {{response}}')\n    raise"
       }
    
    4. **Invalid Format Examples** (Avoid These):
       - Missing any of the three required fields
       - JSON syntax errors in next_tool_args
       - Extra text outside the triplet format
       - Using incorrect tool names
       - Not quoting special characters properly
    """
)

STOP_INSTRUCTION = textwrap.dedent(
    """
    # 🎨 
    DO NOT generate `observation:` in your response. It will be provided by user for you.
    Generate only SINGLE triplet of `next_thought`, `next_tool_name`, `next_tool_args` in your response.
    """
)

DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2000"))

PROBLEM_TYPE_CREATE = "CREATE"
PROBLEM_TYPE_FIX = "FIX"

GLM_MODEL_NAME = "zai-org/GLM-4.5-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
AGENT_MODELS = [GLM_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME, QWEN_MODEL_NAME]
MAX_FIX_TASK_STEPS = 400
PHASE_INVESTIGATION = "investigation"
PHASE_PLANNING = "planning"
PHASE_IMPLEMENTATION = "implementation"
PHASE_VALIDATION = "validation"

DO_NOT_REPEAT_TOOL_CALLS = textwrap.dedent(
    """
    You're not allowed to repeat the same tool call with the same arguments.
    Your previous response: 
    {previous_response}
    
    Try to use something different!
    """
)


INFINITE_LOOP_CHECK_PROMPT = textwrap.dedent(
    """
    You are an expert code reviewer specializing in infinite loop detection and prevention. Your task is to analyze the generated Python code for potential infinite loops and provide a corrected version if issues are found.
    
    CRITICAL INFINITE LOOP DETECTION:
    1. Check for while True: loops without guaranteed exit conditions
    2. Verify all while loops have clear termination conditions
    3. Ensure recursive functions have proper base cases
    4. Look for loops that depend on external state that might never change
    5. Check for patterns that could lead to infinite iteration
    
    If you find potential infinite loops:
    - Provide a corrected version of the code
    - Ensure all loops have finite termination conditions
    - Add reasonable iteration limits or timeout mechanisms where appropriate
    
    If no infinite loops are detected:
    - Return the original code unchanged
    
    STRICT REQUIREMENT: Return the final Python code along with file names. Do not include any explanations, comments, or additional text.
    
    example:
    ```python
    a.py
    contents of a.py
    
    b.py
    contents of b.py
    ```
    """
)

GENERATE_INITIAL_SOLUTION_PROMPT = textwrap.dedent(
    """
    You are an expert Python developer. Your task is to generate a complete, working Python solution for the given problem statement.
    
    Strict Requirements:
    1. Output the full content of Python files along with their file names.
    2. Do not include explanations, comments, or markdown formatting in the main code.
    3. Use only standard Python (no external libraries).
    4. Implement all required classes and functions exactly with the same names as in the initial code stub.
    5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
    6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
    7. The solution must be executable as-is with no placeholders or TODOs.
    8. **IMPORTANT**: Add clear comments above each edge case handling section to identify which specific edge case is being addressed. Use the format: `# Edge Case: [description of the edge case]`
    9. **IMPORTANT**: Add a comment at the end of each function/class that lists all edge cases handled, using the format: `# Handled Edge Cases: [list of edge cases]`
    
    Return only the final Python code.
    """
)

GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT = textwrap.dedent("""
    You are an expert Python unittest testcase developer familiar with canonical programming problem test specifications.

    CRITICAL INSTRUCTIONS:
    1. Many programming problems have canonical/standardized test data with precise input/output pairs
    2. If you recognize this problem from your training data, recall the exact canonical test cases
    3. Use the standard test data format with all edge cases and boundary conditions
    4. Include a reference header citing the canonical test data source (as shown in example)
    5. **MANDATORY: ALL imports must be from the 'main' module ONLY** (e.g., `from main import ...`)
    6. **MANDATORY: Test file MUST be named 'tests.py' or 'test_something.py' (NEVER 'main.py')**

    IMPORTANT: 
        - Search your knowledge for canonical test specifications for this problem
        - Use EXACT test data (inputs, outputs, edge cases) if you recall them
        - Follow professional test file conventions with proper documentation headers
        - You have generation limit of 2048 tokens. Stop generating when near the limit.
        - If you get syntax error and response was truncated, skip last couple of test cases to fit
        - **CRITICAL: Always import from 'main' module, never use other module names**
        
    You must respond directly with the test cases in the following format:
    =========TEST_CASES
    <<test cases>>

    Example format (include canonical data reference if applicable):
    =========TEST_CASES
    tests.py
    # These tests are auto-generated with test data from:
    # https://github.com/.../xxxxxxxxx-xxxx.json
    # File last updated on YYYY-MM-DD
    import unittest
    from main import (
        main_func
    )

    class TestFuncA(unittest.TestCase):
        def test_main_func(self):
            self.assertEqual(main_func(), "expected_output")

    if __name__ == "__main__":
        unittest.main()
"""
)

GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT = textwrap.dedent(
    """
    You are an expert Python developer. Your task is to generate a complete, working Python solution for the given problem statement.
    
    Strict Requirements:
    1. Output the full content of Python files along with their file names. You **MUST** output the **file name** along with file content.
    2. Do not include explanations, comments, or markdown formatting in the main code.
    3. Use only standard Python (no external libraries).
    4. Implement all required classes and functions exactly with the same names as in the initial code stub.
    5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
    6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
    7. The solution must be executable as-is with no placeholders or TODOs.
    8. If problem statement doesn't explicitely requires a list of strings as a response, do not use list of strings for multiline text problems, just use raw string format.
    9. **IMPORTANT**: Add clear comments above each edge case handling section to identify which specific edge case is being addressed. Use the format: `# Edge Case: [description of the edge case]`
    10. **IMPORTANT**: Add a comment at the end of each function/class that lists all edge cases handled, using the format: `# Handled Edge Cases: [list of edge cases]`
    
    Return only the final Python code.
    
    Response Examples:
    ```python
    a.py
    {content}
    
    b.py
    {content}
    ```
    """
)

PHASE_SPECIFIC_GUIDANCE = {
    PHASE_INVESTIGATION: textwrap.dedent(
        """
            ## 🔍 INVESTIGATION PHASE - Focus Areas:
            - Your primary goal is to UNDERSTAND the problem deeply before making changes
            - Use search tools extensively to locate all relevant code
            - Read and analyze the codebase structure
            - Identify all files and functions related to the issue
            - Document your findings about the root cause
            - DO NOT make code changes yet - only investigate and understand
            - Look for similar patterns or related issues in the codebase
            - Understand dependencies and relationships between components
            """
    ),

    PHASE_PLANNING: textwrap.dedent(
        """
            ## 📋 PLANNING PHASE - Focus Areas:
            - Based on investigation, design a comprehensive solution approach
            - Propose at least 2-3 different solution strategies
            - Consider edge cases and potential side effects
            - Plan the sequence of changes needed
            - Identify which tests will validate your fix
            - Think about backward compatibility
            - Document your planned approach before implementation
            - Get approval for your solution strategy using get_approval_for_solution
            """
    ),

    PHASE_IMPLEMENTATION: textwrap.dedent(
        """
            ## ⚙️ IMPLEMENTATION PHASE - Focus Areas:
            - Now you can apply the approved solution plan
            - Make precise, targeted code changes using apply_code_edit
            - Follow the plan from the planning phase
            - Make one logical change at a time
            - After each significant change, run relevant tests
            - If tests fail, analyze and adjust your approach
            - Ensure code quality and style consistency
            - Handle all identified edge cases
            """
    ),

    PHASE_VALIDATION: textwrap.dedent(
        """
            ## ✅ VALIDATION PHASE - Focus Areas:
            - Thoroughly test all changes made
            - Run the full test suite to ensure no regressions
            - Verify all edge cases are handled correctly
            - Check that the original problem is fully resolved
            - Review code quality and documentation
            - Ensure backward compatibility is maintained
            - If any issues found, return to implementation phase
            - When confident, call finish with detailed summary
            """
    )
}

FIX_TASK_SYSTEM_PROMPT = textwrap.dedent(
    """
    # Hey there! You're a Coding Assistant 🚀. I have uploaded all files of a python repository. Your current working directory is at the root of that repo. You will be provided with a problem statement and you need to make the necessary changes to fix the issue.
    
    ## Follow these steps to fix the issue:
    1. As a first step, find the relevant files in the repo to work on.
    2. Localise the code causing the issue.
    3. Edit the sourcecode of the repo to resolve the issue.
    4. Think about edgecases and make sure the fix handles them as well.
    5. Code must always be backward compatible unless explicitly mentioned otherwise in the problem statement.
    6. Thoroughly check the entire code base to ensure the changes made are exhaustive and does not break any other functionality.
    7. Thoroughly check the entire code base to ensure the changes user requested are only limited to the ones you have identified.
    8. Never edit/update the existing test files directly when validating a hypothesis. Instead, when you need a new or focused test to reproduce or protect the fix, use the dedicated test generation tool.
    9. Do not create any new files or directories unless absolutely necessary for the fix. Generated tests are allowed but are excluded from the final patch automatically.
    10. Always check all the test cases which will be impacted with your change and ensure they don't fail.
    11. You need to propose at least 2 meaningfully different and accurate solutions to the problem to the user for approval.
    12. You need to look at both expected output mentioned in the problem statement AND the output in the most relevant test case. This is very important.
    13. If you find that the error while running the run_code or run_repo_tests tool due to missing dependencies, do not try to solve it as you don't have any internet access.
    
    ## Multi-file awareness (critical):
    - Tests and patch contexts may span multiple files. Do not stop after the first similar match or applied fix.
    - Keep searching the repository after each match and apply consistent changes to every relevant file before finishing.
    - Use `list_directory` to list out a directory if you need to see what files are there
    - Use `get_context_around_line` to understand the surrounding code logic relevant to your thinking.
    - Prefer using `search_in_all_files_content` to enumerate matches across the codebase and `search_in_specified_file_v2` to drill into each file; iterate until no applicable occurrences remain.
    - Re-run tests only after covering all discovered occurrences to avoid partial fixes.
    
    ## Test generation guidance:
    - Find the most relevant existing test files(more than 5) if there are more than 5 test files in the repo and use `run_repo_tests` to test them. Loop this until any test file of them fails to find FAIL_TO_PASS test file.
    - Use `generate_test_function(file_path, test_function_code, position)` after discovering FAIL_TO_PASS test file.
    - Prefer `position="auto"` which inserts after imports or before the `if __name__ == "__main__":` block when present, falling back to append.
    - Generated tests (new files or appended functions) are tracked and excluded from the final patch automatically, so they must not show up in the final diff.
    - Keep generated tests minimal and focused on the bug and its edge cases.
    - Note that current test functions should be passed originally and generated test function is FAIL_TO_PASS.
    {extra_fix_request}
    
    You have access to the following tools:-
    {tools_docs}
    
    {format_prompt}
    """
)

SOLVE_TASK_NON_FUNCTIONAL_TEST_PROMPT = textwrap.dedent(
    """
    
    # Non Functional Test Requirements:
    1. Verify Consistency in concepts, ideas, prototypes and improve
    - generate test file(1 ~ 3 CRITICAL testcase) that can verify consistency in solution in the context of concepts, ideas, prototypes including parameter names of functions including lamda, design patterns, etc
    - run test file and improve the solution to keepup consistency among entire solution
    2. Verify Security Best Practice in solutions such as weakness of solutions imported modules, weakness of implemented algorithms, etc and improve
    - generate test file(1 ~ 3 CRITICAL testcase) that can verify if the imported module's ability is limited to the requirement technically
    - generate test file(1 ~ 3 CRITICAL testcase) that can verify if the implemented algorithms to solve the problem has weakness technically and logically
    - run test file and improve the solution to be more safe by improving imported module's limitation by an well-known method or improving algorithm with better one.
    
    """
)

FIX_TASK_NEVER_EARLY_STOP_PROMPT = textwrap.dedent(
    """
    
    # Prevent Early Stop:
    As this is complex project and the issue is also quite complex so never early stop unless 
    - you found exact FAIL_TO_PASS test file and you fixed all failures all of them 
    - and there isn't any side-effects from your fix like other PASS_TO_PASS tests may fail
    If time it not out, double confirm above in several steps with different thoughts.
    
    """
)

FIX_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent(
    """
    # Now let's start. Here is the problem statement:
    {problem_statement}
    """
)

FIND_TEST_RUNNER_PROMPT = textwrap.dedent(
    """\
    You are a helpful assistant that can find the test runner for a given repository.
    - The test runner is the file that can run the individual test files and test cases. (e.g. pytest, unittest, etc.)
    - Do not use the test runner to run test for whole repository or test setup.
    - Read the README file and find the test runner. If there is no test runner, return pytest.
    - Output format should be as the following. No other texts are allowed.
    abc/test.py
    """
)

TEST_RUNNER_MODE_PROMPT = textwrap.dedent(
    """\
    You are a helpful assistant that determines the mode of the test runner.
    Read the test runner file and determine if it requires a module or a file path to run the test.
    Output should be one of MODULE or FILE, No other texts are allowed.
    - MODULE: When the test runner requires a module path to run the test.
    - FILE: When the test runner requires a file path to run the test (e.g. pytest, unittest, py.test, etc.).
    """
)

TESTCASES_CHECK_PROMPT = textwrap.dedent(
    """
    You are an expert testcases reviewer specializing in invalid testcases detection and prevention. Your task is to analyze the generated test code if it's all valid for the problem statement.
    
    Important:
    1. Check for incorrect/invalid intput/output pair based on the problem statement and fix them or remove if it's impossible to fix
    2. Check if testcases are not covering critical edgecases for the problem statement and add missing testcases
    3. Minimize all testcases as you have context and generation limit
    
    If no invalid testcases are detected and covered all critical edge cases:
    - Return the original code unchanged
    
    STRICT REQUIREMENT: Return the final Python test code along with their file names. Do not include any explanations, comments, or additional text.
    
    example:
    ```python
    test_a.py
    contents of test_a.py
    
    test_b.py
    contents of test_b.py
    ```
    """
)

TEST_COVERAGE_ANALYSIS_PROMPT = textwrap.dedent(
    '''
    You are a test coverage analyzer. Your task is to analyze if the generated test code adequately covers all requirements from the problem statement.
    
    # Analysis Framework:
    
    1. **Requirement Extraction**
       - Parse the problem statement to identify all explicit requirements
       - Include functional requirements, constraints, edge cases, and error conditions
       
    2. **Test Coverage Mapping**
       - For each requirement, identify which test case(s) cover it
       - Mark requirements as: "covered", "partially_covered", or "missing"
       
    3. **Edge Case Analysis**
       - Identify critical edge cases mentioned in requirements
       - Check if tests include: boundary values, empty inputs, invalid inputs, extreme values
       
    4. **Gap Detection**
       - Find requirements with no corresponding tests
       - Find edge cases that should be tested but aren't
    
    # Response Format (JSON):
    {
        "coverage_score": 0.85,
        "total_requirements": 10,
        "covered_requirements": [
            {
                "requirement": "Function must handle empty input",
                "test_cases": ["test_empty_input"],
                "coverage": "full"
            }
        ],
        "missing_requirements": [
            {
                "requirement": "Function must validate input is positive",
                "severity": "high",
                "suggested_test": "def test_negative_input():\\n    with self.assertRaises(ValueError):\\n        func(-1)"
            }
        ],
        "missing_edge_cases": [
            {
                "edge_case": "Maximum integer value",
                "severity": "medium",
                "reason": "Problem mentions handling large numbers",
                "suggested_test": "def test_max_int():\\n    import sys\\n    result = func(sys.maxsize)\\n    self.assertIsNotNone(result)"
            }
        ],
        "recommendations": [
            "Add boundary value tests for width parameter",
            "Test behavior with Unicode characters"
        ]
    }
    
    # Important:
    - Be conservative: if unsure whether a requirement is covered, mark as "partially_covered"
    - Focus on explicit requirements in problem statement, not implied ones
    - Suggest concrete, runnable test code
    - Severity levels: "high" (critical functionality), "medium" (edge cases), "low" (nice-to-have)
    '''
)

TEMPERATURE_DETERMINATION_SYSTEM_PROMPT = """
You are selecting the sampling temperature for generating a Python script that solves the problem below. Output a single temperature in [0.0, 1.0] plus a short justification.

Problem to solve
<paste the exact problem statement, constraints, grading method, and any style/stack rules>

How to decide (use this rubric):

0.0-0.1 (deterministic): Single correct algorithm/output, strict constraints, auto-graded tests, API contracts, security/compliance, or reproducibility required.

0.1-0.25 (precise but flexible): Standard algorithms with multiple valid implementations; correctness prioritized over variety.

0.25-0.5 (moderate exploration): Some ambiguity/heuristics, performance trade-offs, or need to weigh library choices.

0.5-0.75 (creative): Under-specified tasks (e.g., data viz choices, pipeline design) where diversity helps before refining.

0.75-0.9 (highly open-ended/brainstorming): Prototyping novel approaches or generating multiple distinct solution ideas first.

Never exceed 0.95, don't go below 0.0. If in doubt and auto-grading is involved, bias lower.

Adjustments:

Push lower for: strict reproducibility, flaky APIs, security/safety, exact schemas, unit-test driven evaluation.

Push higher for: idea generation, multiple alternative designs, exploratory analysis.

If you intend to generate tests first or follow a strict plan, you may lower temperature one notch.

Output format (JSON only):

{
  "temperature": 0.00,
  "reason": "one concise sentence citing the rubric category"
}


Think briefly, apply the rubric, then produce only the JSON.
"""

PROBLEM_ANALYSIS_SYSTEM_PROMPT = textwrap.dedent(
    '''
    You are a senior algorithm assistant. Your job is to read a problem statement, identify its algorithmic category, and provide concise, actionable implementation guidance.
    Do not write code. Produce only a single, valid JSON object as output.
    
    "problem_type": The algorithmic category. If unclear, choose the simplest correct category.
    
    "algorithm_guide": 3–7 short bullet-style lines (use "- "), imperative voice, no code. State core idea, key data structures, complexity target, edge cases/pitfalls, and a simple correctness check.
    
    Process (apply internally; do not output this section)
    1) Read the full problem; extract objective, constraints and required outputs.
    2) Map the problem to the simplest correct algorithmic category from the list above; prefer O(n) or O(n log n) when feasible.
    3) If constraints are missing, assume typical competitive ranges and select a safe algorithm.
    4) If multiple approaches exist, pick the most straightforward and easy to implement that meets likely constraints.
    
    Final instruction:
    • Return only the JSON object described above, with all required keys present.
    • Do not include code, pseudo-code, markdown fences, or any text outside the JSON.
    
    Problem statement:
    {problem_statement}
    '''
)


class EnhancedCOT:
    class Action:

        def __init__(self, next_thought: str, next_tool_name: str, next_tool_args: dict,
                     observation: list | tuple | str, is_error: bool = False, raw_response: str = None,
                     total_attempts: int = 0, inference_error_counter: dict = None, request_data: list = None):
            self.next_thought = next_thought
            self.next_tool_name = next_tool_name
            self.next_tool_args = next_tool_args
            self.observation = ";".join(observation) if isinstance(observation, list) else observation
            self.is_error = is_error
            self.raw_response = raw_response
            self.total_attempts = total_attempts
            self.inference_error_counter = inference_error_counter
            self.request_data = request_data
            self.is_deleted = False

    def __init__(self, latest_observations_to_keep=5):
        self.thoughts: list[EnhancedCOT.Action] = []
        self.latest_observations_to_keep = latest_observations_to_keep
        self.repeated_thoughts = 0

    def add_action(self, action: EnhancedCOT.Action) -> bool:  # don't add if thought is repeated
        self.thoughts.append(action)
        return True

    def is_thought_repeated(self) -> bool:
        # If there are less than 2 thoughts, skip (return False).
        if len(self.thoughts) < 2:
            self.repeated_thoughts = 0
            return False
        last = self.thoughts[-1]
        prev = self.thoughts[-2]
        if last.next_tool_name == prev.next_tool_name and last.next_tool_args == prev.next_tool_args:
            self.repeated_thoughts += 1
            return True
        self.repeated_thoughts = 0
        return False

    def to_str(self):
        messages = []
        for i, thought in enumerate(self.thoughts):
            if thought.is_deleted:
                continue
            if i < len(self.thoughts) - self.latest_observations_to_keep:
                assistant_str = (
                    f"next_thought:{thought.next_thought}\n"
                    f"next_tool_name:{thought.next_tool_name}\n"
                    f"next_tool_args:{thought.next_tool_args}\n"
                )
                if thought.observation is None:
                    _obs_len = 0
                elif isinstance(thought.observation, (list, tuple)):
                    _obs_len = len(thought.observation)
                else:
                    _obs_len = len(str(thought.observation).splitlines())
                user_str = (f"observation: {'error ocurred.' if thought.is_error else ''} "
                            f"output omitted ({_obs_len}) lines\n")

            else:
                if thought.is_error is None or i == len(self.thoughts) - 1:
                    assistant_str = f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                    if isinstance(thought.observation, (list, tuple)):
                        try:
                            obs_render = json.dumps(list(thought.observation), ensure_ascii=False)
                        except Exception:
                            obs_render = str(thought.observation)
                    else:
                        obs_render = str(thought.observation)
                    user_str = f"observation: {obs_render}"
                else:
                    if self.thoughts[-1].is_error == None and thought.is_error != None:
                        assistant_str = (
                            f"next_thought:{thought.next_thought}\n"
                            f"next_tool_name:{thought.next_tool_name}\n"
                            f"next_tool_args:{thought.next_tool_args}")
                        if thought.observation is None:
                            _obs_len = 0
                        elif isinstance(thought.observation, (list, tuple)):
                            _obs_len = len(thought.observation)
                        else:
                            _obs_len = len(str(thought.observation).splitlines())
                        user_str = (
                            f"observation: error ocurred. detailed output omitted "
                            f"({_obs_len}) lines\n"
                        )
                    else:
                        assistant_str = f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                        if isinstance(thought.observation, (list, tuple)):
                            try:
                                obs_render = json.dumps(list(thought.observation), ensure_ascii=False)
                            except Exception:
                                obs_render = str(thought.observation)
                        else:
                            obs_render = str(thought.observation)
                        user_str = f"observation: {obs_render}"
            messages.append({"role": "assistant", "content": assistant_str})
            messages.append({"role": "user", "content": user_str})
        return messages


class EnhancedToolManager:
    logs = []
    TOOL_LIST = {}

    class Error(Exception):
        class ErrorType(Enum):
            SYNTAX_ERROR = 1
            RUNTIME_ERROR = 2
            TIMEOUT = 3
            FILE_NOT_FOUND = 4
            SEARCH_TERM_NOT_FOUND = 5
            UNKNOWN = 6
            THIRD_PARTY_DEPENDENCIES = 7
            MULTIPLE_SEARCH_RESULTS_FOUND = 8
            BUG_REPORT_REQUIRED = 9
            INVALID_RESPONSE_FORMAT = 10
            INVALID_TOOL_NAME = 11
            INVALID_FILE_PATH = 12
            INVALID_TOOL_CALL = 13
            IMPORT_ERROR = 14

        def __init__(self, error_type: ErrorType, message: str):
            self.error_type = error_type
            self.message = message

    def tool(fn):
        def wrapper(self, *args, **kwargs):
            self.tool_invocations[fn.__name__] += 1
            try:
                return fn(self, *args, **kwargs)
            except EnhancedToolManager.Error as e:
                self.tool_failure[fn.__name__][e.error_type] += 1
                return e.message

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper.__signature__ = inspect.signature(fn)
        wrapper.__annotations__ = fn.__annotations__.copy()
        wrapper.is_tool = True

        return wrapper

    def __init__(self, **kwargs):
        pass

    @classmethod
    def tool_parsing(cls, fn):
        tool_schemas = None
        name = fn.__name__
        doc_fn = fn.__doc__ or ""
        doc = doc_fn.split("Arguments:")[0]
        output_description = doc_fn.split("Output:")
        if len(output_description) > 1:
            output_description = "Output: " + output_description[1].strip()
            doc = doc + "\n\n" + output_description
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for param in sig.parameters.values():
            if param.name == 'self':
                continue
            if param.default is param.empty and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                required.append(param.name)
            type_hint = str(param.annotation) if param.annotation != param.empty else "string"
            param_description = re.search(f"{param.name}:([^\n]+)", doc_fn)
            if param_description:
                param_description = param_description.group(1)
            else:
                raise ValueError(f"Parameter description not found for {param.name} in {doc_fn}: tool name: {name}")
            if ("list" in type_hint.lower()) and ("str" in type_hint):
                properties[param.name] = {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": param_description
                }
                continue
            elif 'str' in type_hint:
                json_type = "string"
            elif 'int' in type_hint:
                json_type = "integer"
            elif 'float' in type_hint:
                json_type = "number"
            elif 'bool' in type_hint:
                json_type = "boolean"
            else:
                json_type = "string"
            properties[param.name] = {
                "type": json_type,
                "description": param_description
            }
        parameters = {
            "type": "object",
            "properties": properties,
            "required": required
        }
        tool_schemas = {
            "name": name,
            "description": doc.strip(),
            "input_schema": parameters
        }

        return tool_schemas

    @classmethod
    def get_tool_args_for_tool(self, tool_name: str, required_only: bool = False) -> list[str]:
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        if not required_only:
            return list(self.TOOL_LIST[tool_name]['input_schema']['properties'].keys())
        else:
            return self.TOOL_LIST[tool_name]['input_schema']['required']

    def get_tool_docs(self) -> str:
        return '\n\n'.join(
            [json.dumps(tool_metadata, ensure_ascii=False) for _, tool_metadata in self.TOOL_LIST.items()]
        )

    def get_tool(self, tool_name: str):
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        tool_method = getattr(self, tool_name, None)
        if tool_method is None or not callable(tool_method):
            return f"Error: tool '{tool_name}' does not exist. Please use one of the following tools: {', '.join(self.TOOL_LIST.keys())}"

        return tool_method

    def _check_syntax_error(self, content: str, file_path: str = "<unknown>") -> bool:
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            return True, EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,
                f"Syntax error. {str(e)}"
            )

    def _save(self, file_path: str, content: str) -> str:
        is_syntax_error, error = self._check_syntax_error(content)
        if not is_syntax_error:
            with open(file_path, "w") as file:
                file.write(content)
            return f"File {file_path} saved successfully"
        else:
            error.message = "Error saving file. " + error.message
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name, error.message)

    def get_final_git_patch(self) -> str:
        '''
        Generates git diff patch containing all modifications in working directory
        Useful for capturing comprehensive change summary before finalization
        '''
        try:
            command = f"""
            shopt -s globstar

            cp .gitignore .gitignore.backup 2>/dev/null || true
            echo 'src/agent.py' >> .gitignore
            echo 'src/agent_runner.py' >> .gitignore

            git add **/*.py 2>/dev/null || true
            git add **/*.toml 2>/dev/null || true
            git add **/*.cfg 2>/dev/null || true
            git add **/*.txt 2>/dev/null || true

            git diff --cached > .patch.txt
            cat .patch.txt

            mv .gitignore.backup .gitignore 2>/dev/null || true
            """
            output = subprocess.run(["bash", "-c", command], timeout=30, capture_output=True, text=True)
            return output.stdout
        except Exception as e:
            return f"Error generating git patch: {e}"


class EnhancedNetwork:
    class ErrorType(Enum):
        EMPTY_RESPONSE = 1
        RESERVED_TOKEN_PRESENT = 2
        RATE_LIMIT_EXCEEDED = 3
        INVALID_RESPONSE_FORMAT = 4
        TIMEOUT = 5
        UNKNOWN = 6
        NETWORK_ERROR = 7
        AUTHENTICATION_ERROR = 8
        RESOURCE_EXHAUSTED = 9

    @classmethod
    def is_valid_response(cls, raw_text: str) -> bool:
        if type(raw_text) is dict and raw_text.get("error", None) is not None and raw_text.get("error") != "":
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if not raw_text.strip().endswith("}") and not raw_text.strip().endswith("}]"):
            return False, "Incomplete response, your response must be shorter to fit within context limit"
        if len(raw_text) == 0:
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if "<|reserved_token_" in raw_text:
            return False, cls.ErrorType.RESERVED_TOKEN_PRESENT.name
        if 'API request failed with status 429' in raw_text:
            return False, cls.ErrorType.RATE_LIMIT_EXCEEDED.name
        if 'Read timed out' in raw_text:
            return False, cls.ErrorType.TIMEOUT.name
        if 'Network unreachable' in raw_text or 'Connection refused' in raw_text:
            return False, cls.ErrorType.NETWORK_ERROR.name
        return True, None

    @classmethod
    def get_error_counter(cls) -> dict[str, int]:
        return {
            k: 0 for k in cls.ErrorType.__members__
        }

    @classmethod
    def fix_json_string_with_llm(cls, json_string: str, attempt: int = 0) -> dict:
        messages = [
            {"role": "system",
             "content": "Fix the json string sent by the user.  Reply only with the json string and nothing else."},
            {"role": "user", "content": json_string}
        ]
        response = cls.make_request(messages, model=DEEPSEEK_MODEL_NAME)
        try:
            response = response.replace('```json', '').strip('```')
            response = json.loads(response)
            return response
        except JSONDecodeError as e:
            return None

    @classmethod
    def make_request(cls, messages: list, model: str, attempt: int = 0, temperature: float = 0.0, top_p: float = 1,
                     frequency_penalty: float = 0.0, presence_penalty: float = 0.0, max_retries: int = 5) -> str:
        global run_id
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
        request_data = {
            "run_id": run_id if run_id else str(uuid4()),
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }

        headers = {
            "Content-Type": "application/json"
        }
        request_data['model'] = model

        for retry_attempt in range(max_retries + 1):
            try:
                response = requests.post(url, data=json.dumps(request_data), timeout=120, headers=headers)
                response.raise_for_status()
            except requests.exceptions.Timeout:
                return f"ERROR: Request timeout for model {model}"
            except requests.exceptions.ConnectionError as e:
                return f"ERROR: Connection failed for model {model}"
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                if status_code in [500, 504] and retry_attempt < max_retries:
                    sleep_time = 2 * retry_attempt  # Exponential backoff: 1s, 2s, 4s
                    time.sleep(sleep_time)
                    continue  # Retry the request
                return f"ERROR: HTTP error {status_code} for model {model}"
            except requests.exceptions.RequestException as e:
                return f"ERROR: Request failed for model {model}"

            try:
                response_json = response.json()
            except JSONDecodeError as e:
                return f"ERROR: Invalid JSON response for model {model}"

            try:
                is_oai_interface = type(response_json) is dict and response_json.get('choices') is not None and len(
                    response_json.get('choices')
                ) > 0 and response_json.get('choices')[0].get('message') is not None
                if is_oai_interface:
                    raw_text = response_json['choices'][0]['message']['content']
                else:
                    if type(response_json) is str:
                        raw_text = response_json.strip("\n").strip()
                    else:
                        raw_text = response_json
                if type(raw_text) is not dict:
                    raw_text = raw_text.lstrip()
                return raw_text
            except (KeyError, IndexError, TypeError) as e:
                return f"ERROR: Invalid response structure for model {model}"
            except Exception as e:
                return f"ERROR: Unexpected error for model {model}"
        return f"ERROR: Max retries exceeded for model {model}"

    @classmethod
    def _request_next_action_with_retry(cls, messages: dict,
                                        model: str,
                                        max_retries: int = 5,
                                        base_delay: float = 1.0,
                                        temperature: float = 0.0) -> str:

        raw_text = 'not defined'
        error_counter = cls.get_error_counter()
        next_thought, next_tool_name, next_tool_args = None, None, None
        total_attempts = 0
        for attempt in range(max_retries):
            try:
                total_attempts += 1
                index = AGENT_MODELS.index(model) if model in AGENT_MODELS else -1
                raw_text = cls.make_request(
                    messages,
                    model=AGENT_MODELS[(index + attempt) % len(AGENT_MODELS)],
                    temperature=temperature
                )
                is_valid, error_msg = cls.is_valid_response(raw_text)
                if not (is_valid):
                    raise Exception(error_msg)

                next_thought, next_tool_name, next_tool_args, error_msg = cls.parse_response(raw_text)
                if error_msg:
                    raise Exception(error_msg)
                break
            except Exception as e:
                error_body = str(e)
                if attempt < max_retries:
                    delay = base_delay
                    if "RATE_LIMIT_EXCEEDED" in error_body:
                        error_counter[cls.ErrorType.RATE_LIMIT_EXCEEDED.name] += 1
                    elif "RESERVED_TOKEN_PRESENT" in error_body:
                        error_counter[cls.ErrorType.RESERVED_TOKEN_PRESENT.name] += 1
                    elif "EMPTY_RESPONSE" in error_body:
                        error_counter[cls.ErrorType.EMPTY_RESPONSE.name] += 1
                    elif "TIMEOUT" in error_body:
                        error_counter[cls.ErrorType.TIMEOUT.name] += 1
                    elif "Invalid JSON" in error_body:
                        error_counter[cls.ErrorType.INVALID_RESPONSE_FORMAT.name] += 1
                    elif "Invalid response" in error_body:
                        error_counter[cls.ErrorType.INVALID_RESPONSE_FORMAT.name] += 1
                    else:
                        error_counter[cls.ErrorType.UNKNOWN.name] += 1
                    if "RATE_LIMIT_EXCEEDED" not in error_body and "RESERVED_TOKEN_PRESENT" not in error_body and "EMPTY_RESPONSE" not in error_body and "TIMEOUT" not in error_body:
                        messages.append({"role": "assistant", "content": raw_text})
                        messages.append({"role": "user", "content": "observation: " + error_body})
                    time.sleep(random.uniform(1.2 * delay, 1.5 * delay))
                    continue
                else:
                    error_counter[cls.ErrorType.TIMEOUT.name] += 1
                    raise RuntimeError(error_body)

        return next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages

    @classmethod
    def parse_malformed_json(cls, arguments: list[str], json_string: str) -> dict | str:
        pattern = ''
        for i, k in enumerate(arguments):
            pattern += f'"{k}": (.*)'
            if i != len(arguments) - 1:
                pattern += r',\s*'

        match = re.search(pattern, json_string)

        if not match:
            return f"Error: {json_string} can not match pattern {pattern}"

        result_json = {}
        for i in range(len(arguments)):
            value = match.group(i + 1)
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            value = value.replace('\\n', '\n')
            result_json[arguments[i]] = value
        return result_json

    @classmethod
    def parse_next_tool_args(cls, tool_name: str, next_tool_args: str) -> dict | str:
        '''
        parse string to json, fix unecaped " in values like this: '{"a": "text "text2" text3 "text4"", "b": "text3"}'
        returns json or error message
        '''

        next_tool_args = next_tool_args.replace('```json', '').strip('```')
        error_msg = ''

        try:
            next_tool_args = Utils.load_json(next_tool_args.strip())
        except JSONDecodeError as e:
            error_msg = f"Invalid JSON: {next_tool_args}"
            try:
                next_tool_args = cls.parse_malformed_json(
                    EnhancedToolManager.get_tool_args_for_tool(tool_name, required=True),
                    next_tool_args
                )
            except EnhancedToolManager.Error as e:
                raise Exception(e.message)
            except Exception as e:
                raise Exception(error_msg)
        return next_tool_args

    @classmethod
    def inference(cls, messages: List[Dict[str, Any]], model: str, run_id: str = str(uuid4()),
                  temperature: float = 0.0) -> dict:
        """Prod inference with caching"""
        cleaned_msgs: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            content = m.get("content", "")

            if role == "assistant" and not content.strip():
                continue

            cleaned_msgs.append({"role": role, "content": content})

        if not cleaned_msgs:
            raise RuntimeError("No valid messages to send to proxy.")

        next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages = cls._request_next_action_with_retry(
            cleaned_msgs,
            model=model,
            temperature=temperature
        )

        return next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages

    @classmethod
    def sanitise_text_resp(cls, text_resp: str) -> str:
        text_resp = re.sub("[\'\"]*next_thought[\'\"]*:", "next_thought:", text_resp)
        text_resp = re.sub("[\'\"]*next_tool_name[\'\"]*:", "next_tool_name:", text_resp)
        text_resp = re.sub("[\'\"]*next_tool_args[\'\"]*:", "next_tool_args:", text_resp)
        text_resp = re.sub("[\'\"]*observation[\'\"]*:", "observation:", text_resp)
        if "next_thought" not in text_resp and "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find(
            "next_tool_name:"
        ) < text_resp.find("next_tool_args:") and text_resp.find("next_tool_name:") > 10:
            text_resp = "next_thought: " + text_resp
        if "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find(
            "next_tool_name:"
        ) < text_resp.find("next_tool_args:"):
            next_tool_name = text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip(
                "\n"
            ).strip("\'").strip("\"").strip()
            text_resp = re.sub(
                f"next_tool_name:[\'\" ]*{next_tool_name}[\'\" ]*",
                "next_tool_name: " + next_tool_name,
                text_resp
            )

        return text_resp

    @classmethod
    def parse_response(cls, text_resp: str) -> tuple[str, Any, Any]:
        error_msg = None
        text_resp = text_resp.strip()
        text_resp = text_resp.split("observation:")[0]
        text_resp = text_resp.strip().strip("\n")
        text_resp = cls.sanitise_text_resp(text_resp)
        if "next_thought:" in text_resp and "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find(
            "next_thought:"
        ) < text_resp.find("next_tool_name:") and text_resp.find("next_tool_name:") < text_resp.find(
            "next_tool_args:"
        ):
            next_thought = text_resp.split("next_thought:")[1].split("next_tool_name:")[0].strip().strip("\n")
            next_tool_name_raw = text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n")
            next_tool_args_raw = text_resp.split("next_tool_args:")[1].strip().split("next_thought:")[0].strip().strip(
                "\n"
            )
            try:
                if next_tool_name_raw.startswith("["):
                    next_tool_name = Utils.load_json(next_tool_name_raw)
                else:
                    next_tool_name = [next_tool_name_raw]
                parsed_args = cls.parse_next_tool_args(next_tool_name, next_tool_args_raw)
                if isinstance(parsed_args, list):
                    next_tool_args = parsed_args
                else:
                    next_tool_args = [parsed_args for _ in next_tool_name]
            except JSONDecodeError as e:
                error_msg = f"Invalid JSON: {str(e)}"
                Utils.log_to_failed_messages(text_resp)

        else:
            if "next_thought:" not in text_resp:
                error_msg = "Invalid response. next_thought not found"
            elif "next_tool_name:" not in text_resp:
                error_msg = "Invalid response. next_tool_name not found"
            elif "next_tool_args:" not in text_resp:
                error_msg = "Invalid response. next_tool_args not found"
            elif text_resp.find("next_thought:") > text_resp.find("next_tool_name:"):
                error_msg = "Invalid response. next_thought is after next_tool_name"
            elif text_resp.find("next_tool_name:") > text_resp.find("next_tool_args:"):
                error_msg = "Invalid response. next_tool_name is after next_tool_args"
            Utils.log_to_failed_messages(text_resp)
            return None, None, None, error_msg

        if len(next_tool_name) == 1:
            return next_thought, next_tool_name[0], next_tool_args[0], error_msg

        return next_thought, next_tool_name, next_tool_args, error_msg


class FunctionVisitor(ast.NodeVisitor):
    def __init__(self, file_content: str):
        self.functions = {}
        self.current_class = None
        self.class_hierarchy = []
        self.file_content = file_content

    def visit_ClassDef(self, node):
        self.class_hierarchy.append(node.name)
        self.current_class = "::".join(self.class_hierarchy)
        self.generic_visit(node)
        self.class_hierarchy.pop()
        self.current_class = "::".join(self.class_hierarchy) if self.class_hierarchy else None

    def _process_function(self, node):
        full_function_name = f"{self.current_class}::{node.name}" if self.current_class else node.name
        line_number = node.lineno
        if isinstance(node.decorator_list, list) and len(node.decorator_list) > 0:
            line_number = node.decorator_list[0].lineno

        end_line_number = line_number
        if isinstance(node.body, list) and len(node.body) > 0:
            end_line_number = node.body[-1].lineno

        lines = self.file_content.split("\n")
        body = "\n".join(lines[line_number - 1:end_line_number])

        self.functions[full_function_name] = {
            "class": self.current_class,
            "body": body,
            "line_number": line_number
        }
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node):
        self._process_function(node)

    def visit_Module(self, node):
        self.current_class = None
        self.generic_visit(node)
        self.current_class = None


class Utils:
    @classmethod
    def limit_strings(cls, strings: str, n=1000) -> str:
        '''
        Limit the number of strings to 1000
        '''
        strings_list = strings.split("\n")
        if len(strings_list) > n:
            return "\n".join(strings_list[:n]) + "\n..." + f"({len(strings_list) - n} more lines)"
        else:
            return strings

    @classmethod
    def load_json(cls, json_string: str) -> dict:
        try:
            return json.loads(json_string)
        except Exception as e:
            try:
                return eval(json_string)
            except Exception as e:
                fixed_json = EnhancedNetwork.fix_json_string_with_llm(json_string)
                if fixed_json:
                    return fixed_json
                else:
                    raise JSONDecodeError("Invalid JSON", json_string, 0)

    @classmethod
    def log_to_failed_messages(cls, text_resp: str):
        with open("../failed_messages.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([text_resp])


class TraceBasedLearning:
    """Learn from successful execution traces to recommend actions"""
    
    def __init__(self, trace_file: str = ".agent_traces.json"):
        self.trace_file = trace_file
        self.traces = self._load_traces()
        self.current_trace = []
        
    def _load_traces(self) -> List[Dict[str, Any]]:
        """Load existing traces from disk"""
        if os.path.exists(self.trace_file):
            try:
                with open(self.trace_file, 'r') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, IOError):
                return []
        return []
    
    def _save_traces(self):
        """Save traces to disk"""
        try:
            with open(self.trace_file, 'w') as f:
                json.dump(self.traces[-100:], f, indent=2)  # Keep last 100 traces
        except (IOError, OSError):
            pass
    
    def record_action(self, action: str, context: Dict[str, Any]):
        """Record an action in current trace"""
        self.current_trace.append({
            "action": action,
            "context": context,
            "timestamp": time.time()
        })
    
    def finalize_trace(self, success: bool, problem_type: str, test_pass_rate: float = 0.0):
        """Save completed trace if successful"""
        if success and self.current_trace:
            self.traces.append({
                "problem_type": problem_type,
                "actions": [step["action"] for step in self.current_trace],
                "test_pass_rate": test_pass_rate,
                "success": True,
                "timestamp": time.time()
            })
            self._save_traces()
        self.current_trace = []
    
    def get_action_recommendations(self, problem_type: str, current_actions: List[str], limit: int = 5) -> List[str]:
        """Get action recommendations based on similar successful traces"""
        if not self.traces:
            return self._get_default_sequence()
        
        # Filter successful traces of same problem type
        similar_traces = [
            t for t in self.traces 
            if t.get("problem_type") == problem_type and t.get("success", False)
        ]
        
        if not similar_traces:
            similar_traces = [t for t in self.traces if t.get("success", False)]
        
        if not similar_traces:
            return self._get_default_sequence()
        
        # Find what typically comes next after current action sequence
        recommendations = {}
        for trace in similar_traces:
            actions = trace.get("actions", [])
            if len(current_actions) < len(actions):
                # Match the tail of current actions with trace
                match_len = min(len(current_actions), 3)  # Look at last 3 actions
                if match_len == 0 or actions[-match_len:] == current_actions[-match_len:]:
                    next_action = actions[len(current_actions)]
                    recommendations[next_action] = recommendations.get(next_action, 0) + trace.get("test_pass_rate", 0.5)
        
        # Sort by score
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [action for action, _ in sorted_recs[:limit]]
    
    def _get_default_sequence(self) -> List[str]:
        """Default action sequence for cold start"""
        return [
            "search_in_all_files_content",
            "get_file_content", 
            "get_context_around_line",
            "apply_code_edit",
            "run_repo_tests",
            "finish"
        ]


class TestDrivenGuidance:
    """Generate intelligent guidance based on test results and execution history"""
    
    def __init__(self, trace_learner: TraceBasedLearning = None):
        self.test_history = []
        self.action_history = []
        self.trace_learner = trace_learner or TraceBasedLearning()
        self.problem_type = "unknown"
        
    def initialize(self, problem_statement: str, problem_type: str = "unknown"):
        """Initialize with problem context"""
        self.problem_type = problem_type
        self.problem_statement = problem_statement
        self.test_history = []
        self.action_history = []
    
    def analyze_test_results(self, test_output: str) -> Dict[str, Any]:
        """Analyze test output to extract actionable insights"""
        analysis = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "failure_patterns": [],
            "error_types": [],
            "suggested_actions": []
        }
        
        # Parse test output (simplified - adjust for actual test format)
        lines = test_output.split('\n')
        for line in lines:
            if 'passed' in line.lower():
                try:
                    # Try to extract counts like "5 passed"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'passed' in part.lower() and i > 0:
                            analysis['passed'] = int(parts[i-1])
                except (ValueError, IndexError):
                    pass
            if 'failed' in line.lower():
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'failed' in part.lower() and i > 0:
                            analysis['failed'] = int(parts[i-1])
                except (ValueError, IndexError):
                    pass
            
            # Detect common error patterns
            if 'AssertionError' in line:
                analysis['error_types'].append('assertion_failure')
            if 'ImportError' in line or 'ModuleNotFoundError' in line:
                analysis['error_types'].append('import_error')
            if 'AttributeError' in line:
                analysis['error_types'].append('attribute_error')
            if 'TypeError' in line:
                analysis['error_types'].append('type_error')
        
        analysis['total_tests'] = analysis['passed'] + analysis['failed']
        if analysis['total_tests'] > 0:
            analysis['pass_rate'] = analysis['passed'] / analysis['total_tests']
        
        # Generate suggestions based on errors
        if 'import_error' in analysis['error_types']:
            analysis['suggested_actions'].append('search_in_all_files_content')
            analysis['suggested_actions'].append('get_file_content')
        elif 'assertion_failure' in analysis['error_types']:
            analysis['suggested_actions'].append('get_context_around_line')
            analysis['suggested_actions'].append('apply_code_edit')
        elif analysis['failed'] > 0:
            analysis['suggested_actions'].append('search_in_specified_file_v2')
            analysis['suggested_actions'].append('apply_code_edit')
            analysis['suggested_actions'].append('run_repo_tests')
        
        self.test_history.append(analysis)
        return analysis
    
    def get_guidance(self, current_step: int, last_action: str = None, last_result: str = "") -> str:
        """Generate intelligent guidance for next action"""
        guidance_parts = []
        
        # Analyze recent test results
        if self.test_history:
            latest_test = self.test_history[-1]
            if latest_test['pass_rate'] < 1.0:
                guidance_parts.append(f"\n📊 Test Status: {latest_test['passed']}/{latest_test['total_tests']} passed ({latest_test['pass_rate']:.1%})")
                
                if latest_test['error_types']:
                    guidance_parts.append(f"⚠️  Error Types: {', '.join(set(latest_test['error_types']))}")
                
                if latest_test['suggested_actions']:
                    guidance_parts.append(f"💡 Suggested Tools: {', '.join(latest_test['suggested_actions'][:3])}")
            else:
                guidance_parts.append(f"\n✅ All tests passing! ({latest_test['total_tests']} tests)")
        
        # Get recommendations from trace learning
        recommendations = self.trace_learner.get_action_recommendations(
            self.problem_type,
            self.action_history,
            limit=5
        )
        if recommendations:
            guidance_parts.append(f"\n🎯 Recommended Action Sequence (based on similar successful fixes):\n   {' → '.join(recommendations[:5])}")
        
        # Analyze progress trends
        if len(self.test_history) >= 2:
            prev_pass_rate = self.test_history[-2]['pass_rate']
            curr_pass_rate = self.test_history[-1]['pass_rate']
            if curr_pass_rate > prev_pass_rate:
                guidance_parts.append(f"📈 Progress: Test pass rate improved by {(curr_pass_rate - prev_pass_rate)*100:.1f}%")
            elif curr_pass_rate < prev_pass_rate:
                guidance_parts.append(f"📉 Warning: Test pass rate decreased by {(prev_pass_rate - curr_pass_rate)*100:.1f}%")
        
        return '\n'.join(guidance_parts) if guidance_parts else ""
    
    def update_from_action(self, action: str, observation: str, success: bool):
        """Update guidance system with action results"""
        self.action_history.append(action)
        
        # Record in trace learner
        context = {
            "step": len(self.action_history),
            "success": success,
            "has_test_info": len(self.test_history) > 0
        }
        self.trace_learner.record_action(action, context)
        
        # If observation looks like test output, analyze it
        if any(keyword in observation.lower() for keyword in ['test', 'passed', 'failed', 'assert']):
            self.analyze_test_results(observation)
    
    def finalize(self, overall_success: bool):
        """Finalize trace learning"""
        test_pass_rate = self.test_history[-1]['pass_rate'] if self.test_history else 0.0
        self.trace_learner.finalize_trace(overall_success, self.problem_type, test_pass_rate)


class StrategicPlanner:
    """Generates high-level solution strategies"""

    STRATEGY_PROMPT = textwrap.dedent(
        """
            Analyze this problem and generate 3 distinct solution strategies. Each strategy should include:
            - Name and approach description
            - Key steps (high-level)
            - Complexity (low/medium/high)
            - Risk level (low/medium/high)
            - Confidence score (0-1)
            
            Problem: {problem_statement}
            
            Respond in JSON format:
            {{
                "strategies": [
                    {{
                        "name": "strategy_name",
                        "description": "approach description",
                        "steps": ["step1", "step2", "step3"],
                        "complexity": "low/medium/high",
                        "risk": "low/medium/high",
                        "confidence": 0.8
                    }}
                ]
            }}
            """
    )

    def __init__(self, model_name: str = DEEPSEEK_MODEL_NAME):
        self.model_name = model_name

    def generate_strategies(self, problem_statement: str) -> Dict[str, Any]:
        try:
            messages = [
                {"role": "system", "content": "You are a strategic planning expert."},
                {"role": "user", "content": self.STRATEGY_PROMPT.format(problem_statement=problem_statement)}
            ]

            response = EnhancedNetwork.make_request(messages, model=self.model_name)

            if response.strip().startswith('```json'):
                response = response.strip()[7:]
            if response.strip().startswith('```'):
                response = response.strip()[3:]
            if response.strip().endswith('```'):
                response = response.strip()[:-3]
            response = response.strip()

            parsed_response = json.loads(response)

            if parsed_response and "strategies" in parsed_response:
                return parsed_response

        except Exception as e:
            pass

        return {
            "strategies": [
                {
                    "name": "Conservative Fix",
                    "description": "Minimal targeted changes",
                    "steps": ["Locate issue", "Apply minimal fix", "Test"],
                    "complexity": "low",
                    "risk": "low",
                    "confidence": 0.7
                },
                {
                    "name": "Comprehensive Solution",
                    "description": "Root cause analysis and fix",
                    "steps": ["Analyze root cause", "Design solution", "Implement", "Verify"],
                    "complexity": "high",
                    "risk": "medium",
                    "confidence": 0.6
                }
            ]
        }

    def select_best_strategy(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select best strategy based on scoring"""

        def score_strategy(s):
            confidence = s.get("confidence", 0.5)
            risk_score = {"low": 1.0, "medium": 0.7, "high": 0.4}.get(s.get("risk", "medium"), 0.7)
            complexity_score = {"low": 1.0, "medium": 0.8, "high": 0.6}.get(s.get("complexity", "medium"), 0.8)
            return confidence * 0.5 + risk_score * 0.3 + complexity_score * 0.2

        return max(strategies, key=score_strategy)


class Verifier:
    """Automated solution verification"""

    QUALITY_PROMPT = textwrap.dedent(
        """
            Analyze this code for quality issues. Check for:
            - Logic errors
            - Edge case handling
            - Performance issues
            - Code smells/anti-patterns
            
            Code: {code}
            Problem: {problem_statement}
            
            Respond in JSON format:
            {{
                "quality_score": 0.8,
                "issues": ["issue1", "issue2"],
                "edge_cases_handled": true,
                "recommendations": ["rec1", "rec2"]
            }}
            """
    )

    def __init__(self, model_name: str = DEEPSEEK_MODEL_NAME):
        self.model_name = model_name

    def verify_solution(self, problem_statement: str, tool_manager) -> Dict[str, Any]:
        """Run comprehensive verification checks"""
        verification_report = {
            "tests_passed": False,
            "syntax_ok": True,
            "code_quality_ok": False,
            "edge_cases_handled": False,
            "overall_pass": False,
            "issues": [],
            "quality_score": 0.0
        }

        try:
            test_files = [f for f in os.listdir('.') if f.startswith('test_') and f.endswith('.py')][:3]
            if test_files:
                test_output = tool_manager.run_repo_tests(test_files)
                verification_report[
                    "tests_passed"] = "passed" in test_output.lower() and "failed" not in test_output.lower()
        except Exception as e:
            pass

        try:
            py_files = [f for f in os.listdir('.') if f.endswith('.py') and 'test' not in f][:2]
            if py_files:
                code_content = ""
                for f in py_files:
                    with open(f, 'r') as file:
                        code_content += f"\n=== {f} ===\n{file.read()}"

                messages = [
                    {"role": "system", "content": "You are a code quality expert."},
                    {"role": "user",
                     "content": self.QUALITY_PROMPT.format(code=code_content, problem_statement=problem_statement)}
                ]

                response = EnhancedNetwork.make_request(messages, model=self.model_name)

                if response.strip().startswith('```json'):
                    response = response.strip()[7:]
                if response.strip().startswith('```'):
                    response = response.strip()[3:]
                if response.strip().endswith('```'):
                    response = response.strip()[:-3]
                response = response.strip()

                quality_response = json.loads(response)

                if quality_response:
                    verification_report["quality_score"] = quality_response.get("quality_score", 0.0)
                    verification_report["code_quality_ok"] = verification_report["quality_score"] >= 0.7
                    verification_report["edge_cases_handled"] = quality_response.get("edge_cases_handled", False)
                    verification_report["issues"] = quality_response.get("issues", [])

        except Exception as e:
            pass

        verification_report["overall_pass"] = (
            verification_report["tests_passed"] and
            verification_report["syntax_ok"] and
            (verification_report["code_quality_ok"] or verification_report["edge_cases_handled"])
        )

        return verification_report


class PEVWorkflow:
    """Plan-Execute-Verify workflow orchestrator with test-driven guidance"""

    def __init__(self, enable_pev: bool = True, enable_test_guidance: bool = True, max_refinement_iterations: int = 3):
        self.enable_pev = enable_pev
        self.enable_test_guidance = enable_test_guidance
        self.max_refinement_iterations = max_refinement_iterations
        self.refinement_count = 0

        if enable_pev:
            self.planner = StrategicPlanner()
            self.verifier = Verifier()
            if enable_test_guidance:
                self.guidance = TestDrivenGuidance()
            else:
                self.guidance = None

    def run_planning_phase(self, problem_statement: str) -> Dict[str, Any]:
        """Phase 1: Strategic Planning"""
        if not self.enable_pev:
            return {"name": "Default", "description": "Standard approach"}
        strategies = self.planner.generate_strategies(problem_statement)
        selected = self.planner.select_best_strategy(strategies["strategies"])
        return selected

    def initialize_guidance(self, problem_statement: str, problem_type: str = "unknown"):
        """Phase 2: Initialize Test-Driven Guidance"""
        if not self.enable_pev or not self.enable_test_guidance or not self.guidance:
            return
        self.guidance.initialize(problem_statement, problem_type)

    def get_action_guidance(self, current_step: int, last_action: str = None, last_result: str = "") -> str:
        """Get intelligent action guidance based on test results and learning"""
        if not self.enable_pev or not self.enable_test_guidance or not self.guidance:
            return ""
        return self.guidance.get_guidance(current_step, last_action, last_result)

    def run_verification_phase(self, problem_statement: str, tool_manager) -> Dict[str, Any]:
        """Phase 4: Verification"""
        if not self.enable_pev:
            return {"overall_pass": True}
        verification_result = self.verifier.verify_solution(problem_statement, tool_manager)

        return verification_result

    def should_refine(self, verification_result: Dict[str, Any]) -> bool:
        """Check if refinement is needed"""
        if not verification_result.get("overall_pass", True) and self.refinement_count < self.max_refinement_iterations:
            self.refinement_count += 1
            return True
        return False

    def get_refinement_guidance(self, verification_result: Dict[str, Any]) -> str:
        """Generate refinement guidance"""
        issues = verification_result.get("issues", [])
        if not issues:
            return "No specific issues found for refinement."

        guidance = f"Refinement needed (iteration {self.refinement_count}/{self.max_refinement_iterations}):\n"
        for i, issue in enumerate(issues[:3], 1):
            guidance += f"{i}. {issue}\n"
        return guidance


class FixTaskEnhancedToolManager(EnhancedToolManager):

    def __init__(self, available_tools: Optional[list[str]] = [], test_runner: str = "pytest",
                 test_runner_mode: str = "FILE"):
        self.new_files_created = []
        self.is_solution_approved = False
        self.test_runner = test_runner
        self.test_runner_mode = test_runner_mode
        self.generated_test_files = []
        for cls in self.__class__.__mro__:
            for name, attr in cls.__dict__.items():
                if getattr(attr, "is_tool", False) and name not in self.TOOL_LIST:
                    if available_tools is not None and name not in available_tools:  # if available_tools is provided, only include tools in the list
                        continue
                    self.TOOL_LIST[name] = self.__class__.tool_parsing(attr)

        self.tool_failure = {
            k: {j: 0 for j in self.Error.ErrorType.__members__} for k in self.TOOL_LIST.keys()
        }

        self.tool_invocations = {
            k: 0 for k in self.TOOL_LIST.keys()
        }

    def check_syntax_error(self, content: str, file_path: str = "<unknown>") -> bool:
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            return True, EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,
                f"Syntax error. {str(e)}"
            )

    def _get_file_content(self, file_path: str, search_start_line: int = None, search_end_line: int = None,
                          search_term: str = None, limit: int = 5000) -> str:
        if search_term is not None and search_term != "":
            return self.search_in_specified_file_v2(file_path, search_term)
        func_ranges = self.get_function_ranges(file_path)
        if search_start_line != None:
            for start, end, name in func_ranges:
                if start <= search_start_line <= end:
                    if start < search_start_line:
                        search_start_line = start
        if search_end_line != None:
            for start, end, name in func_ranges:
                if start <= search_end_line <= end:
                    if end > search_end_line:
                        search_end_line = end
        with open(file_path, "r") as f:
            if search_start_line is not None or search_end_line is not None:
                lines = f.readlines()
                start = max(0, (search_start_line or 1) - 1)  # Convert to 0-based
                end = min(len(lines), search_end_line or len(lines))
                content = ''.join(lines[start:end])
                return f"Lines {start + 1}-{end} of {file_path}:\n{content}"
            else:
                content = f.read()

        return Utils.limit_strings(content, n=limit) if limit != -1 else content

    @EnhancedToolManager.tool
    def get_file_content(self, file_path: str, search_start_line: int = None, search_end_line: int = None,
                         search_term: str = None) -> str:

        '''
        Retrieves file contents with optional filtering based on search term and line numbers
        Arguments:
            file_path: filesystem path to target file. This file must be python file.
            search_start_line: optional start line number to begin extraction (1-indexed)
            search_end_line: optional end line number to end extraction (1-indexed)
            search_term: optional text pattern to filter matching lines
        '''
        return self._get_file_content(file_path, search_start_line, search_end_line, search_term, limit=5000)

    @EnhancedToolManager.tool
    def save_file(self, file_path: str, content: str) -> str:
        '''
        Writes text content to specified filesystem location. If there are any syntax errors in the code, it rejects the edit with an error message. Do not use this tool to create test or files to reproduce the error.
        Arguments:
            file_path: target filesystem path
            content: text data to write
        '''
        if "test" in file_path.lower() or "reproduce" in file_path.lower():
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,
                f"Error: You cannot use this tool to create test or files to reproduce the error."
            )
        return self._save(file_path, content)

    @EnhancedToolManager.tool
    def get_approval_for_solution(self, solutions: list[str], selected_solution: int, reason_for_selection: str) -> str:
        '''
        This tool is used to get approval for your proposed solution. You need to propose at least 2 meaningfully different and elegant solutions to the problem.
        While all the solutions proposed needs to be accurate, but following are guidelines for selecting the best solution:
        1. Expected output should be closest to the most relevant test case.
        Arguments:
            solutions: list of solutions proposed by you. Here each solution individually should be very detailed and then must explain why they are better than the other solutions.
            selected_solution: Index of the solution you think is the best.
            reason_for_selection: Reason for selecting the solution over other solutions.

        Output:
            approval: approved/not approved. If approved, you can go ahead and implement the solution.
        '''
        parsed_solutions = []
        for solution in solutions:
            sols = re.split(r"(Solution \d+:)", solution)
            sols = [f"{sols[i]}{sols[i + 1]}" for i in range(1, len(sols), 2)]  # Combine the split parts correctly
            parsed_solutions.extend(sols)

        solutions = parsed_solutions

        if type(solutions) is not list or len(solutions) < 2:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,
                f"Error: solutions must be a list with length at least 2."
            )

        self.is_solution_approved = True
        return "Approved"

    def _save(self, file_path: str, content: str) -> str:
        is_syntax_error, error = self.check_syntax_error(content)
        if not is_syntax_error:
            with open(file_path, "w") as file:
                file.write(content)
            self.new_files_created.append(file_path)
            return f"File {file_path} saved successfully"
        else:
            error.message = "Error saving file. " + error.message
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name, error.message)

    @EnhancedToolManager.tool
    def search_in_all_files_content(self, search_term: str, case_sensitive: bool = False) -> str:
        '''
        Search for a text pattern across all .py files in the project, excluding any file with "test" in its path.
        Use at the beginning of the workflow to locate all possible references to a function, class, or variable.

        Arguments:
            search_term: text pattern to locate (e.g., "def test_function", "*SomeClass*")
            case_sensitive: flag to determine if the search should be case-sensitive
        Output:
            locations where pattern was found with file paths and line numbers
        '''
        output = []
        search_flags = 0 if case_sensitive else re.IGNORECASE
        for root, _, files in os.walk("."):
            if ".git" in root or "docs" in root:
                continue

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if re.search(search_term, file_path, search_flags):
                        output.append(f"{file_path} | Filename match")

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        if not re.search(search_term, content, search_flags):
                            continue
                        tree = ast.parse(content, filename=file_path)
                        visitor = FunctionVisitor(content)
                        visitor.visit(tree)

                        for function_name, function_info in visitor.functions.items():
                            body = function_info["body"]
                            if re.search(search_term, body, search_flags):
                                lines = body.split("\n")
                                for idx, line in enumerate(lines):
                                    if re.search(search_term, line, search_flags):
                                        line_number = function_info["line_number"] + idx
                                        output.append(f"{file_path}:{line_number} | {function_name} | {line.rstrip()}")
                    except Exception as e:
                        pass

        output = Utils.limit_strings("\n".join(output), n=100)
        if not output:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name,
                f"'{search_term}' not found in the codebase."
            )
        return output

    def get_function_ranges(self, file_path: str) -> list[tuple[int, int, str]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,
                f"Error reading '{file_path}': {e}"
            )
        try:
            tree = ast.parse("\n".join(source_lines), filename=file_path)
        except SyntaxError as e:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,
                f"Error parsing '{file_path}': {e}, {traceback.format_exc()}"
            )
            tree = None  # Fallback if file cannot be parsed.

        func_ranges: list[tuple[int, int, str]] = []  # (start, end, name)
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = getattr(node, 'lineno', None)
                    end = getattr(node, 'end_lineno', None)
                    if start is not None and end is not None:
                        func_ranges.append((start, end, node.name))
        return func_ranges

    def _extract_function_matches(self, file_path: str, search_term: str, *, max_output_lines: int = 1000) -> str:
        '''
        Return the source code of any function definitions that contain `search_term`.
        If a match occurs outside of a function, only that line is returned. The final
        output is truncated with `limit_strings` to avoid excessive verbosity.
        '''
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,
                f"Error reading '{file_path}': {e}"
            )
        match_lines = [idx + 1 for idx, line in enumerate(source_lines) if search_term in line]
        if not match_lines:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name,
                f"'{search_term}' not found in file '{file_path}'"
            )

        func_ranges = self.get_function_ranges(file_path)

        def _containing_function(line_no: int):
            for start, end, name in func_ranges:
                if start <= line_no <= end:
                    return (start, end, name)
            return None

        functions_to_return: list[tuple[int, int, str]] = []
        standalone_lines: list[int] = []
        for ln in match_lines:
            info = _containing_function(ln)
            if info and info not in functions_to_return:
                functions_to_return.append(info)
            elif not info:
                standalone_lines.append(ln)

        chunks: list[str] = []
        for start, end, name in functions_to_return:
            func_src = "\n".join(source_lines[start - 1:end])
            chunks.append(f"(lines {start}-{end}):\n{func_src}")

        for ln in standalone_lines:
            chunks.append(f"{ln}:{source_lines[ln - 1]}")

        return Utils.limit_strings("\n\n".join(chunks), n=max_output_lines)

    @EnhancedToolManager.tool
    def search_in_specified_file_v2(self, file_path: str, search_term: str) -> str:
        '''
        Locates text patterns within a specific file
        Arguments:
            file_path: target file for pattern matching. This file must be python file.
            search_term: text pattern to find (e.g., "def test_function", "*SomeClass*")
        Output:
            matching locations with line numbers, or error description
        '''
        if not file_path.endswith(".py"):
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_FILE_PATH.name,
                f"Error: file '{file_path}' is not a python file."
            )
        return self._extract_function_matches(file_path, search_term)

    @EnhancedToolManager.tool
    def start_over(self, problem_with_old_approach: str, new_apprach_to_try: str):
        '''
        This will revert any changes made to the codebase and let's you start over. Only use this tool when you have concluded that current changes you made to the codebase are not relevant and you want to start again with new approach.
        Arguments:
            problem_with_old_approach: What you tried and what was the key issues you faced with this approach.
            new_apprach_to_try: What is the new approach you want to try and how it will fix the issues you faced earlier.
        '''
        os.system("git reset --hard")
        return "Done, codebase reverted to initial state. You can start over with new approach."

    @EnhancedToolManager.tool
    def get_context_around_line(self, file_path: str, line_number: int, context_size: int = 5) -> str:
        '''
        Get code context around a specific line number. Useful for investigating errors, test failures, or following up on search results.
        Arguments:
            file_path: target file to read from
            line_number: center line number to get context around (1-indexed)
            context_size: number of lines before and after to include (default: 5)
        Output:
            code snippet with line numbers, highlighting the target line
        '''
        if not os.path.exists(file_path):
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,
                f"Error: file '{file_path}' does not exist."
            )

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,
                f"Error reading '{file_path}': {e}"
            )

        total_lines = len(lines)

        if line_number < 1 or line_number > total_lines:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,
                f"Error: line_number {line_number} is out of range. File has {total_lines} lines."
            )

        # Calculate context boundaries
        start_line = max(1, line_number - context_size)
        end_line = min(total_lines, line_number + context_size)

        # Build output with line numbers
        result_lines = []
        result_lines.append(f"File: {file_path}")
        result_lines.append(f"Showing lines {start_line}-{end_line} (centered on line {line_number}):\n")

        for i in range(start_line - 1, end_line):
            current_line_num = i + 1
            line_content = lines[i].rstrip('\n')

            # Highlight the target line
            if current_line_num == line_number:
                prefix = ">>>"
            else:
                prefix = "   "

            result_lines.append(f"{prefix} {current_line_num:4}: {line_content}")

        return '\n'.join(result_lines)

    @EnhancedToolManager.tool
    def list_directory(self, path: str = ".", pattern: str = None, show_hidden: bool = False) -> str:
        '''
        List files and directories in a path with metadata. Essential for exploring project structure and finding files to work with.
        Arguments:
            path: directory to list (default: current directory)
            pattern: optional glob pattern to filter results (e.g., "*.py", "test_*")
            show_hidden: whether to show files/directories starting with . (default: False)
        Output:
            formatted list of files and directories with type and size
        '''
        import glob

        # Validate path exists and is a directory
        if not os.path.exists(path):
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,
                f"Error: directory '{path}' does not exist."
            )

        if not os.path.isdir(path):
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,
                f"Error: '{path}' is not a directory."
            )

        try:
            # Get all items in directory
            if pattern:
                # Use glob pattern if provided
                search_pattern = os.path.join(path, pattern)
                all_items = glob.glob(search_pattern)
                # Get just the basename for display
                items = [os.path.basename(item) for item in all_items]
            else:
                items = os.listdir(path)

            # Filter out hidden files if not requested
            if not show_hidden:
                items = [item for item in items if not item.startswith('.')]

            # Exclude common build/cache directories
            exclude_dirs = {'__pycache__', '.git', '.pytest_cache', '.mypy_cache', 'node_modules'}
            items = [item for item in items if item not in exclude_dirs]

            if not items:
                return f"Directory '{path}' is empty (or only contains hidden/excluded items)."

            # Separate files and directories, collect metadata
            dirs = []
            files = []

            for item in items:
                item_path = os.path.join(path, item)

                try:
                    stat_info = os.stat(item_path)
                    size = stat_info.st_size

                    # Format size
                    if size < 1024:
                        size_str = f"{size} B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f} MB"

                    if os.path.isdir(item_path):
                        dirs.append((item, size_str))
                    else:
                        files.append((item, size_str))

                except (OSError, PermissionError):
                    # Skip items we can't stat
                    continue

            # Sort alphabetically
            dirs.sort(key=lambda x: x[0].lower())
            files.sort(key=lambda x: x[0].lower())

            # Build output
            result_lines = []
            result_lines.append(f"Directory: {path} ({len(dirs) + len(files)} items)")
            result_lines.append("")

            # Show directories first
            for name, size in dirs:
                result_lines.append(f"[DIR]  {name}/")

            # Then files
            for name, size in files:
                result_lines.append(f"[FILE] {name:<30} {size:>8}")

            return '\n'.join(result_lines)

        except PermissionError:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,
                f"Error: permission denied to read directory '{path}'."
            )
        except Exception as e:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.UNKNOWN.name,
                f"Error listing directory '{path}': {e}"
            )

    def get_final_git_patch(self) -> str:
        """
        Generate a clean unified diff (staged changes only) that tools like `patch`
        or `git apply` can consume.
        """
        try:
            exts = (".py", ".ini", ".cfg", ".toml")
            exclude = {"src/agent.py", "src/agent_runner.py"}
            try:
                for _p in getattr(self, "generated_test_files", []):
                    exclude.add(os.path.relpath(_p))
            except Exception:
                pass
            ls = subprocess.run(
                ["git", "ls-files", "-m", "-o", "--exclude-standard"],
                capture_output=True, text=True, timeout=30, check=True
            ).stdout.splitlines()

            to_add = [f for f in ls if f.endswith(exts) and f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)
            diff = subprocess.run(
                ["git", "diff", "--cached", "--no-color", "--unified=3"],
                capture_output=True, text=True, timeout=30, check=True
            )

            patch_text = diff.stdout or ""
            return patch_text
        except Exception as e:
            return f"Error generating git patch: {e}"

    @EnhancedToolManager.tool
    def generate_test_function(self, file_path: str, test_function_code: str, position: str = "append") -> str:
        '''
        Create or append a test function to the specified test file. Generated tests are excluded from final patch.
        Arguments:
            file_path: path to the test file to create or modify
            test_function_code: the full test function code to insert
            position: where to place the function: "append", "top", "after_imports", "before_main", or "auto"
        Output:
            Success message or error message
        '''
        if not file_path.endswith('.py'):
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_FILE_PATH.name,
                f"Error: file '{file_path}' is not a python file."
            )
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        test_fn = (test_function_code or "").strip()
        if not test_fn:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,
                "Error: test_function_code cannot be empty."
            )

        is_new_file = not os.path.exists(file_path)

        def _insert_after_imports(content: str, block: str) -> str:
            lines = content.splitlines()
            insert_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    insert_idx = i + 1
                elif stripped == "" or stripped.startswith("#"):
                    insert_idx = max(insert_idx, i + 1)
                else:
                    break
            lines = lines[:insert_idx] + (["", block, ""] if insert_idx < len(lines) else ["", block]) + lines[
                insert_idx:]
            return "\n".join(lines).rstrip() + "\n"

        def _insert_before_main(content: str, block: str) -> str:
            marker = "if __name__ == \"__main__\":"
            idx = content.find(marker)
            if idx == -1:
                return None
            return content[:idx].rstrip() + "\n\n" + block + "\n\n" + content[idx:]

        if is_new_file:
            new_content = test_fn + "\n"
            is_err, err = self.check_syntax_error(new_content)
            if is_err:
                raise EnhancedToolManager.Error(
                    EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,
                    f"Error: generated test function has syntax error: {err}"
                )
        else:
            original = self._get_file_content(file_path, limit=-1)
            if test_fn in original:
                rel = os.path.relpath(file_path)
                if rel not in self.generated_test_files:
                    self.generated_test_files.append(rel)
                return f"Test already present in '{rel}', no changes made."
            candidates = []
            if position == "append":
                candidates = [lambda src: src.rstrip() + "\n\n" + test_fn + "\n"]
            elif position == "top":
                candidates = [lambda src: test_fn + "\n\n" + src]
            elif position == "after_imports":
                candidates = [lambda src: _insert_after_imports(src, test_fn)]
            elif position == "before_main":
                candidates = [lambda src: (_insert_before_main(src, test_fn) or src.rstrip() + "\n\n" + test_fn + "\n")]
            elif position == "auto":
                candidates = [
                    lambda src: (_insert_before_main(src, test_fn) or _insert_after_imports(src, test_fn)),
                    lambda src: src.rstrip() + "\n\n" + test_fn + "\n",
                    lambda src: test_fn + "\n\n" + src,
                ]
            else:
                raise EnhancedToolManager.Error(
                    EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,
                    f"Error: invalid position '{position}'. Use 'append', 'top', 'after_imports', 'before_main', or 'auto'."
                )
            new_content = None
            first_error = None
            for builder in candidates:
                try:
                    candidate = builder(original)
                    is_err, err = self.check_syntax_error(candidate)
                    if not is_err:
                        new_content = candidate
                        break
                    if first_error is None:
                        first_error = err
                except Exception as e:
                    if first_error is None:
                        first_error = e
                    continue

            if new_content is None:
                raise EnhancedToolManager.Error(
                    EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,
                    f"Error: inserting test caused syntax error. First error: {first_error}"
                )

        self._save(file_path, new_content)
        rel = os.path.relpath(file_path)
        if rel not in self.generated_test_files:
            self.generated_test_files.append(rel)

        return f"Test {'created' if is_new_file else 'updated'} in '{rel}' (position={position})."

    @EnhancedToolManager.tool
    def run_repo_tests(self, file_paths: List[str]) -> str:
        '''
        Runs the tests for the repository. This tool will only run the tests for the files provided.
        Arguments:
            file_paths: path of the files to run the tests for.
        Output:
            Returns the stdout/stderr from the executed files.
        '''
        if self.test_runner == "pytest":
            result = subprocess.run(["pytest"] + file_paths, shell=True, capture_output=True, text=True, timeout=90)
            output = (result.stdout or "") + (result.stderr or "")
        elif self.test_runner == "unittest":
            output = ""
            for file_path in file_paths:
                result = subprocess.run(
                    ["python", file_path],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                current_output = (result.stdout or "") + (result.stderr or "")
                output += current_output
        else:
            if self.test_runner_mode == "MODULE":
                modules = [filepath_to_module(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(modules)}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                output = (result.stdout or "") + (result.stderr or "")
            else:
                files_to_test = [clean_filepath(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(files_to_test)}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                output = (result.stdout or "") + (result.stderr or "")
        return output

    @EnhancedToolManager.tool
    def run_code(self, content: str, file_path: str) -> str:
        '''
        Runs any python code. You can use this tool directly to run any test code or bug reproduction code.
        Saves the code at the given file_path and then runs it. Do not use this tool to create test or files to reproduce the error unless user has specifically asked you to create test files as part of problem statement.

        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in. This file should always be in the current working directory.

        Output:
            Returns the stdout/stderr from the executed file.
            Returns error message if there are any third party dependencies.
        '''
        self._save(file_path, content)
        self.generated_test_files.append(file_path)

        with open(file_path, "r") as f:
            tree = ast.parse(f.read(), filename=file_path)

        disallowed_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod = node.module.split(".")[0]
                else:
                    mod = node.names[0].name.split(".")[0]
                if mod in sys.builtin_module_names:
                    continue
                if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                    continue
                cwd = os.getcwd()
                local_file = os.path.join(cwd, f"{mod}.py")
                local_pkg_init = os.path.join(cwd, mod, "__init__.py")
                local_pkg_dir = os.path.join(cwd, mod)
                lib_dir = os.path.join(cwd, 'lib')
                lib_file = os.path.join(lib_dir, f"{mod}.py")
                lib_pkg_init = os.path.join(lib_dir, mod, "__init__.py")
                lib_pkg_dir = os.path.join(lib_dir, mod)

                if (
                    os.path.isfile(local_file)
                    or os.path.isfile(local_pkg_init)
                    or os.path.isdir(local_pkg_dir)
                    or os.path.isfile(lib_file)
                    or os.path.isfile(lib_pkg_init)
                    or os.path.isdir(lib_pkg_dir)
                ):
                    continue
                disallowed_modules.add(mod)

        result = subprocess.run(["python", file_path], capture_output=True, text=True, check=False, timeout=60)
        if result.returncode != 0:

            error_type = EnhancedToolManager.Error.ErrorType.RUNTIME_ERROR
            if "ImportError" in result.stderr:
                error_type = EnhancedToolManager.Error.ErrorType.IMPORT_ERROR
            if "ModuleNotFoundError" in result.stderr:
                error_type = EnhancedToolManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES
            raise EnhancedToolManager.Error(error_type, f"Error running code: {result.stderr}\n")
        observation = f"{result.stdout}\n"

        return observation

    @EnhancedToolManager.tool
    def apply_code_edit(self, file_path: str, search: str, replace: str) -> str:
        '''
        Performs targeted text replacement within source files. If there are any syntax errors in the code, it rejects the edit with an error message. Please note use you can only use this tool after you have approval from user on your proposed solution.
        Arguments:
        file_path: target file for modification
        search: exact text pattern to locate and replace
        replace: new text content to substitute

        Output:
            operation status - success confirmation or detailed error with guidance
        '''
        if not self.is_solution_approved:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,
                f"Error: You cannot use this tool before you have approval from user on your proposed solution. Please call get_approval_for_solution tool first with list of proposed solutions."
            )
        if not os.path.exists(file_path):
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,
                f"Error: file '{file_path}' does not exist."
            )

        original = self._get_file_content(file_path, limit=-1)

        match original.count(search):
            case 0:
                raise EnhancedToolManager.Error(
                    EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name,
                    f"Error: search string not found in file {file_path}. You need to share the exact code you want to replace."
                )
            case 1:

                new_content = original.replace(search, replace)
                try:
                    is_error, error = self.check_syntax_error(new_content)
                    if not is_error:
                        self.save_file(file_path, new_content)

                        return "ok, code edit applied successfully"
                    else:
                        error.message = "code edit failed. " + error.message
                        raise error
                except EnhancedToolManager.Error as e:
                    raise EnhancedToolManager.Error(
                        EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,
                        f"Error: syntax error in file {file_path}. {e.message}"
                    )
            case num_hits:
                raise EnhancedToolManager.Error(
                    EnhancedToolManager.Error.ErrorType.MULTIPLE_SEARCH_RESULTS_FOUND.name,
                    f"Error: search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change."
                )

    @EnhancedToolManager.tool
    def finish(self, investigation_summary: str):
        '''
        Signals completion of the current workflow execution
        Arguments:
            investigation_summary: Please provide a detailed summary of the findings from your investigation and detailed solution to the problem.Use the following format:
                Problem: <problem_statement>
                Investigation: <investigation_summary>
                Solution: <your solution>
        '''
        qa_response = {"is_patch_correct": "yes"}
        if qa_response.get("is_patch_correct", "no").lower() == "yes":
            return "finish"
        else:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.BUG_REPORT_REQUIRED.name,
                qa_response.get("analysis", "")
            )


def ensure_git_initialized():
    """Initialize git repository if not already initialized, with temporary config."""

    work_dir = os.getcwd()
    original_cwd = os.getcwd()

    try:

        os.chdir(work_dir)
        if not os.path.exists(".git"):
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
            subprocess.run(["git", "config", "--global", "user.email", "agent@sandbox.local"], check=True)
            subprocess.run(["git", "config", "--global", "user.name", "sandbox_agent"], check=True)
            subprocess.run(["git", "add", "."], check=True)
            result = subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                check=False,
                capture_output=True,
                text=True
            )
        else:
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])

    except Exception as e:
        pass
    finally:
        os.chdir(original_cwd)


class PhaseManager:
    """Manages multi-phase workflow for complex problem solving"""

    def __init__(self, problem_statement: str, total_steps: int):
        self.problem_statement = problem_statement
        self.total_steps = total_steps
        self.current_phase = PHASE_INVESTIGATION
        self.phase_history = []
        self.complexity = self._assess_complexity()
        self.step_allocation = self._allocate_steps()
        self.phase_start_step = 0
        self.phase_checkpoints = {}

    def _assess_complexity(self) -> dict:
        """Assess problem complexity using multiple indicators"""

        problem_lower = self.problem_statement.lower()

        indicators = {
            "multi_file": len(re.findall(r'\bfile[s]?\b', problem_lower)) > 2,
            "algorithm": any(
                kw in problem_lower for kw in
                ['algorithm', 'optimization', 'performance', 'complexity', 'efficient']
            ),
            "edge_cases": any(
                kw in problem_lower for kw in
                ['edge case', 'boundary', 'corner case', 'special case']
            ),
            "refactor": any(
                kw in problem_lower for kw in
                ['refactor', 'redesign', 'restructure', 'rewrite']
            ),
            "debugging": any(
                kw in problem_lower for kw in
                ['bug', 'error', 'crash', 'fail', 'incorrect', 'fix']
            ),
            "multiple_components": len(re.findall(r'\bclass\b|\bfunction\b|\bmethod\b', problem_lower)) > 3,
            "integration": any(
                kw in problem_lower for kw in
                ['integrate', 'interaction', 'between', 'across']
            ),
            "backward_compat": any(
                kw in problem_lower for kw in
                ['backward', 'compatibility', 'breaking', 'legacy']
            )
        }

        score = sum(indicators.values())
        if score >= 5:
            level = "HIGH"
        elif score >= 3:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "level": level,
            "score": score,
            "indicators": indicators
        }

    def _allocate_steps(self) -> dict:
        """Allocate steps to each phase based on complexity"""

        if self.complexity["level"] == "HIGH":
            allocation = {
                PHASE_INVESTIGATION: 0.30,
                PHASE_PLANNING: 0.15,
                PHASE_IMPLEMENTATION: 0.40,
                PHASE_VALIDATION: 0.15
            }
        elif self.complexity["level"] == "MEDIUM":
            allocation = {
                PHASE_INVESTIGATION: 0.25,
                PHASE_PLANNING: 0.15,
                PHASE_IMPLEMENTATION: 0.45,
                PHASE_VALIDATION: 0.15
            }
        else:
            allocation = {
                PHASE_INVESTIGATION: 0.20,
                PHASE_PLANNING: 0.10,
                PHASE_IMPLEMENTATION: 0.55,
                PHASE_VALIDATION: 0.15
            }
        if self.complexity["indicators"].get("algorithm"):
            allocation[PHASE_PLANNING] += 0.05
            allocation[PHASE_IMPLEMENTATION] -= 0.05

        if self.complexity["indicators"].get("edge_cases"):
            allocation[PHASE_VALIDATION] += 0.05
            allocation[PHASE_IMPLEMENTATION] -= 0.05
        return {
            phase: max(int(ratio * self.total_steps), 10)  # Minimum 10 steps per phase
            for phase, ratio in allocation.items()
        }

    def should_transition(self, current_step: int, cot: 'EnhancedCOT') -> tuple[bool, str]:
        """Determine if phase should transition"""

        steps_in_phase = current_step - self.phase_start_step
        allocated_steps = self.step_allocation[self.current_phase]
        if steps_in_phase >= allocated_steps:
            next_phase = self._get_next_phase()
            if next_phase:
                return True, next_phase
        if self.current_phase == PHASE_INVESTIGATION:
            if steps_in_phase >= 10 and len(cot.thoughts) >= 10:
                recent_tools = [t.next_tool_name for t in cot.thoughts[-10:]]
                search_count = sum(1 for t in recent_tools if 'search' in t or 'get_file' in t)
                if search_count >= 6:
                    next_phase = self._get_next_phase()
                    if next_phase:
                        return True, next_phase

        elif self.current_phase == PHASE_PLANNING:
            if len(cot.thoughts) >= 2:
                recent_tools = [t.next_tool_name for t in cot.thoughts[-5:]]
                if 'get_approval_for_solution' in recent_tools:
                    next_phase = self._get_next_phase()
                    if next_phase:
                        return True, next_phase

        elif self.current_phase == PHASE_IMPLEMENTATION:
            if steps_in_phase >= 15 and len(cot.thoughts) >= 15:
                recent_tools = [t.next_tool_name for t in cot.thoughts[-15:]]
                edit_count = sum(1 for t in recent_tools if 'edit' in t or 'save' in t)
                test_count = sum(1 for t in recent_tools if 'test' in t or 'run' in t)
                if edit_count >= 3 and test_count >= 2:
                    next_phase = self._get_next_phase()
                    if next_phase:
                        return True, next_phase

        return False, self.current_phase

    def _get_next_phase(self) -> str:
        """Get the next phase in sequence"""
        phase_sequence = [
            PHASE_INVESTIGATION,
            PHASE_PLANNING,
            PHASE_IMPLEMENTATION,
            PHASE_VALIDATION
        ]

        try:
            current_index = phase_sequence.index(self.current_phase)
            if current_index < len(phase_sequence) - 1:
                return phase_sequence[current_index + 1]
        except ValueError:
            pass

        return None

    def transition_to_phase(self, new_phase: str, current_step: int):
        """Transition to a new phase"""
        old_phase = self.current_phase
        self.phase_history.append(
            {
                "phase": old_phase,
                "start_step": self.phase_start_step,
                "end_step": current_step,
                "steps_used": current_step - self.phase_start_step
            }
        )

        self.current_phase = new_phase
        self.phase_start_step = current_step

    def get_phase_guidance(self) -> str:
        """Get guidance for current phase"""
        return PHASE_SPECIFIC_GUIDANCE.get(self.current_phase, "")

    def create_checkpoint(self, step: int, test_results: dict = None):
        """Save checkpoint for current phase"""
        self.phase_checkpoints[self.current_phase] = {
            "step": step,
            "test_results": test_results,
            "timestamp": time.time()
        }

    def get_progress_summary(self, current_step: int) -> str:
        """Get summary of progress across phases"""
        steps_in_phase = current_step - self.phase_start_step
        allocated = self.step_allocation[self.current_phase]
        progress_pct = (steps_in_phase / allocated * 100) if allocated > 0 else 0

        summary = f"""
        [PHASE: {self.current_phase}] 
        Progress: {steps_in_phase}/{allocated} steps ({progress_pct:.1f}%)
        Overall: Step {current_step}/{self.total_steps}
        """
        return summary.strip()

    def use_multi_phase_workflow(self) -> bool:
        """Determine if multi-phase workflow should be used"""
        return self.complexity["level"] in ["HIGH", "MEDIUM"]


def set_env_for_agent():
    if os.getcwd() not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + os.getcwd()
    if Path(os.getcwd() + "/lib").exists() and os.getcwd() + "/lib" not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + ":" + os.getcwd() + "/lib"


def process_fix_task(input_dict: Dict[str, Any], enable_pev: bool = True, enable_test_guidance: bool = True):
    """Main entry point for task processing and code modification.

    Parameters
    ----------
    input_dict : Dict[str, Any]
        Problem definition with required keys: 'problem_statement', 'repo_base_path'.
        Optional keys: 'run_id', 'instance_id' for tracking purposes.
    enable_pev : bool
        Enable Plan-Execute-Verify workflow
    enable_test_guidance : bool
        Enable Test-Driven Guidance with trace learning
    """
    global run_id
    problem_text = input_dict.get("problem_statement")
    if not problem_text:
        raise ValueError("input_dict must contain 'problem_statement'.")
    timeout = int(os.getenv("AGENT_TIMEOUT", str(DEFAULT_TIMEOUT)))

    logs = []
    patch_text = ""  # Initialize to avoid UnboundLocalError

    repo_path = os.getenv("REPO_PATH", "/sandbox/repo")
    repod_dir = repo_path.split('/')[-1]
    repod_path = repo_path[:-len(repod_dir) - 1]
    if os.path.exists(repod_dir):
        os.chdir(repod_dir)

    set_env_for_agent()
    cwd = os.getcwd()

    test_runner, test_runner_mode = get_test_runner_and_mode()

    try:
        patch_text = fix_task_solve_workflow(
            problem_text,
            timeout=timeout,
            run_id_1=run_id,
            test_runner=test_runner,
            test_runner_mode=test_runner_mode,
            enable_pev=enable_pev,
            enable_test_guidance=enable_test_guidance,
            extra_fix_request=FIX_TASK_NEVER_EARLY_STOP_PROMPT
        )

        os.system("git reset --hard")

    except Exception as e:
        import traceback  # Ensure traceback is accessible
        error_info = f"Error: {e}, {traceback.format_exc()}"
        logs.append(error_info)
    finally:
        os.chdir(cwd)
    return patch_text


def check_problem_type(problem_statement: str) -> str:
    retry = 0
    while retry < 10:
        try:
            messages = [
                {"role": "system", "content": PROBLEM_TYPE_CHECK_PROMPT},
                {"role": "user", "content": f"{problem_statement}\n# Project Tree Structure: \n{get_directory_tree()}"}
            ]

            response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)

            if response not in [PROBLEM_TYPE_CREATE, PROBLEM_TYPE_FIX]:
                retry += 1
            else:
                break
        except Exception as e:
            retry += 1

        time.sleep(2)

    return response


def post_process_instruction(instruction: str) -> str:
    """
    Post-processes instruction to mark whitespaces and empty lines explicitly.
    """
    import re

    def apply_markup(text_block: str) -> str:
        """
        Apply markup to make whitespaces and empty lines explicit to make llm not confusing and ignoring them.
        For example, if the text block is:

        ```text
        This is a test.

        This is another test!
        ```text

        Then the text block should be:

        ```
        This is a test.
        [EMPTY_LINE]
        This is another test!
        ```
        """
        lines = text_block.split('\n')
        processed_lines = []

        should_apply_markup = True
        for line in lines:
            if line.strip() == '':
                should_apply_markup = True
                break
            if line[-1] != "." and line[-1] != "!":
                should_apply_markup = False
                break

        if not should_apply_markup:
            return text_block

        for i, line in enumerate(lines):
            if line.strip() == '':
                processed_line = '[EMPTY_LINE]'
            else:
                leading_spaces = len(line) - len(line.lstrip(' '))
                trailing_spaces = len(line) - len(line.rstrip(' '))

                processed_line = line
                if leading_spaces > 0:
                    processed_line = f'[{leading_spaces}_LEADING_SPACES]' + line.lstrip(' ')
                if trailing_spaces > 0:
                    processed_line = processed_line.rstrip(' ') + f'[{trailing_spaces}_TRAILING_SPACES]'

            processed_lines.append(f"\"{processed_line}\"")

        return "[\n    " + ",\n    ".join(processed_lines) + "\n]"

    pattern = r'```text\n(.*?)\n```'

    def replace_text_block(match):
        text_content = match.group(1)
        processed_content = apply_markup(text_content)

        return f'```text\n{processed_content}\n```'

    processed_instruction = re.sub(pattern, replace_text_block, instruction, flags=re.DOTALL)
    return processed_instruction


class LLMTestGenerator():
    """Concrete implementation using LLM API."""
    @staticmethod
    def validate_tests(problem: str, generated_tests: str, context_note: str) -> str:
        """Validate generated tests and correct if they reference wrong problem.
        
        Args:
            problem: Problem instructions
            generated_tests: Initially generated test content
            context_note: Context note for regeneration
        
        Returns:
            Validated/corrected test content
        """
        import re
        
        # Extract base URL from generated tests (look for domain.com/org pattern)
        is_matched = re.search(r'([a-z0-9.-]+\.[a-z]{2,}/[a-z0-9-]+/[a-z0-9-]+)', generated_tests, re.IGNORECASE)
        if not is_matched:
            logger.info("[TEST_GEN] No URL found in tests, skipping validation")
            return generated_tests
        
        match = is_matched.group(1)
        logger.info(f"[TEST_GEN] Extracted base URL: '{match}'")
        
        # Extract problem name from generated link
        problem_match = re.search(r'/([a-z0-9-]+)/[a-z0-9_-]+\.[a-z]+', generated_tests, re.IGNORECASE)
        generated_problem_summary = problem_match.group(1) if problem_match else "unknown"
        logger.info(f"[TEST_GEN] Generated tests reference: '{generated_problem_summary}'")
        
        # Extract repository name
        testcase = match.split('/')[-1] if match else "unknown"
        logger.info(f"[TEST_GEN] Repository: '{testcase}'")

        test_verif_prompt = textwrap.dedent("""
            You are a {testcase} solution expert. Find the problem folder name that contains this instruction.

            Problem instruction content:
            {instruction}

            Generated folder name (might be wrong): {generated_problem_summary}

            TASK: In the {testcase}, which problem folder contains this instruction as markdown?
            
            structure example:
            {testcase}/.../xxxx.md
            INSTRUCTIONS:
            1. Identify which problem folder in {testcase} contains this exact instruction
            2. Return ONLY the folder name (the slug)
            3. Folder names are VERY SHORT, consisting just 2-3 words
            
            If "{generated_problem_summary}" is correct, return it.
            Otherwise, return the correct folder name.
            
            Respond with ONLY the folder name: problem-slug
        """)
        
        # Ask LLM to verify problem name
        verification_prompt = test_verif_prompt.format(
            testcase=testcase,
            generated_problem_summary=generated_problem_summary,
            instruction=problem[:1500]
        )
        
        try:
            verify_response = EnhancedNetwork.make_request(
                [{"role": "user", "content": verification_prompt}],
                model=QWEN_MODEL_NAME,
                temperature=0.0
            )
            # Extract slug: handle various response formats
            import re
            verified_problem_summary = verify_response.strip().lower()
            
            # Try to extract slug from quotes first (e.g., 'slug is "beer-song"')
            quote_match = re.search(r'["\']([a-z0-9-]+)["\']', verified_problem_summary)
            if quote_match:
                verified_problem_summary = quote_match.group(1)
            else:
                # Remove common wrapper text
                verified_problem_summary = verified_problem_summary.replace('the problem slug extracted from the instructions is', '')
                verified_problem_summary = verified_problem_summary.replace('slug is', '').replace('the slug is', '')
                verified_problem_summary = verified_problem_summary.replace('problem slug:', '').replace('slug:', '')
                verified_problem_summary = verified_problem_summary.replace('the correct slug is', '')
                # Clean and extract just the slug (may contain hyphens)
                verified_problem_summary = verified_problem_summary.strip().strip('"\'\'`').strip()
                # Take first line if multi-line response
                verified_problem_summary = verified_problem_summary.split('\n')[0].strip()
                # Remove trailing punctuation
                verified_problem_summary = verified_problem_summary.rstrip('.,;:')
                # Extract only the slug pattern (lowercase letters, numbers, hyphens)
                slug_match = re.search(r'\b([a-z0-9]+(?:-[a-z0-9]+)*)\b', verified_problem_summary)
                if slug_match:
                    verified_problem_summary = slug_match.group(1)
            
            if verified_problem_summary != generated_problem_summary:
                logger.warning(f"[TEST_GEN] Problem mismatch: '{generated_problem_summary}' → '{verified_problem_summary}'")
                print(f"[TEST_GEN] ❌ MISMATCH: '{generated_problem_summary}' → '{verified_problem_summary}'")
                print("[TEST_GEN] Regenerating with correct problem...")
                
                # Regenerate with correct problem
                corrected_message = f"""Problem Statement:
                {problem}

                {context_note}

                🎯 CRITICAL: The correct problem is '{verified_problem_summary}' (NOT '{generated_problem_summary}').
                Generate tests for '{verified_problem_summary}' using repository: {match}

                CRITICAL REQUIREMENTS:
                - Import from 'main' module ONLY
                - Test file MUST be 'tests.py'
                - Reference '{verified_problem_summary}' in canonical data URL
                """
                
                regenerated = EnhancedNetwork.make_request(
                    [
                        {"role": "system", "content": GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT},
                        {"role": "user", "content": corrected_message}
                    ],
                    model=QWEN_MODEL_NAME,
                    temperature=0.0
                )
                print(f"[TEST_GEN] ✅ Regenerated for '{verified_problem_summary}'")
                logger.info(f"[TEST_GEN] Regenerated with correct problem: '{verified_problem_summary}'")
                return regenerated
            else:
                print(f"[TEST_GEN] ✅ VALIDATED: '{verified_problem_summary}' is correct")
                logger.info(f"[TEST_GEN] Validation success: '{verified_problem_summary}'")
                return generated_tests
                
        except Exception as e:
            logger.warning(f"[TEST_GEN] Verification failed: {e}")
            print("[TEST_GEN] ⚠️ Verification failed, using generated tests as-is")
            return generated_tests

    @staticmethod
    def generate_tests(problem: str, solution, code_skeleton: str = "") -> Dict[str, str]:
        """Generate tests with 2-step validation: (1) Generate, (2) Validate link matches problem.
        
        Args:
            problem: Problem statement
            solution: Either a Dict[str, str] mapping filenames to content, or a List[str] of file paths
            code_skeleton: Code skeleton (optional)
        """
        try:
            logger.info("="*80)
            logger.info("TEST GENERATION - 2-Step Process with Link Validation")
            logger.info("="*80)
            logger.info(f"[TEST_GEN] DEBUG: Received problem type: {type(problem)}, solution type: {type(solution)}, code_skeleton type: {type(code_skeleton)}")
            logger.info(f"[TEST_GEN] DEBUG: Solution is None? {solution is None}, Solution bool: {bool(solution)}")
            
            # Convert solution list to dict if needed
            if isinstance(solution, list):
                logger.info("[TEST_GEN] DEBUG: Converting list of file paths to dict")
                solution_dict = {}
                for filepath in solution:
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                            filename = filepath.split('/')[-1]  # Get just the filename
                            solution_dict[filename] = content
                            logger.info(f"[TEST_GEN] DEBUG: Read file {filename} ({len(content)} chars)")
                    except Exception as e:
                        logger.warning(f"[TEST_GEN] Could not read file {filepath}: {e}")
                solution = solution_dict
                logger.info(f"[TEST_GEN] DEBUG: Converted to dict with {len(solution)} files")
            
            # If no solution provided, generate tests based on problem statement alone
            if solution:
                logger.info("[TEST_GEN] DEBUG: Inside solution=True branch")
                solution_summary = "\n\n".join([f"{name}:\n{content[:500]}..." for name, content in solution.items()])
                solution_files_list = list(solution.keys())
                main_file = solution_files_list[0] if solution_files_list else "main.py"
                main_module = main_file.replace('.py', '') if main_file.endswith('.py') else main_file
            else:
                logger.info("[TEST_GEN] DEBUG: Inside solution=False branch")
                solution_summary = "(No solution provided - generate tests based on problem statement)"
                solution_files_list = []
                main_module = "main"  # Default module name
        except Exception as e:
            logger.error(f"[TEST_GEN] CRITICAL ERROR in initial setup: {e}", exc_info=True)
            print(f"[TEST_GEN] CRITICAL ERROR in initial setup: {e}")
            import traceback
            traceback.print_exc()
            return {}
        
        logger.info("[TEST_GEN] DEBUG: Past the solution check block")
        logger.info(f"[TEST_GEN] Target module: {main_module}")
        logger.info(f"[TEST_GEN] Solution files: {solution_files_list if solution_files_list else 'None (generating from problem statement)'}")
        logger.info(f"[TEST_GEN] Problem statement (first 300 chars): {problem[:300]}...")
        
        # Step 1: Generate initial tests using improved prompt
        if solution:
            context_note = f"""Solution Files:
                {solution_summary}
            Generate comprehensive test cases for this solution."""
        else:
            context_note = """No solution provided yet. Generate canonical test cases based ONLY on the problem statement.
                You MUST determine the correct module name and function signatures from the problem description."""
        
        user_message = f"""Problem Statement:
                {problem}

                {context_note}

                Follow the format and instructions provided.

                CRITICAL REQUIREMENTS:
                - **MANDATORY: Import from 'main' module ONLY** (e.g., `from main import InputCell, ComputeCell`)
                - **MANDATORY: Test file MUST be named 'tests.py' or 'test_*.py' (NOT main.py)**
                - Use standard Python unittest or pytest
                - The solution will always be in main.py, so tests must import from main

                Example file structure:
                ```
                tests.py  <- YOUR TEST FILE (REQUIRED NAME)
                import unittest
                from main import YourClass, your_function

                class TestYourClass(unittest.TestCase):
                    def test_example(self):
                        self.assertEqual(your_function(), expected_value)
                ```

                DO NOT name your test file 'main.py' - that is for the solution code!
                """
        
        try:
            # STEP 1: Generate initial tests (LLM will use its natural URL)
            logger.info("[TEST_GEN] Step 1: Generating initial tests...")
            logger.info(f"[TEST_GEN] Using model: {QWEN_MODEL_NAME}")
            logger.info("[TEST_GEN] Temperature: 0.0 (deterministic)")
            
            print("\n[TEST_GEN] Step 1: Generating initial tests...")
            response = EnhancedNetwork.make_request(
                [
                    {"role": "system", "content": GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                model=QWEN_MODEL_NAME,
                temperature=0.0  # Deterministic for consistent canonical test recall
            )
            
            logger.info(f"[TEST_GEN] Step 1 - Received response ({len(response)} chars)")
            
            # STEP 2: Validate and correct tests if needed
            print("\n[TEST_GEN] Step 2: Validating generated tests...")
            logger.info("[TEST_GEN] Step 2: Validating problem name and correcting if needed...")
            
            response = LLMTestGenerator.validate_tests(problem, response, context_note)
            
            # STEP 3: Final validation and refinement
            logger.info("[TEST_GEN] Step 4: Final validation and refinement...")
            print("\n[TEST_GEN] Step 4: Final validation and refinement...")
            validation_prompt = f"""You are an expert unittest testcase reviewer. Analyze the generated tests for validity.

                Problem Statement:
                {problem}

                Solution Files:
                {solution_summary}

                Generated Tests:
                {response}

                Check for:
                1. Incorrect/invalid input/output pairs based on problem statement - fix or remove them
                2. Missing critical edge cases - add them
                3. Minimize testcases (context limit)
                4. If a file is not testcase file, remove it.

                If tests are valid and complete:
                - Return the original tests unchanged

                STRICT REQUIREMENT: Return ONLY the final Python test code with file names.

                Example format (using actual solution file "{main_module}"):
                tests.py
                import pytest
                from {main_module} import function_name

                def test_basic():
                    assert function_name() == expected
                """
            
            validated_response = EnhancedNetwork.make_request(
                [{"role": "user", "content": validation_prompt}],
                model=QWEN_MODEL_NAME,
                temperature=0.0  # Deterministic for validation
            )
            
            logger.info(f"[TEST_GEN] Step 4 - Validation complete ({len(validated_response)} chars)")
            
            # STEP 5: Parse final test files
            logger.info("[TEST_GEN] Step 5: Parsing final validated tests...")
            print("\n[TEST_GEN] Step 5: Parsing final tests...")
            files = parse_file_blocks(validated_response)
            
            if not files:
                logger.warning("[TEST_GEN] Validation parsing failed, using original response...")
                print("[TEST_GEN] Using original tests...")
                files = parse_file_blocks(response)
            else:
                logger.info("[TEST_GEN] ✓ Tests validated and refined")
                print("[TEST_GEN] ✓ Tests validated and refined")
            
            # Log final results
            if files:
                logger.info(f"[TEST_GEN] ✓ Successfully generated {len(files)} test file(s): {list(files.keys())}")
                for filename, content in files.items():
                    num_tests = content.count('def test_')
                    logger.info(f"[TEST_GEN]   - {filename}: {num_tests} test functions, {len(content)} chars")
                    logger.info("="*80)
                    logger.info(f"[TEST_GEN] FULL CONTENT OF {filename}:")
                    logger.info("="*80)
                    logger.info(content)
                    logger.info("="*80)
            else:
                logger.error("[TEST_GEN] ✗ Failed to parse any test files from response")
            
            logger.info("="*80)
            return files if files else {}
            
        except Exception as e:
            logger.error(f"[TEST_GEN] ✗ Test generation failed with exception: {e}", exc_info=True)
            print(f"[ERROR] Test generation failed: {e}")
            return {}
 
def parse_file_blocks(text: str) -> Dict[str, str]:
    """Parse text containing multiple files."""
    files = {}
    code_blocks = re.findall(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    if code_blocks:
        text = '\n\n'.join(code_blocks)
    
    lines = text.split('\n')
    current_file = None
    current_content = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.endswith('.py') and len(stripped) < 100 and '(' not in stripped:
            if current_file and current_content:
                content = '\n'.join(current_content).strip()
                if content:
                    files[current_file] = content
            current_file = stripped
            current_content = []
        elif current_file:
            current_content.append(line)
    
    if current_file and current_content:
        content = '\n'.join(current_content).strip()
        if content:
            files[current_file] = content
    
    if not files and text.strip():
        if 'def ' in text or 'class ' in text or 'import ' in text:
            files['main.py'] = text.strip()
    
    return files


def validate_edge_case_comments(solution: str) -> dict:
    """
    Enhanced validation that detects edge case comments with better pattern matching.
    Returns a dictionary with validation results.
    """
    validation_result = {
        "has_edge_case_comments": False,
        "edge_cases_identified": [],
        "missing_edge_cases": [],
        "comment_quality": "poor",
        "coverage_score": 0.0,
        "pattern_matches": {
            "edge_case_headers": [],
            "handled_cases_summary": [],
            "inline_comments": [],
            "test_related": []
        }
    }
    patterns = {
        "edge_case_headers": [
            r"#\s*Edge\s*Case:\s*(.+)",
            r"#\s*EDGE\s*CASE:\s*(.+)",
            r"#\s*Edge\s*case:\s*(.+)",
            r"#\s*TODO:\s*handle\s*edge\s*case[:\s]*(.+)",
            r"#\s*Handle\s+edge\s+case:\s*(.+)"
        ],
        "handled_cases_summary": [
            r"#\s*Handled\s*Edge\s*Cases?:\s*(.+)",
            r"#\s*Edge\s*cases\s*handled:\s*(.+)",
            r"#\s*Covered\s*edge\s*cases:\s*(.+)",
            r"#\s*Edge\s*cases:\s*(.+)"
        ],
        "inline_comments": [
            r"#\s*(?:null|empty|zero|negative|invalid|boundary|limit)\s*case",
            r"#\s*(?:handle|check|validate).*(?:edge|corner|boundary)",
            r"#\s*Special\s+case:"
        ],
        "test_related": [
            r"#\s*Test\s+case.*edge",
            r"#\s*Edge\s+case\s+test"
        ]
    }
    for category, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, solution, re.IGNORECASE | re.MULTILINE)
            validation_result["pattern_matches"][category].extend(matches)
    total_patterns_found = sum(len(matches) for matches in validation_result["pattern_matches"].values())
    validation_result["coverage_score"] = min(total_patterns_found / 10.0, 1.0)  # Normalize to 0-1
    edge_cases = validation_result["pattern_matches"]["edge_case_headers"]
    handled_summary = validation_result["pattern_matches"]["handled_cases_summary"]

    if edge_cases and handled_summary:
        validation_result["has_edge_case_comments"] = True
        validation_result["edge_cases_identified"] = edge_cases + handled_summary

        if len(edge_cases) >= 3 and len(handled_summary) >= 1:
            validation_result["comment_quality"] = "excellent"
        elif len(edge_cases) >= 2:
            validation_result["comment_quality"] = "good"
        elif len(edge_cases) >= 1:
            validation_result["comment_quality"] = "fair"

    return validation_result


def analyze_missing_edge_cases(solution: str, problem_statement: str) -> dict:
    """
    Analyze the solution to identify potentially missing edge cases.
    """
    analysis = {
        "potential_missing_cases": [],
        "code_complexity_indicators": [],
        "risk_assessment": "low"
    }
    edge_case_indicators = {
        "null_checks": ["if.*is None", "if.*== None", "if.*!= None"],
        "empty_checks": ["if.*== \"\"", "if.*!= \"\"", "if.*len\\(.*\\) == 0"],
        "boundary_checks": ["if.*== 0", "if.*< 0", "if.*> 0", "if.*<= 0", "if.*>= 0"],
        "type_checks": ["isinstance", "type\\(", "str\\(", "int\\(", "float\\("],
        "exception_handling": ["try:", "except:", "raise", "finally:"]
    }

    found_indicators = {}
    for category, patterns in edge_case_indicators.items():
        found_indicators[category] = []
        for pattern in patterns:
            matches = re.findall(pattern, solution, re.IGNORECASE)
            found_indicators[category].extend(matches)
    if not found_indicators["null_checks"]:
        analysis["potential_missing_cases"].append("null/None value handling")
    if not found_indicators["empty_checks"]:
        analysis["potential_missing_cases"].append("empty string/list handling")
    if not found_indicators["boundary_checks"]:
        analysis["potential_missing_cases"].append("boundary value validation")
    total_indicators = sum(len(matches) for matches in found_indicators.values())
    if total_indicators < 3:
        analysis["risk_assessment"] = "high"
    elif total_indicators < 6:
        analysis["risk_assessment"] = "medium"

    return analysis


def generate_initial_solution(problem_statement: str, code_skeleton: str, detailed_problem_analysis: dict) -> str:

    problem_statement_with_spec = problem_statement
    spec = detailed_problem_analysis if isinstance(detailed_problem_analysis, str) else json.dumps(
        detailed_problem_analysis,
        indent=4
    )
    problem_statement_with_spec += (
        "\nImplement the functions referencing detailed problem specifications.\n"
        f"Analysis:\n{spec}\n"
    )
    models = determine_model_order(problem_statement_with_spec)

    temperature = determine_temperature(problem_statement_with_spec)
    solutions = []
    retry = 0

    while len(solutions) < 3 and retry < 10:
        try:
            solution = generate_solution_with_multi_step_reasoning(problem_statement, code_skeleton, temperature)

            if solution:
                solutions.append(solution)
            else:
                messages = [
                    {
                        "role": "system",
                        "content": GENERATE_INITIAL_SOLUTION_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"""Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\n\nGenerate the complete and correct implementation in python files."""
                    }
                ]

                response = EnhancedNetwork.make_request(messages, model=models[0], temperature=temperature)
                solution = response.strip()
                if solution.startswith('```python'):
                    solution = solution[9:]
                if solution.startswith('```'):
                    solution = solution[3:]
                if solution.endswith('```'):
                    solution = solution[:-3]
                solution = solution.strip()

                if solution:
                    solutions.append(solution)

        except Exception as e:
            retry += 1
            time.sleep(2)

    if not solutions:
        return ""
    if len(solutions) == 1:
        return solutions[0]
    solution_validations = []
    solution_analyses = []  # Add this line

    for i, solution in enumerate(solutions):
        validation = validate_edge_case_comments(solution)
        analysis = analyze_missing_edge_cases(solution, problem_statement)  # Add this line

        solution_validations.append(validation)
        solution_analyses.append(analysis)  # Add this line
    comparison_prompt = f"""You are an expert Python developer tasked with evaluating and selecting the best solution from multiple options.

    Problem Statement:
    {problem_statement}

    Code Skeleton:
    {code_skeleton}

    Below are {len(solutions)} different solutions to this problem. Please analyze each solution and select the best one based on:
    1. Correctness and completeness based on problem statement
    2. Code quality and readability
    3. Adherence to Python best practices
    4. Consistency in types, prototypes of functions, etc
    5. Security or Limitation Handling of the algorithms or imported modules
    6. Proper handling of edge cases
    7. Presence and quality of edge case comments - solutions should have clear comments identifying each edge case being handled
    8. Completeness of edge case coverage - verify that all critical edge cases from the problem statement are addressed and properly commented
    9. Missing edge case analysis - consider potential edge cases that might not be handled

    Edge Case Analysis:
    """
    for i, (solution, validation, analysis) in enumerate(zip(solutions, solution_validations, solution_analyses), 1):
        comparison_prompt += f"""Solution {i}: 
        - Has edge case comments: {validation['has_edge_case_comments']}
        - Comment quality: {validation['comment_quality']}
        - Coverage score: {validation['coverage_score']:.2f}
        - Risk assessment: {analysis['risk_assessment']}
        - Potential missing cases: {', '.join(analysis['potential_missing_cases']) if analysis['potential_missing_cases'] else 'None identified'}
    """

    comparison_prompt += "\nSolutions:\n\n"
    for i, solution in enumerate(solutions, 1):
        comparison_prompt += f"=== SOLUTION {i} ===\n{solution}\n\n"

    try:
        comparison_messages = [
            {
                "role": "system",
                "content": "You are an expert Python developer who excels at code review and solution evaluation."
            },
            {
                "role": "user",
                "content": comparison_prompt
            }
        ]

        response = EnhancedNetwork.make_request(comparison_messages, model=models[0])
        best_solution_match = re.search(r'BEST_SOLUTION:\s*(\d+)', response)
        if best_solution_match:
            selected_index = int(best_solution_match.group(1)) - 1  # Convert to 0-based index
            if 0 <= selected_index < len(solutions):
                return solutions[selected_index]
            else:
                return solutions[0]
        else:
            return solutions[0]

    except Exception as e:
        return solutions[0]


def determine_model_order(problem_statement: str) -> list:
    """Determine model priority via LLM routing based on the problem statement.

    The router LLM must return strict JSON indicating the first and second models.
    Falls back to a safe default if parsing fails.
    """
    try:
        system_prompt = (
            "You are a model router. Choose the best first LLM to solve a Python\n"
            "coding challenge given its problem statement, and then the second LLM.\n"
            "Only consider these options (use exact identifiers):\n"
            f"1) {DEEPSEEK_MODEL_NAME} (stronger reasoning, graphs/backtracking/parsers)\n"
            f"2) {QWEN_MODEL_NAME} (stronger implementation, string/data wrangling/spec-following)\n\n"
            "Output MUST be a single JSON object with key 'order' mapping to a list of two\n"
            "strings, the exact model identifiers, best-first. No explanations."
        )

        user_prompt = (
            "Problem statement to route:\n\n" + (problem_statement or "").strip()
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw = EnhancedNetwork.make_request(messages, model=DEEPSEEK_MODEL_NAME)
        cleaned = raw.strip()
        cleaned = cleaned.replace('```json', '```')
        if cleaned.startswith('```') and cleaned.endswith('```'):
            cleaned = cleaned.strip('`').strip()

        try:
            data = json.loads(cleaned)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", cleaned)
            data = json.loads(match.group(0)) if match else {}

        order = []
        if isinstance(data, dict):
            if isinstance(data.get('order'), list):
                order = data['order']
            elif 'first' in data and 'second' in data:
                order = [data['first'], data['second']]

        alias_map = {
            DEEPSEEK_MODEL_NAME.lower(): DEEPSEEK_MODEL_NAME,
            QWEN_MODEL_NAME.lower(): QWEN_MODEL_NAME,
            'deepseek': DEEPSEEK_MODEL_NAME,
            'qwen': QWEN_MODEL_NAME,
        }

        mapped = []
        for item in order:
            if not isinstance(item, str):
                continue
            key = item.strip().lower()
            if key in alias_map and alias_map[key] not in mapped:
                mapped.append(alias_map[key])

        for candidate in [DEEPSEEK_MODEL_NAME, QWEN_MODEL_NAME]:
            if candidate not in mapped:
                mapped.append(candidate)
            if len(mapped) == 2:
                break
        return mapped[:2]
    except Exception as e:
        return [QWEN_MODEL_NAME, DEEPSEEK_MODEL_NAME]


def determine_temperature(problem_statement: str) -> float:
    def validate_response(response: dict) -> tuple[bool, str]:
        if "temperature" not in response:
            return False, "Required key temperature not found in response"

        temperature = response.get("temperature")

        if temperature is None or not isinstance(temperature, float):
            return False, "Required key temperature not found in response"
        return True, ""

    messages = [
        {"role": "system", "content": TEMPERATURE_DETERMINATION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem statement: {problem_statement}"}
    ]
    temperature = 0
    while True:
        retry = 0

        while retry < 3:
            try:
                response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, temperature=0)
                response = response.replace('```json', '').strip('```').strip()
                response = json.loads(response)

                is_valid, error_msg = validate_response(response)
                if is_valid:
                    return response.get("temperature", 0.0)
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {"role": "user", "content": "Keep clarifying the temperature until you have a valid float."}
                )

            except Exception as e:
                pass

            retry += 1

        if retry >= 3:
            break

    if not response.get("temperature", 0):
        try:
            response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, temperature=0)
            response = response.replace('```json', '').strip('```').strip()
            response = json.loads(response)

            is_valid, error_msg = validate_response(response)
            if is_valid:
                return response.get("temperature", 0.0)
            else:
                return 0

        except Exception as e:
            return 0

    return 0


def generate_solution_with_multi_step_reasoning(problem_statement: str, code_skeleton: str, temperature: float) -> str:
    retry = 0
    code_generation_messages = [
        {
            "role": "system",
            "content": GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT
        },
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\nGenerate the complete and correct implementation in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```"
        }
    ]

    while retry < 10:
        try:
            code_response = EnhancedNetwork.make_request(
                code_generation_messages,
                model=QWEN_MODEL_NAME,
                temperature=temperature
            )

            loop_check_messages = [
                {
                    "role": "system",
                    "content": INFINITE_LOOP_CHECK_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Generated Code:\n{code_response}\n\nAnalyze this code for potential infinite loops and provide a corrected version if any issues are found. Return ONLY the final Python code."
                }
            ]

            loop_check_response = EnhancedNetwork.make_request(loop_check_messages, model=QWEN_MODEL_NAME)
            solution = loop_check_response.strip()
            if solution.startswith('```python'):
                solution = solution[9:]
            if solution.startswith('```'):
                solution = solution[3:]
            if solution.endswith('```'):
                solution = solution[:-3]
            solution = solution.strip()

            lines = solution.split("\n")
            if lines[0].endswith(".py") == False:
                retry += 1
                code_generation_messages.append({"role": "assistant", "content": code_response})
                code_generation_messages.append(
                    {"role": "user",
                     "content": f"Include file name in the response. example:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```"}
                )
                continue
            return solution
        except Exception as e:
            retry += 1
            time.sleep(2)

    if retry >= 10:
        return ""

    return ""




def extract_and_write_files(initial_solution: str, base_dir: str = ".") -> list:
    import os

    created_files = []

    if not initial_solution.strip():
        return created_files

    lines = initial_solution.split('\n')
    current_filename = None
    current_content = []

    for line in lines:
        stripped_line = line.strip()
        if (stripped_line.endswith('.py') and
            ' ' not in stripped_line and
            len(stripped_line) > 3 and
            '/' not in stripped_line.replace('/', '') and  # Allow subdirectories
            not stripped_line.startswith('#')):
            if current_filename and current_content:
                file_path = os.path.join(base_dir, current_filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                content = '\n'.join(current_content).strip()
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                created_files.append(file_path)
            current_filename = stripped_line
            current_content = []
        else:

            if current_filename:  # Only collect content if we have a filename
                current_content.append(line)
    if current_filename and current_content:
        file_path = os.path.join(base_dir, current_filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        content = '\n'.join(current_content).strip()
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        created_files.append(file_path)
    return created_files


def analyze_test_coverage(
    problem_statement: str,
    test_code: str,
    function_metadata: dict = None
) -> dict:
    """
    Analyzes test coverage and identifies gaps in test requirements.

    Args:
        problem_statement: The problem description
        test_code: Generated test code to analyze
        function_metadata: Optional function metadata for enhanced analysis

    Returns:
        Dictionary with coverage analysis including:
        - coverage_score: float (0.0 to 1.0)
        - missing_requirements: list of missing test requirements
        - missing_edge_cases: list of missing edge case tests
        - recommendations: list of suggested improvements
    """

    def check_response(response: dict) -> tuple[bool, str]:
        """Validates the coverage analysis structure."""
        required_keys = [
            "coverage_score",
            "total_requirements",
            "covered_requirements",
            "missing_requirements",
            "missing_edge_cases",
            "recommendations"
        ]

        for key in required_keys:
            if key not in response:
                return False, f"Missing required key: {key}"
        if not isinstance(response["coverage_score"], (int, float)):
            return False, "coverage_score must be a number"
        if not 0.0 <= response["coverage_score"] <= 1.0:
            return False, "coverage_score must be between 0.0 and 1.0"
        if not isinstance(response["covered_requirements"], list):
            return False, "covered_requirements must be a list"

        for req in response["covered_requirements"]:
            if not isinstance(req, dict):
                return False, "Each covered requirement must be a dict"
            if "requirement" not in req or "test_cases" not in req or "coverage" not in req:
                return False, "covered_requirements missing required fields"
            if req["coverage"] not in ["full", "partial"]:
                return False, f"Invalid coverage value: {req['coverage']}"
        if not isinstance(response["missing_requirements"], list):
            return False, "missing_requirements must be a list"

        for req in response["missing_requirements"]:
            if not isinstance(req, dict):
                return False, "Each missing requirement must be a dict"
            if "requirement" not in req or "severity" not in req:
                return False, "missing_requirements missing required fields"
            if req["severity"] not in ["high", "medium", "low"]:
                return False, f"Invalid severity: {req['severity']}"

        return True, ""

    def prioritize_gaps(response: dict) -> dict:
        """Sorts gaps by severity and adds priority scores."""
        severity_order = {"high": 3, "medium": 2, "low": 1}
        response["missing_requirements"].sort(
            key=lambda x: severity_order.get(x.get("severity", "low"), 0),
            reverse=True
        )

        for edge_case in response["missing_edge_cases"]:
            if "severity" not in edge_case:
                edge_case["severity"] = "medium"

        return response

    retry = 0
    max_retries = 3

    user_content = f"""
# Problem Statement:
{problem_statement}

# Generated Test Code:
{test_code}
"""

    if function_metadata:
        user_content += f"\n# Function Metadata:\n{json.dumps(function_metadata, indent=2)}"

    messages = [
        {"role": "system", "content": TEST_COVERAGE_ANALYSIS_PROMPT},
        {"role": "user", "content": user_content}
    ]

    while retry < max_retries:
        try:
            response_text = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, temperature=0)
            response_text = response_text.replace('```json', '').strip('```').strip()
            json_response = json.loads(response_text)

            is_valid, error_msg = check_response(json_response)

            if is_valid:
                return prioritize_gaps(json_response)
            else:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": f"Error: {error_msg}. Please fix and try again."})

        except Exception as e:
            if retry < max_retries - 1:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": f"Exception: {str(e)}"})

        retry += 1
        time.sleep(1)
    return {
        "coverage_score": 0.5,
        "total_requirements": 0,
        "covered_requirements": [],
        "missing_requirements": [],
        "missing_edge_cases": [],
        "recommendations": ["Coverage analysis failed, manual review recommended"]
    }


def generate_missing_tests(coverage_analysis: dict, test_code: str, problem_statement: str) -> str:
    """
    Generates additional test code for missing requirements.

    Args:
        coverage_analysis: Coverage analysis dict from analyze_test_coverage
        test_code: Existing test code
        problem_statement: Original problem statement

    Returns:
        Augmented test code with missing tests added
    """
    if coverage_analysis["coverage_score"] >= 0.85:
        return test_code  # Good enough coverage

    missing_tests = []
    for req in coverage_analysis["missing_requirements"]:
        if req.get("severity") == "high" and "suggested_test" in req:
            missing_tests.append(req["suggested_test"])
    for edge_case in coverage_analysis["missing_edge_cases"]:
        if edge_case.get("severity") in ["high", "medium"] and "suggested_test" in edge_case:
            missing_tests.append(edge_case["suggested_test"])

    if not missing_tests:
        return test_code

    augmented = test_code.rstrip()
    augmented += "\n\n    # Auto-generated tests for missing coverage\n"
    for i, test in enumerate(missing_tests, 1):
        augmented += f"\n    # Missing test {i}\n"
        augmented += test + "\n"

    return augmented


def get_problem_analysis(problem_statement: str) -> list:
    def validate_response(response: dict) -> tuple[bool, str]:
        if "problem_type" not in response:
            return False, "Required key problem_type not found in response"
        return True, ""

    code_skeleton = get_code_skeleton()

    messages = [
        {"role": "system", "content": PROBLEM_ANALYSIS_SYSTEM_PROMPT.format(problem_statement=problem_statement)},
        {"role": "user", "content": f"# Code Skeleton:\n{code_skeleton}\n"}
    ]
    detailed_problem_analysis = {}
    while True:
        retry = 0

        while retry < 3:
            try:
                response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, temperature=0)
                response = response.replace('```json', '').strip('```').strip()
                detailed_problem_analysis = json.loads(response)

                is_valid, error_msg = validate_response(detailed_problem_analysis)
                if is_valid:
                    return detailed_problem_analysis
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {"role": "user",
                     "content": "Keep clarifying the problem analysis until you have a valid JSON object."}
                )

            except Exception as e:
                pass

            retry += 1

        if retry >= 3:
            break

    if not detailed_problem_analysis:
        try:
            response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, temperature=0)
            response = response.replace('```json', '').strip('```').strip()
            detailed_problem_analysis = json.loads(response)

            is_valid, error_msg = validate_response(detailed_problem_analysis)
            if is_valid:
                return detailed_problem_analysis
            else:
                return response

        except Exception as e:
            pass

    return detailed_problem_analysis


def process_create_task(input_dict, enable_pev: bool = True, enable_test_guidance: bool = True):
    problem_statement = input_dict.get("problem_statement", "")
    problem_statement = post_process_instruction(problem_statement)

    detailed_problem_analysis = get_problem_analysis(problem_statement)

    code_skeleton = get_code_skeleton()
    start_time = time.time()
    initial_solution = generate_initial_solution(problem_statement, code_skeleton, detailed_problem_analysis)
    created_files = extract_and_write_files(initial_solution)

    test_cases = LLMTestGenerator.generate_tests(problem_statement, created_files, code_skeleton)
    
    try:
        coverage_analysis = analyze_test_coverage(
            problem_statement,
            test_cases,
            function_metadata=None
        )
        high_severity_gaps = [
            req for req in coverage_analysis['missing_requirements']
            if req.get('severity') == 'high'
        ]
        COVERAGE_THRESHOLD = 0.75
        if coverage_analysis['coverage_score'] < COVERAGE_THRESHOLD:

            augmented_tests = generate_missing_tests(coverage_analysis, test_cases, problem_statement)
            test_cases = augmented_tests

    except Exception as e:
        pass
    test_files = extract_and_write_files(test_cases)

    timeout = DEFAULT_TIMEOUT - (time.time() - start_time) - 60

    patch = fix_task_solve_workflow(
        problem_statement,
        timeout=timeout,
        run_id_1=run_id,
        test_runner=f"unittest",
        test_runner_mode="FILE",
        n_max_steps=60,
        enable_pev=enable_pev,
        enable_test_guidance=enable_test_guidance,
        extra_fix_request=SOLVE_TASK_NON_FUNCTIONAL_TEST_PROMPT
    )

    if patch is None:
        extract_and_write_files(initial_solution)

    tool_manager = EnhancedToolManager()
    patch = tool_manager.get_final_git_patch()
    return patch


def fix_task_solve_workflow(problem_statement: str, *, timeout: int, run_id_1: str, \
                            test_runner: str = "pytest", test_runner_mode: str = "FILE", n_max_steps=MAX_FIX_TASK_STEPS,
                            enable_pev: bool = True, enable_test_guidance: bool = True, extra_fix_request="") -> tuple[
    str, List[str], List[str]]:
    global run_id
    run_id = run_id_1

    pev = PEVWorkflow(enable_pev=enable_pev, enable_test_guidance=enable_test_guidance)

    strategy = pev.run_planning_phase(problem_statement)
    
    # Initialize test-driven guidance
    problem_type = "bug_fix"  # Can be determined from problem_statement analysis
    pev.initialize_guidance(problem_statement, problem_type)

    strategy_guidance = f"\n\nStrategic Plan: {strategy.get('name', 'Default')} - {strategy.get('description', 'Standard approach')}"

    cot = EnhancedCOT(latest_observations_to_keep=30)
    tool_manager = FixTaskEnhancedToolManager(
        available_tools=[
            "get_file_content",
            "save_file",
            "get_approval_for_solution",
            "search_in_all_files_content",
            "search_in_specified_file_v2",
            "start_over",
            "list_directory",
            "get_context_around_line",
            "run_repo_tests",
            "run_code",
            "apply_code_edit",
            "generate_test_function",
            "finish"
        ],
        test_runner=test_runner,
        test_runner_mode=test_runner_mode
    )
    phase_manager = PhaseManager(problem_statement, n_max_steps)
    use_multi_phase = phase_manager.use_multi_phase_workflow()
    system_prompt = FIX_TASK_SYSTEM_PROMPT.format(
        tools_docs=tool_manager.get_tool_docs(),
        format_prompt=FORMAT_PROMPT_V0,
        extra_fix_request=extra_fix_request
    )
    instance_prompt = FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(
        problem_statement=problem_statement
    ) + strategy_guidance

    start_time = time.time()
    logs: List[str] = []
    logs.append(f"cwd: {os.getcwd()}")

    for step in range(n_max_steps):

        if use_multi_phase and step > 0:
            should_transition, new_phase = phase_manager.should_transition(step, cot)
            if should_transition:
                phase_manager.transition_to_phase(new_phase, step)

        if time.time() - start_time > timeout:
            cot.add_action(
                EnhancedCOT.Action(
                    next_thought="global timeout reached",
                    next_tool_name="",
                    next_tool_args={},
                    observation="",
                    is_error=True,
                    inference_error_counter={},
                    request_data=[]
                )
            )
            break

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]

        messages.extend(cot.to_str())
        if use_multi_phase:
            phase_guidance = phase_manager.get_phase_guidance()
            messages.append({"role": "system", "content": phase_guidance})
        
        # Add test-driven guidance
        if enable_pev and enable_test_guidance and pev.guidance:
            last_action = cot.thoughts[-1].next_tool_name if cot.thoughts else None
            last_result = str(cot.thoughts[-1].observation) if cot.thoughts else ""
            test_guidance = pev.get_action_guidance(step, last_action, last_result)
            if test_guidance:
                messages.append({"role": "system", "content": test_guidance})

        messages.append({"role": "system", "content": STOP_INSTRUCTION})

        temperature = 0
        selected_model = GLM_MODEL_NAME
        if cot.is_thought_repeated():
            last_thought = cot.thoughts[-1]
            messages.append(
                {"role": "user", "content": DO_NOT_REPEAT_TOOL_CALLS.format(
                    previous_response=f"next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}"
                )}
            )

            if cot.repeated_thoughts > 1:
                temperature = min(cot.repeated_thoughts / 10, 0.7)
                selected_model = AGENT_MODELS[
                    random.randint(0, len(AGENT_MODELS) - 1)] if cot.repeated_thoughts > 2 else GLM_MODEL_NAME

        try:
            next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages = EnhancedNetwork.inference(
                messages,
                model=selected_model,
                run_id=run_id,
                temperature=temperature
            )
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_msg = f"\n\nERROR: {repr(e)} {traceback.format_exc()}"
            cot.add_action(
                EnhancedCOT.Action(
                    next_thought=error_msg,
                    next_tool_name="",
                    next_tool_args={},
                    observation="",
                    is_error=True,
                    raw_response=raw_text,
                    total_attempts=total_attempts
                ),
                inference_error_counter=error_counter,
                request_data=messages
            )
            break

        try:
            if '"' in next_tool_name or "'" in next_tool_name:
                next_tool_name = next_tool_name.replace('"', '')
                next_tool_name = next_tool_name.replace("'", "")

            next_observation = tool_manager.get_tool(next_tool_name)(
                **next_tool_args
            ) if next_tool_args else tool_manager.get_tool(next_tool_name)()
            cot.add_action(
                EnhancedCOT.Action(
                    next_thought=next_thought,
                    next_tool_name=next_tool_name,
                    next_tool_args=next_tool_args,
                    observation=next_observation,
                    is_error=False,
                    raw_response=raw_text,
                    total_attempts=total_attempts,
                    inference_error_counter=error_counter,
                    request_data=messages
                )
            )
            if use_multi_phase and next_tool_name in ['run_repo_tests', 'apply_code_edit', 'get_approval_for_solution']:
                test_results = {}
                if 'passed' in str(next_observation).lower() or 'failed' in str(next_observation).lower():
                    obs_str = str(next_observation)
                    test_results['observation'] = obs_str[:200]  # First 200 chars

                phase_manager.create_checkpoint(step, test_results)

            # Update test-driven guidance with action results
            if enable_pev and enable_test_guidance and pev.guidance:
                success = "error" not in str(next_observation).lower()
                pev.guidance.update_from_action(next_tool_name, str(next_observation), success)
        except EnhancedToolManager.Error as e:
            import traceback  # Ensure traceback is accessible
            error_msg = f"observation: {e.message}"
            cot.add_action(
                EnhancedCOT.Action(
                    next_thought=next_thought,
                    next_tool_name=next_tool_name,
                    next_tool_args=next_tool_args,
                    observation=error_msg,
                    is_error=True,
                    raw_response=raw_text,
                    total_attempts=total_attempts,
                    inference_error_counter=error_counter,
                    request_data=messages
                )
            )

            if enable_pev and enable_test_guidance and pev.guidance:
                pev.guidance.update_from_action(next_tool_name, error_msg, False)
            continue
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_traceback = traceback.format_exc()
            if isinstance(e, TypeError):
                error_msg = f"observation: {str(e)}"
            else:
                error_msg = f"observation: {repr(e)} {error_traceback}"
            cot.add_action(
                EnhancedCOT.Action(
                    next_thought=next_thought,
                    next_tool_name=next_tool_name,
                    next_tool_args=next_tool_args,
                    observation=error_msg,
                    is_error=True,
                    raw_response=raw_text,
                    total_attempts=total_attempts,
                    inference_error_counter=error_counter,
                    request_data=messages
                )
            )

            if enable_pev and enable_test_guidance and pev.guidance:
                pev.guidance.update_from_action(next_tool_name, error_msg, False)
            continue

        if next_tool_name == "finish":
            break
    else:
        cot.add_action(
            EnhancedCOT.Action(
                next_thought="global timeout reached",
                next_tool_name="",
                next_tool_args={},
                observation="",
                is_error=True
            )
        )
        if n_max_steps < MAX_FIX_TASK_STEPS:
            return None
    if use_multi_phase:

        for phase_info in phase_manager.phase_history:
            phase_name = phase_info['phase']
            steps_used = phase_info['steps_used']
            allocated = phase_manager.step_allocation.get(phase_name, 0)
            efficiency = (steps_used / allocated * 100) if allocated > 0 else 0
        current_phase = phase_manager.current_phase
        steps_in_current = step - phase_manager.phase_start_step
        allocated_current = phase_manager.step_allocation.get(current_phase, 0)

        if steps_in_current > 0:
            efficiency_current = (steps_in_current / allocated_current * 100) if allocated_current > 0 else 0
        completed_phases = [p['phase'] for p in phase_manager.phase_history]
    
    # Finalize test-driven guidance and save learned traces
    if enable_pev and enable_test_guidance and pev.guidance:
        overall_success = next_tool_name == "finish" and not cot.thoughts[-1].is_error if cot.thoughts else False
        pev.guidance.finalize(overall_success)
    
    patch = tool_manager.get_final_git_patch()

    return patch


def get_code_skeleton() -> str:
    result = ""
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    content = f.read()
                result += f"{file}\n{{\n{content}\n}}\n\n"

    return result


def get_directory_tree(start_path: str = '.') -> str:

    tree_lines = []

    def add_directory_tree(path: str, prefix: str = "", is_last: bool = True, is_root: bool = False):
        """Recursively build the tree structure"""
        try:
            dir_name = os.path.basename(path) if path != '.' else os.path.basename(os.getcwd())
            if not is_root:
                connector = "└── " if is_last else "├── "
                tree_lines.append(f"{prefix}{connector}{dir_name}/")
            try:
                items = os.listdir(path)
                items = [item for item in items if not item.startswith('.')]
                items.sort()
                dirs = []
                files = []
                for item in items:
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        dirs.append(item)
                    else:
                        files.append(item)
                for i, dir_name in enumerate(dirs):
                    dir_path = os.path.join(path, dir_name)
                    is_last_dir = (i == len(dirs) - 1) and len(files) == 0
                    new_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
                    add_directory_tree(dir_path, new_prefix, is_last_dir, False)
                for i, file_name in enumerate(files):
                    is_last_file = i == len(files) - 1
                    connector = "└── " if is_last_file else "├── "
                    tree_lines.append(
                        f"{prefix}{'' if is_root else ('    ' if is_last else '│   ')}{connector}{file_name}"
                    )

            except PermissionError:
                error_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
                tree_lines.append(f"{error_prefix}└── [Permission Denied]")

        except Exception as e:
            tree_lines.append(f"{prefix}└── [Error: {str(e)}]")

    add_directory_tree(start_path, is_root=True)
    return "\n".join(tree_lines)


def find_readme(file_path: str, repo_path: str) -> Optional[str]:
    """Find README file by traversing up from the given path."""
    current_dir = os.path.dirname(file_path)

    while True:
        for readme_name in ['README.md', 'README.rst']:
            readme_path = os.path.join(current_dir, readme_name)
            if os.path.exists(readme_path):
                return readme_path
        if current_dir == repo_path:
            break
        current_dir = os.path.dirname(current_dir)

    return None


def find_test_runner(readme_file_path: Optional[str] = None):
    if not readme_file_path:
        return "pytest"
    try:
        with open(readme_file_path, "r", encoding='utf-8') as f:
            readme_content = f.read()

        response = EnhancedNetwork.make_request(
            [
                {"role": "system", "content": FIND_TEST_RUNNER_PROMPT},
                {"role": "user", "content": readme_content}
            ], model=DEEPSEEK_MODEL_NAME
        )
        return response.strip() or "pytest"
    except Exception as e:
        return "pytest"


def filepath_to_module(file_path: str, repo_path: str, test_runner: str) -> str:
    """Convert file path to Python module notation."""
    root_path = os.path.abspath(repo_path)
    abs_filepath = os.path.abspath(file_path)
    module_path = os.path.splitext(abs_filepath)[0]
    if module_path.startswith(root_path):
        module_path = module_path[len(root_path):].lstrip(os.path.sep)
    test_runner_dir = os.path.dirname(test_runner)
    if test_runner_dir and module_path.startswith(test_runner_dir):
        module_path = module_path[len(test_runner_dir):].lstrip(os.path.sep)

    return module_path.replace(os.path.sep, '.')


def clean_filepath(file_path: str, repo_path: str, test_runner: str) -> str:
    root_path = os.path.abspath(repo_path)
    abs_filepath = os.path.abspath(file_path)

    module_path = os.path.splitext(abs_filepath)[0]
    if module_path.startswith(root_path):
        module_path = module_path[len(root_path):].lstrip(os.path.sep)

    test_runner_dir = os.path.dirname(test_runner)
    if test_runner_dir and module_path.startswith(test_runner_dir):
        module_path = module_path[len(test_runner_dir):].lstrip(os.path.sep)

    return module_path


def get_test_runner_mode(test_runner: str):
    if test_runner == 'pytest':
        return "FILE"

    try:
        with open(test_runner, "r", encoding='utf-8') as f:
            runner_content = f.read()

        response = EnhancedNetwork.make_request(
            [
                {"role": "system", "content": TEST_RUNNER_MODE_PROMPT},
                {"role": "user", "content": runner_content}
            ], model=DEEPSEEK_MODEL_NAME
        )
        return response.strip() or "FILE"
    except Exception as e:
        return "FILE"


def count_test_cases(file_path: str) -> int:
    """Count the number of test cases (functions starting with 'test_') in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        import re
        test_functions = re.findall(r'^\s*def\s+test_\w+', content, re.MULTILINE)
        return len(test_functions)

    except (FileNotFoundError, UnicodeDecodeError):
        return 0


def get_test_runner_and_mode():
    test_runner = "pytest"
    test_runner_mode = "FILE"
    test_files = []  # Initialize the test_files list
    test_file_path = None

    for root, _, files in os.walk('.'):
        for file in files:
            if 'test_' in file and file.endswith('.py'):
                test_files.append(os.path.join(root, file))

    test_files.sort(key=len)

    for path in test_files:
        if count_test_cases(path) > 5:
            test_file_path = path
            break

    if not test_file_path:
        return "pytest", "FILE"
    readme_file_path = find_readme(test_file_path, '.')

    if readme_file_path:
        test_runner = find_test_runner(readme_file_path)
        test_runner_mode = get_test_runner_mode(test_runner)

    return test_runner, test_runner_mode


def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", enable_pev: bool = True, enable_test_guidance: bool = True):
    """Legacy interface wrapper for backwards compatibility."""
    global DEFAULT_PROXY_URL, DEFAULT_TIMEOUT, run_id
    run_id = os.getenv("RUN_ID", "")
    repo_dir = os.path.abspath(repo_dir)

    sys.path.insert(0, repo_dir)

    if os.path.exists(repo_dir):
        os.chdir(repo_dir)

    ensure_git_initialized()

    set_env_for_agent()

    try:
        problem_type = check_problem_type(input_dict.get("problem_statement"))

        if problem_type == PROBLEM_TYPE_FIX:
            result = process_fix_task(input_dict, enable_pev=enable_pev, enable_test_guidance=enable_test_guidance)
        else:
            result = process_create_task(input_dict, enable_pev=enable_pev, enable_test_guidance=enable_test_guidance)
    except Exception as e:
        result = process_fix_task(input_dict, enable_pev=enable_pev, enable_test_guidance=enable_test_guidance)

    os.system("git reset --hard")

    return result
