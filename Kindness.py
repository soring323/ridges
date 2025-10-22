from __future__ import annotations
import ast
import json
import os
import requests
import subprocess
import ast, sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from json import JSONDecodeError
import re
import inspect
import random
from enum import Enum
import json
import csv
import logging
from uuid import uuid4

DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2000"))

PROBLEM_TYPE_CREATE = "CREATE"
PROBLEM_TYPE_FIX = "FIX"

GLM_MODEL_NAME = "zai-org/GLM-4.5-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
AGENT_MODELS=[GLM_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME, QWEN_MODEL_NAME]
MAX_FIX_TASK_STEPS = 400

FIX_TASK_SYSTEM_PROMPT = textwrap.dedent("""
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
- Prefer using `search_in_all_files_content` to enumerate matches across the codebase and `search_in_specified_file_v2` to drill into each file; iterate until no applicable occurrences remain.
- Re-run tests only after covering all discovered occurrences to avoid partial fixes.

## Test generation guidance:
- Use `generate_test_function(file_path, test_function_code, position)` after discovering the most relevant existing test file.
- Prefer `position="auto"` which inserts after imports or before the `if __name__ == "__main__":` block when present, falling back to append.
- Generated tests (new files or appended functions) are tracked and excluded from the final patch automatically, so they must not show up in the final diff.
- Keep generated tests minimal and focused on the bug and its edge cases.
- Note that current test functions should be passed originally and generated test function is FAIL_TO_PASS.

You have access to the following tools:-
{tools_docs}

{format_prompt}
""")

STOP_INSTRUCTION=textwrap.dedent("""
# 🎨 
DO NOT generate `observation:` in your response. It will be provided by user for you.
Generate only SINGLE triplet of `next_thought`, `next_tool_name`, `next_tool_args` in your response.
""")

PROBLEM_TYPE_CHECK_PROMPT = textwrap.dedent(
'''
You are the problem type checker that will categories problem type into:

1. CREATE: If the problem statement is about creating a new functionality from scratch.
2. FIX: If the problem statement is about fixing a bug, creating a new functionality or improving the existing codebase.

Only respond with the "FIX" or "CREATE".
'''
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

for h in list(logger.handlers):
    logger.removeHandler(h)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
run_id=None

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


DO_NOT_REPEAT_TOOL_CALLS=textwrap.dedent("""
You're not allowed to repeat the same tool call with the same arguments.
Your previous response: 
{previous_response}

Try to use something different!
""")

GENERATE_INITIAL_TESTCASES_PROMPT = textwrap.dedent("""
You are an expert Python testcase developer. Your task is to generate a complete testcases for the given problem statement.

Important things:
1. Test functions declared in code skeleton, don't customized those prototypes.
2. Read the problem statement carefully and deeply and generate testcases that exactly match the rules, mathmatical fomulas, algorithms, data, and workflow in it.
3. Do not generate testcases that are not mentioned in problem statement
4. Minimize all testcases as you have context and generation limit

Strict Requirements:
1. Output the full content of Python test files along with their file names. You **MUST** output the **file name** along with file content.
2. Do not include explanations, comments, or markdown formatting.
3. Use only standard Python (no external libraries).

Response Examples:
```python
test_a.py
contents of test_a.py

test_b.py
contents of test_b.py
```
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

GENERATE_INITIAL_SOLUTION_PROMPT = textwrap.dedent("""
You are an expert Python developer. Your task is to generate a complete, working Python solution for the given problem statement.

Strict Requirements:
1. Output the full content of Python files along with their file names.
2. Do not include explanations, comments, or markdown formatting.
3. Use only standard Python (no external libraries).
4. Implement all required classes and functions exactly with the same names as in the initial code stub.
5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
7. The solution must be executable as-is with no placeholders or TODOs.

Return only the final python files code.

Response Examples:
```python
a.py
{content}

b.py
{content}
```
"""
)


GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT = textwrap.dedent(
"""
You are an expert Python testcase developer. Your task is to generate a complete testcases for the given problem statement.

Important things:
1. Test functions declared in code skeleton, don't customized those prototypes.
2. Read the problem statement carefully and deeply and generate testcases that exactly match the rules, mathmatical fomulas, algorithms, data, and workflow in it.
3. Do not generate testcases that are not mentioned in problem statement
4. Minimize all testcases as you have context and generation limit

Strict Requirements:
1. Output the full content of Python test files along with their file names. You **MUST** output the **file name** along with file content.
2. Do not include explanations, comments, or markdown formatting.
3. Use only standard Python (no external libraries).

Response Examples:
```python
test_a.py
contents of test_a.py

test_b.py
contents of test_b.py
```
"""
)

GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT = textwrap.dedent(
"""
You are an expert Python developer. Your task is to generate a complete, working Python solution for the given problem statement.

Strict Requirements:
1. Output the full content of Python files along with their file names. You **MUST** output the **file name** along with file content.
2. Do not include explanations, comments, or markdown formatting.
3. Use only standard Python (no external libraries).
4. Implement all required classes and functions exactly with the same names as in the initial code stub.
5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
7. The solution must be executable as-is with no placeholders or TODOs.
8. If problem statement doesn't explicitely requires a list of strings as a response, do not use list of strings for multiline text problems, just use raw string format.
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

FIX_TASK_FOR_FIXING_SYSTEM_PROMPT = textwrap.dedent("""

# YOUR PRIMARY MISSION

You are a CODEBASE CLEANER. Your goal: **ZERO FAILING TESTS in the repository**.
The problem statement describes a bug - this is your **starting point**, not your only task.
Your TRUE mission: Make the codebase clean with NO bugs (no failing tests).

**FUNDAMENTAL PRINCIPLE: TEST CASES ARE THE SOURCE OF TRUTH**
- Test cases define what is correct behavior
- Your implementation must match test expectations
- NEVER modify test cases to match your implementation
- If tests fail, fix your code, not the tests

## What "Success" Means

✅ **SUCCESS** = ALL existing repository tests pass (ZERO failures)
❌ **FAILURE** = ANY repository test fails

The problem statement bug is just ONE bug. If your fix creates OTHER bugs (failing tests), you have FAILED.

## Your Validation Method

**PRIMARY (The Only Thing That Matters):**
- `run_repo_tests_for_fixing(file_paths=['./tests/...'])` on existing test files
- These tests were passing BEFORE you started
- They MUST pass AFTER you finish
- This is your SUCCESS CRITERIA

**OPTIONAL (For Specific Cases Only):**
- `generate_validation_test()` + `run_validation_test()` 
- ONLY use if the problem statement bug has no existing test
- This validates your specific fix works
- BUT: Even if this passes, you're NOT done until run_repo_tests_for_fixing passes

## The Critical Truth

Imagine: You fix the problem statement bug ✓ BUT there is a failing tests ✗
Result: You FAILED. The codebase is WORSE than before.

## Your Workflow
1. **Read problem statement** - understand the bug
2. **Find and fix** the root cause
3. **Run run_repo_tests_for_fixing()** - validate NO tests are broken
4. **If ANY test fails** - you broke it, fix it or revise your approach
5. **(Optional) Generate validation test** - only if problem statement has no test
6. **Call submit_solution** - only when ALL tests pass

## The Rules (No Exceptions)

1. After code change → run run_repo_tests_for_fixing()
2. If even ONE test fails → YOU broke it (no excuses about "unrelated")
3. Cannot call submit_solution with failing tests → it will be REJECTED automatically
4. "Problem statement fixed but tests fail" → YOU FAILED
5. "My validation test passed" → IRRELEVANT if run_repo_tests_for_fixing fails

## Common Wrong Excuses (DO NOT MAKE THESE)

❌ "These failing tests don't directly involve what I was asked to fix"
❌ "These tests are edge cases not in the problem statement"
❌ "While there are X failing tests, these appear unrelated"
❌ "The problem statement is fixed, so other failures don't matter"
❌ "My validation test passed, so my fix is correct"
❌ "I need to update the failing tests to match my implementation"
❌ "The existing tests expect the old behavior, but the problem statement wants new behavior"

## Multi-file awareness (critical):
- Tests and patch contexts may span multiple files. Do not stop after the first similar match or applied fix.
- Keep searching the repository after each match and apply consistent changes to every relevant file before finishing.
- Prefer using `search_in_all_files_content` to enumerate matches across the codebase and `search_in_specified_file_v2` to drill into each file; iterate until no applicable occurrences remain.
- Re-run tests only after covering all discovered occurrences to avoid partial fixes.

## CRITICAL RULE: TEST CASES ARE THE ABSOLUTE TRUTH

🚨 **NEVER EDIT EXISTING TEST CASES** 🚨

- Test cases are the SOURCE OF TRUTH - they define what is correct behavior
- If your implementation doesn't match the test expectations, FIX YOUR IMPLEMENTATION
- If test cases fail after your changes, you broke something - REVERT and try a different approach
- The problem statement is just a starting point - test cases are the actual requirements
- Even if the problem statement seems to contradict test cases, TRUST THE TEST CASES
- Test cases were written by the original developers who understand the intended behavior
- Your job is to make the code work according to the test cases, not to change the test cases

## Important Guidelines

- NEVER edit existing test files
- Code must be backward compatible unless stated otherwise
- Search entire codebase for similar patterns - fix ALL occurrences
- Re-run tests after change
- submit_solution will automatically verify tests - you CANNOT bypass this
- If you can't fix failing tests, your approach is WRONG - revise it

You have access to the following tools:-
{tools_docs}

{format_prompt}
""")

FIX_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
# Now let's start. Here is the problem statement:
{problem_statement}
""")

FIND_TEST_RUNNER_PROMPT = textwrap.dedent("""\
You are a helpful assistant that can find the test runner for a given repository.
- The test runner is the file that can run the individual test files and test cases. (e.g. pytest, unittest, etc.)
- Do not use the test runner to run test for whole repository or test setup.
- Read the README file and find the test runner. If there is no test runner, return pytest.
- Output format should be as the following. No other texts are allowed.
abc/test.py
""")

TEST_RUNNER_MODE_PROMPT = textwrap.dedent("""\
You are a helpful assistant that determines the mode of the test runner.
Read the test runner file and determine if it requires a module or a file path to run the test.
Output should be one of MODULE or FILE, No other texts are allowed.
- MODULE: When the test runner requires a module path to run the test.
- FILE: When the test runner requires a file path to run the test (e.g. pytest, unittest, py.test, etc.).
""")

FORMAT_PROMPT_V0=textwrap.dedent("""
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
""")

class EnhancedCOT:
    class Action:            
        def __init__(self, next_thought: str, next_tool_name: str, next_tool_args: dict, observation: list|tuple|str,is_error:bool=False,raw_response:str=None,total_attempts:int=0,inference_error_counter:dict=None,request_data:list=None):
            self.next_thought=next_thought
            self.next_tool_name=next_tool_name
            self.next_tool_args=next_tool_args
            self.observation=";".join(observation) if isinstance(observation,list) else observation
            self.is_error=is_error
            self.raw_response=raw_response
            self.total_attempts=total_attempts
            self.inference_error_counter=inference_error_counter
            self.request_data=request_data
            self.is_deleted=False
    def __init__(self,latest_observations_to_keep=5):
        self.thoughts: list[EnhancedCOT.Action] = []
        self.latest_observations_to_keep=latest_observations_to_keep

    def add_action(self, action: EnhancedCOT.Action) -> bool: # don't add if thought is repeated
        self.thoughts.append(action)
        return True
        
    def is_thought_repeated(self)->bool:
        if len(self.thoughts) < 2:
            return False
        last = self.thoughts[-1]
        prev = self.thoughts[-2]
        if last.next_tool_name == prev.next_tool_name and last.next_tool_args == prev.next_tool_args:
            return True
        return False
    
    def to_str(self):
        messages=[]
        for i,thought in enumerate(self.thoughts):
            if thought.is_deleted:
                continue
            if i<len(self.thoughts)-self.latest_observations_to_keep:
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
                user_str=( f"observation: {'error ocurred.' if thought.is_error else ''} "
                    f"output omitted ({_obs_len}) lines\n")
            else:
                if thought.is_error is None or i==len(self.thoughts)-1:
                    assistant_str=f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                    if isinstance(thought.observation, (list, tuple)):
                        try:
                            obs_render=json.dumps(list(thought.observation), ensure_ascii=False)
                        except Exception:
                            obs_render=str(thought.observation)
                    else:
                        obs_render=str(thought.observation)
                    user_str=f"observation: {obs_render}"
                else:
                    if self.thoughts[-1].is_error==None and thought.is_error!=None:
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
                        user_str=(
                            f"observation: error ocurred. detailed output omitted "
                            f"({_obs_len}) lines\n"
                        )
                    else:
                        assistant_str=f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                        if isinstance(thought.observation, (list, tuple)):
                            try:
                                obs_render=json.dumps(list(thought.observation), ensure_ascii=False)
                            except Exception:
                                obs_render=str(thought.observation)
                        else:
                            obs_render=str(thought.observation)
                        user_str=f"observation: {obs_render}"
            messages.append({"role":"assistant","content":assistant_str})
            messages.append({"role":"user","content":user_str})
        return messages

class EnhancedToolManager:
    logs = []
    TOOL_LIST = {}

    class Error(Exception):
        class ErrorType(Enum):
            SYNTAX_ERROR=1
            RUNTIME_ERROR=2
            TIMEOUT=3
            FILE_NOT_FOUND=4
            SEARCH_TERM_NOT_FOUND=5
            UNKNOWN=6
            THIRD_PARTY_DEPENDENCIES=7
            MULTIPLE_SEARCH_RESULTS_FOUND=8
            BUG_REPORT_REQUIRED=9
            INVALID_RESPONSE_FORMAT=10
            INVALID_TOOL_NAME=11
            INVALID_FILE_PATH=12
            INVALID_TOOL_CALL=13
            IMPORT_ERROR=14
            
        def __init__(self,error_type:ErrorType,message:str):    
            self.error_type=error_type
            self.message=message

    def tool(fn):
        def wrapper(self, *args, **kwargs):
            self.tool_invocations[fn.__name__]+=1
            try:
                return fn(self, *args, **kwargs)
            except EnhancedToolManager.Error as e:
                self.tool_failure[fn.__name__][e.error_type]+=1
                return e.message

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper.__signature__ = inspect.signature(fn)
        wrapper.__annotations__ = fn.__annotations__.copy()
        wrapper.is_tool=True
        return wrapper

    def __init__(self, **kwargs):
        pass
    
    @classmethod
    def tool_parsing(cls,fn):
        tool_schemas = None
        name = fn.__name__
        doc_fn = fn.__doc__ or ""
        doc=doc_fn.split("Arguments:")[0]
        output_description=doc_fn.split("Output:")
        if len(output_description)>1:
            output_description="Output: "+output_description[1].strip()
            doc=doc+"\n\n"+output_description
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for param in sig.parameters.values():
            if param.name == 'self':
                continue
            if param.default is param.empty and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                required.append(param.name)
            type_hint = str(param.annotation) if param.annotation != param.empty else "string"
            param_description=re.search(f"{param.name}:([^\n]+)",doc_fn)
            if param_description:
                param_description=param_description.group(1)
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
        tool_schemas={
            "name": name,
            "description": doc.strip(),
            "input_schema": parameters
        }
        return tool_schemas

    @classmethod
    def get_tool_args_for_tool(self,tool_name:str,required_only:bool=False)->list[str]:
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        if not required_only: 
            return list(self.TOOL_LIST[tool_name]['input_schema']['properties'].keys())
        else:
            return self.TOOL_LIST[tool_name]['input_schema']['required']

    def get_tool_docs(self)->str:
        return '\n\n'.join([json.dumps(tool_metadata, ensure_ascii=False) for _,tool_metadata in self.TOOL_LIST.items()])

    def get_tool(self,tool_name:str):
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        tool_method = getattr(self, tool_name, None)
        if tool_method is None or not callable(tool_method):
            return f"Error: tool '{tool_name}' does not exist. Please use one of the following tools: {', '.join(self.TOOL_LIST.keys())}"
        return tool_method
    
    def _check_syntax_error(self,content:str,file_path:str="<unknown>")->bool:
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            logger.error(f"Syntax error: {e}")
            return True, EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Syntax error. {str(e)}")

    def _save(self,file_path: str, content: str)->str:
        is_syntax_error, error = self._check_syntax_error(content)
        if not is_syntax_error:
            with open(file_path, "w") as file:
                file.write(content)
            return f"File {file_path} saved successfully"
        else:
            logger.error(f"Error saving file: {error.message}")
            error.message="Error saving file. "+error.message
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,error.message)

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
            print("Generating git patch...")
            output = subprocess.run(["bash", "-c", command], timeout=30, capture_output=True, text=True)
            
            return output.stdout
        except Exception as e:
            logger.error(f"Error generating git patch: {e}")
            return f"Error generating git patch: {e}"

class EnhancedNetwork:
    class ErrorType(Enum):
        EMPTY_RESPONSE=1
        RESERVED_TOKEN_PRESENT=2
        RATE_LIMIT_EXCEEDED=3
        INVALID_RESPONSE_FORMAT=4
        TIMEOUT=5
        UNKNOWN=6
        NETWORK_ERROR=7
        AUTHENTICATION_ERROR=8
        RESOURCE_EXHAUSTED=9
    
    @classmethod
    def is_valid_response(cls,raw_text:str)->bool:
        if type(raw_text) is dict and raw_text.get("error",None) is not None and raw_text.get("error")!="":
            return False,cls.ErrorType.EMPTY_RESPONSE.name
        if not raw_text.strip().endswith("}") and not raw_text.strip().endswith("}]"):
            return False, "Incomplete response, your response must be shorter to fit within context limit"
        if len(raw_text)==0:
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
    def get_error_counter(cls)->dict[str,int]:
        return {
            k:0 for k in cls.ErrorType.__members__
        }   

    @classmethod
    def fix_json_string_with_llm(cls,json_string:str,attempt:int=0)->dict:
        messages=[
            {"role":"system", "content":"Fix the json string sent by the user.  Reply only with the json string and nothing else."},
            {"role":"user", "content":json_string}
        ]
        response=cls.make_request(messages, model=DEEPSEEK_MODEL_NAME)
        try:
            response=response.replace('```json','').strip('```')
            response=json.loads(response)
            return response
        except JSONDecodeError as e:
            logger.error(f"Error fixing json string: {e},trying again..")
            logger.error(f"json string is :{json_string}")
            logger.error(f"LLM response is :{response}")
            return None
    
    @classmethod
    def make_request(cls,messages:list,model:str,attempt:int=0, temperature:float=0.0)->str:
        global run_id
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
        print("[REQUEST] run_id:", run_id)

        request_data = {
                "run_id": run_id if run_id else str(uuid4()),
                "messages": messages,
                "temperature": temperature,
            }

        headers = {
            "Content-Type": "application/json"
        }
        request_data['model'] = model
        
        try:
            response = requests.post(url, json=request_data, timeout=120, headers=headers)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout after 120 seconds for model {model}")
            return f"ERROR: Request timeout for model {model}"
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for model {model}: {e}")
            return f"ERROR: Connection failed for model {model}"
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for model {model}: {e}")
            return f"ERROR: HTTP error {e.response.status_code} for model {model}"
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for model {model}: {e}")
            return f"ERROR: Request failed for model {model}"
        
        try:
            response_json = response.json()
        except JSONDecodeError as e:
            logger.error(f"Invalid JSON response for model {model}: {e}")
            logger.error(f"Response content: {response.text[:500]}...")
            return f"ERROR: Invalid JSON response for model {model}"
        
        try:
            is_oai_interface= type(response_json) is dict and response_json.get('choices') is not None and len(response_json.get('choices'))>0 and response_json.get('choices')[0].get('message') is not None
            if is_oai_interface:
                raw_text=response_json['choices'][0]['message']['content']
            else:
                if type(response_json) is str:
                    raw_text=response_json.strip("\n").strip()
                else:
                    raw_text=response_json
            if type(raw_text) is not dict:
                raw_text=raw_text.lstrip()
            return raw_text
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error parsing response structure for model {model}: {e}")
            logger.error(f"Response JSON: {response_json}")
            return f"ERROR: Invalid response structure for model {model}"
        except Exception as e:
            logger.error(f"Unexpected error processing response for model {model}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"ERROR: Unexpected error for model {model}"

    @classmethod
    def _request_next_action_with_retry(cls, messages: dict, 
                            model: str,
                            max_retries: int = 5, 
                            base_delay: float = 1.0,
                            temperature: float = 0.0) -> str:
        
        raw_text='not defined'
        error_counter=cls.get_error_counter()
        next_thought, next_tool_name, next_tool_args = None, None, None
        total_attempts=0
        for attempt in range(max_retries):
            try:
                total_attempts+=1
                index = AGENT_MODELS.index(model) if model in AGENT_MODELS else -1
                raw_text=cls.make_request(messages,model=AGENT_MODELS[(index + attempt)%len(AGENT_MODELS)], temperature=temperature)
                is_valid,error_msg=cls.is_valid_response(raw_text)
                if not(is_valid):
                    raise Exception(error_msg)
                    
                next_thought, next_tool_name, next_tool_args,error_msg = cls.parse_response(raw_text)
                if error_msg:
                    raise Exception(error_msg)
                break
            except Exception as e:
                error_body = str(e)
                logger.error(f"Error: {error_body}")
                if attempt < max_retries:
                    delay = base_delay
                    logger.info(error_body)
                    logger.error("--------------------------------")
                    logger.error(f"response: {raw_text}")
                    logger.error("--------------------------------")
                    logger.info(f"[agent] Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})") 
                    if "RATE_LIMIT_EXCEEDED" in error_body:
                        error_counter[cls.ErrorType.RATE_LIMIT_EXCEEDED.name]+=1
                    elif "RESERVED_TOKEN_PRESENT" in error_body:
                        error_counter[cls.ErrorType.RESERVED_TOKEN_PRESENT.name]+=1
                    elif "EMPTY_RESPONSE" in error_body:
                        error_counter[cls.ErrorType.EMPTY_RESPONSE.name]+=1
                    elif "TIMEOUT" in error_body:
                        error_counter[cls.ErrorType.TIMEOUT.name]+=1
                    elif "Invalid JSON" in error_body:
                        error_counter[cls.ErrorType.INVALID_RESPONSE_FORMAT.name]+=1
                    elif "Invalid response" in error_body:
                        error_counter[cls.ErrorType.INVALID_RESPONSE_FORMAT.name]+=1
                    else:
                        error_counter[cls.ErrorType.UNKNOWN.name]+=1
                    if "RATE_LIMIT_EXCEEDED" not in error_body and "RESERVED_TOKEN_PRESENT" not in error_body and "EMPTY_RESPONSE" not in error_body and  "TIMEOUT" not in error_body:
                        messages.append({"role":"assistant","content":raw_text})
                        messages.append({"role":"user","content":"observation: "+error_body})
                    time.sleep(random.uniform(1.2*delay, 1.5*delay))
                    continue
                else:
                    error_counter[cls.ErrorType.TIMEOUT.name]+=1
                    raise RuntimeError(error_body)
        
        return next_thought, next_tool_name, next_tool_args,raw_text,total_attempts,error_counter,messages
    
    @classmethod
    def parse_malformed_json(cls,arguments:list[str], json_string:str)->dict | str:    
        pattern = ''
        for i, k in enumerate(arguments):
            pattern += f'"{k}": (.*)'
            if i != len(arguments) - 1:
                pattern += r',\s*'

        match=re.search(pattern, json_string)

        if not match:
            return f"Error: {json_string} can not match pattern {pattern}"
        
        result_json={}
        for i in range(len(arguments)):
            value=match.group(i+1)
            value=value.strip()
            if value.startswith('"') and value.endswith('"'):
                value=value[1:-1]
            #value=value.replace('"', '\\"')
            value=value.replace('\\n','\n')
            result_json[arguments[i]]=value
        return result_json
    
    @classmethod
    def parse_next_tool_args(cls,tool_name:str, next_tool_args: str)->dict | str:
        '''
        parse string to json, fix unecaped " in values like this: '{"a": "text "text2" text3 "text4"", "b": "text3"}'
        returns json or error message
        '''
        next_tool_args=next_tool_args.replace('```json','').strip('```')
        error_msg=''

        try:
            next_tool_args = Utils.load_json(next_tool_args.strip())
        except JSONDecodeError as e:
            error_msg=f"Invalid JSON: {next_tool_args}"    
            try:
                next_tool_args = cls.parse_malformed_json(EnhancedToolManager.get_tool_args_for_tool(tool_name,required=True), next_tool_args)
            except EnhancedToolManager.Error as e:
                raise Exception(e.message)
            except Exception as e:
                raise Exception(error_msg)
        return next_tool_args

    @classmethod
    def inference(cls, messages: List[Dict[str, Any]], model: str, run_id: str = str(uuid4()), temperature:float=0.0) -> dict:
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

        next_thought,next_tool_name,next_tool_args,raw_text,total_attempts,error_counter,messages = cls._request_next_action_with_retry(cleaned_msgs, model=model, temperature=temperature)    
        return next_thought,next_tool_name,next_tool_args,raw_text,total_attempts,error_counter,messages
    
    @classmethod
    def sanitise_text_resp(cls,text_resp:str)->str:
        text_resp=re.sub("[\'\"]*next_thought[\'\"]*:","next_thought:",text_resp)
        text_resp=re.sub("[\'\"]*next_tool_name[\'\"]*:","next_tool_name:",text_resp)
        text_resp=re.sub("[\'\"]*next_tool_args[\'\"]*:","next_tool_args:",text_resp)
        text_resp=re.sub("[\'\"]*observation[\'\"]*:","observation:",text_resp)
        if "next_thought" not in text_resp and "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_tool_name:")<text_resp.find("next_tool_args:") and text_resp.find("next_tool_name:")>10:
            logger.info(f"next_thought not found in {text_resp[:50]}, adding it")
            text_resp="next_thought: "+text_resp
        if "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_tool_name:")<text_resp.find("next_tool_args:"):
            next_tool_name=text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n").strip("\'").strip("\"").strip()
            text_resp=re.sub(f"next_tool_name:[\'\" ]*{next_tool_name}[\'\" ]*","next_tool_name: "+next_tool_name,text_resp)    
        return text_resp

    @classmethod
    def parse_response(cls,text_resp: str)->tuple[str, Any, Any]:
        error_msg=None
        text_resp = text_resp.strip()
        text_resp=text_resp.split("observation:")[0]
        text_resp=text_resp.strip().strip("\n")
        text_resp=cls.sanitise_text_resp(text_resp)
        if "next_thought:" in text_resp and "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_thought:")<text_resp.find("next_tool_name:") and text_resp.find("next_tool_name:")<text_resp.find("next_tool_args:"):
            next_thought=text_resp.split("next_thought:")[1].split("next_tool_name:")[0].strip().strip("\n")
            next_tool_name_raw=text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n")
            next_tool_args_raw=text_resp.split("next_tool_args:")[1].strip().split("next_thought:")[0].strip().strip("\n")
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
                error_msg=f"Invalid JSON: {str(e)}"
                Utils.log_to_failed_messages(text_resp)
                
        else:
            if "next_thought:" not in text_resp:
                error_msg="Invalid response. next_thought not found"
            elif "next_tool_name:" not in text_resp:
                error_msg="Invalid response. next_tool_name not found"
            elif "next_tool_args:" not in text_resp:
                error_msg="Invalid response. next_tool_args not found"
            elif text_resp.find("next_thought:")>text_resp.find("next_tool_name:"):
                error_msg="Invalid response. next_thought is after next_tool_name"
            elif text_resp.find("next_tool_name:")>text_resp.find("next_tool_args:"):
                error_msg="Invalid response. next_tool_name is after next_tool_args"
            else:
                logger.error(f"We have no clue why parsing failed. Please check this \n{text_resp}\n")
            Utils.log_to_failed_messages(text_resp)
            return None,None,None,error_msg

        if len(next_tool_name) == 1:
            return next_thought, next_tool_name[0], next_tool_args[0], error_msg
            
        return next_thought, next_tool_name, next_tool_args,error_msg

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
        body = "\n".join(lines[line_number-1:end_line_number])
        
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

class ClassVisitor(ast.NodeVisitor):
    def __init__(self, file_content: str):
        self.classes = {}
        self.file_content = file_content

    def visit_ClassDef(self, node):
        line_number = node.lineno
        if isinstance(node.decorator_list, list) and len(node.decorator_list) > 0:
            line_number = node.decorator_list[0].lineno
        end_line_number = line_number
        if isinstance(node.body, list) and len(node.body) > 0:
            end_line_number = node.body[-1].lineno
        lines = self.file_content.split("\n")
        body = "\n".join(lines[line_number-1:end_line_number])
        self.classes[node.name] = {
            "body": body,
            "line_number": line_number
        }
        self.generic_visit(node)

class Utils:
    @classmethod
    def limit_strings(cls,strings: str, n=1000)->str:
        '''
        Limit the number of strings to 1000
        '''
        strings_list=strings.split("\n")
        if len(strings_list)>n:
            return "\n".join(strings_list[:n])+"\n..." + f"({len(strings_list)-n} more lines)"
        else:
            return strings
    @classmethod
    def load_json(cls,json_string:str)->dict:
        try:
            return json.loads(json_string)
        except Exception as e:
            try:
                return eval(json_string)
            except Exception as e:
                logger.info(f"unable to fix manually, trying with llm")
                fixed_json=EnhancedNetwork.fix_json_string_with_llm(json_string)
                if fixed_json:
                    return fixed_json
                else:
                    raise JSONDecodeError("Invalid JSON", json_string, 0)
                
    @classmethod
    def log_to_failed_messages(cls,text_resp:str):
        with open("../failed_messages.csv","a") as f:
                writer=csv.writer(f)
                writer.writerow([text_resp])

class FixTaskEnhancedToolManager(EnhancedToolManager):

    def __init__(self, available_tools: Optional[list[str]] = [], test_runner: str = "pytest", test_runner_mode: str = "FILE"):
        self.new_files_created=[]
        self.is_solution_approved=False
        self.test_runner=test_runner
        self.test_runner_mode=test_runner_mode
        self.generated_test_files=[]

        for cls in self.__class__.__mro__:
            for name, attr in cls.__dict__.items():
                if getattr(attr, "is_tool", False) and name not in self.TOOL_LIST:
                    if available_tools is not None and name not in available_tools: # if available_tools is provided, only include tools in the list
                        continue
                    self.TOOL_LIST[name] = self.__class__.tool_parsing(attr)
                
        self.tool_failure={
            k:{j:0 for j in self.Error.ErrorType.__members__} for k in self.TOOL_LIST.keys()
        }

        self.tool_invocations={
          k:0 for k in self.TOOL_LIST.keys()
        }

    def check_syntax_error(self,content:str,file_path:str="<unknown>")->bool:
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            logger.error(f"Syntax error: {e}")
            return True, EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Syntax error. {str(e)}")

    def _extract_test_file_paths_from_output(self, test_output: str) -> List[str]:
        file_paths = []
        seen = set()
        
        pattern = r'File "([^"]+\.py)", line \d+'
        
        for match in re.finditer(pattern, test_output):
            file_path = match.group(1)
            if file_path.startswith('/'):
                try:
                    rel_path = os.path.relpath(file_path)
                    if rel_path not in seen and os.path.exists(rel_path):
                        file_paths.append(rel_path)
                        seen.add(rel_path)
                except ValueError:
                    if file_path not in seen and os.path.exists(file_path):
                        file_paths.append(file_path)
                        seen.add(file_path)
            else:
                if file_path not in seen and os.path.exists(file_path):
                    file_paths.append(file_path)
                    seen.add(file_path)
        
        return file_paths
    
    def _get_file_content(self,file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None,limit:int=5000)->str:
        if search_term is not None and search_term!="":
            logger.debug(f"search_term specified: {search_term}, searching in v2")
            return self.search_in_specified_file_v2(file_path, search_term)
            
        func_ranges=self.get_function_ranges(file_path)
        if search_start_line!=None:
            for start, end, name in func_ranges:
                if start<=search_start_line<=end:
                    if start<search_start_line:
                        logger.debug(f"search start line {search_start_line} is between a function {start}-{end} for function {name}, setting to {start}")
                        search_start_line=start
        if search_end_line!=None:
            for start, end, name in func_ranges:
                if start<=search_end_line<=end:
                    if end>search_end_line:
                        logger.debug(f"search end line {search_end_line} is between a function {start}-{end} for function {name}, setting to {end}")
                        search_end_line=end
        logger.debug(f"search start line: {search_start_line}, search end line: {search_end_line}")
        with open(file_path, "r") as f:
            if search_start_line is not None or search_end_line is not None:
                lines = f.readlines()
                start = max(0, (search_start_line or 1) - 1)  # Convert to 0-based
                end = min(len(lines), search_end_line or len(lines))
                content = ''.join(lines[start:end])
                return f"Lines {start+1}-{end} of {file_path}:\n{content}"
            else:
                content = f.read()

        return Utils.limit_strings(content, n=limit) if limit!=-1  else content
    
    @EnhancedToolManager.tool
    def generate_validation_test(self, test_code: str, description: str = "validation test") -> str:
        '''
        [OPTIONAL] Generate MINIMAL test - just import the function you fixed and call it directly.
        
        RULES - Keep it SIMPLE:
        1. Import the specific function/method you modified
        2. Call it with test inputs
        3. Print or assert the result
        4. That's it!
        DO NOT use TestCase classes
        
        WARNING: This is NOT your success criteria! You MUST run run_repo_tests_for_fixing() and pass ALL existing tests.
        Arguments:
            test_code: MINIMAL Python code (imports + direct function calls + print/assert)
            description: what you're testing
        Output:
            File path of the test.
        '''
        import hashlib
        
        # Validate test_code
        test_code = (test_code or "").strip()
        if not test_code:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,
                "Error: test_code cannot be empty."
            )
        
        # Check syntax
        is_err, err = self.check_syntax_error(test_code)
        if is_err:
            error_msg = err.message if isinstance(err, EnhancedToolManager.Error) else str(err)
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,
                f"Error: validation test has syntax error: {error_msg}"
            )
        
        import time
        timestamp = int(time.time())
        desc_hash = hashlib.md5(description.encode()).hexdigest()[:8]
        filename = f"validation_test_{timestamp}_{desc_hash}.py"
        
        validation_dir = "/tmp/agent_validation_tests"
        os.makedirs(validation_dir, exist_ok=True)
        file_path = os.path.join(validation_dir, filename)
        
        # Write the test file
        with open(file_path, 'w') as f:
            f.write(test_code)
        
        if not hasattr(self, 'validation_test_files'):
            self.validation_test_files = []
        self.validation_test_files.append(file_path)
        
        return f"""✅ Validation test created: {file_path}
Description: {description}

To run this validation test, use:
    run_validation_test(file_path='{file_path}')

⚠️ CRITICAL REMINDERS:
1. This test is for YOUR debugging only - NOT your success criteria
2. Passing this test does NOT mean you're done!
3. You MUST still run run_repo_tests_for_fixing() on existing tests
4. You MUST pass ALL existing repository tests before calling finish_for_fixing
5. This test will NOT be included in the final patch

Think of this as "scratch paper" - helpful for debugging, but the real test is run_repo_tests_for_fixing()!"""

    @EnhancedToolManager.tool
    def get_file_content(self,file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None)->str:
       
        '''
        Retrieves file contents with optional filtering based on search term and line numbers
        Arguments:
            file_path: filesystem path to target file. This file must be python file.
            search_start_line: optional start line number to begin extraction (1-indexed)
            search_end_line: optional end line number to end extraction (1-indexed)
            search_term: optional text pattern to filter matching lines
        '''
        return self._get_file_content(file_path,search_start_line,search_end_line,search_term,limit=5000)
        
    @EnhancedToolManager.tool
    def save_file(self,file_path: str, content: str)->str:
        '''
        Writes text content to specified filesystem location. If there are any syntax errors in the code, it rejects the edit with an error message. Do not use this tool to create test or files to reproduce the error.
        Arguments:
            file_path: target filesystem path
            content: text data to write
        '''
        if "test" in file_path.lower() or "reproduce" in file_path.lower():
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: You cannot use this tool to create test or files to reproduce the error.")
        return self._save(file_path, content)
    
    @EnhancedToolManager.tool   
    def get_approval_for_solution(self,solutions:list[str],selected_solution:int,reason_for_selection:str)->str:
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
        logger.info(f"solutions: {solutions}")
        logger.info(f"selected_solution: {selected_solution}")
        logger.info(f"reason_for_selection: {reason_for_selection}")
        parsed_solutions = []
        for solution in solutions:
            sols = re.split(r"(Solution \d+:)", solution)
            sols = [f"{sols[i]}{sols[i+1]}" for i in range(1, len(sols), 2)]  # Combine the split parts correctly
            parsed_solutions.extend(sols)
        
        solutions = parsed_solutions

        if type(solutions) is not list or len(solutions)<2:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: solutions must be a list with length at least 2.")

        self.is_solution_approved = True
        return "Approved"
          
    def _save(self,file_path: str, content: str)->str:
        is_syntax_error, error = self.check_syntax_error(content)
        if not is_syntax_error:
            with open(file_path, "w") as file:
                file.write(content)
            self.new_files_created.append(file_path)
            return f"File {file_path} saved successfully"
        else:
            logger.error(f"Error saving file: {error.message}")
            error.message="Error saving file. "+error.message
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,error.message)
 
    @EnhancedToolManager.tool
    def get_functions(self, function_paths: List[str]) -> Dict[str, str]:
        '''
        Get functions from a list of function paths.
        Arguments:
            function_paths: list of function paths (e.g. ["folder1/file1.py::class1::function1", "folder2/file2.py::class2::function2"])
        Output:
            dictionary of functions with function paths as keys and function bodies as values
        '''
        functions = {}
        for function_path in function_paths:
            parts = function_path.split("::")
            file_path = parts[0]
            function_name = "::".join(parts[1:])
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                tree = ast.parse(content, filename=file_path)
                visitor = FunctionVisitor(content)
                visitor.visit(tree)
                
                if function_name in visitor.functions:
                    functions[function_path] = visitor.functions[function_name].get("body", "")
                else:
                    functions[function_path] = f"Function {function_name} not found in {file_path}"
            except FileNotFoundError:
                functions[function_path] = f"File {file_path} not found"
            except Exception as e:
                functions[function_path] = f"Error processing {file_path}: {str(e)}"

        return functions

    @EnhancedToolManager.tool
    def get_classes(self, class_paths: List[str])->Dict[str, str]:
        '''
        Get classes from a list of class paths.
        Arguments:
            class_paths: list of class paths (e.g. ["folder1/file1.py::class1", "folder2/file2.py::class2"])
        Output:
            dictionary of classes with class paths as keys and class bodies as values
        '''
        classes = {}
        for class_path in class_paths:
            parts = class_path.split("::")
            file_path = parts[0]
            class_name = "::".join(parts[1:])
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                tree = ast.parse(content, filename=file_path)
                visitor = ClassVisitor(content)
                visitor.visit(tree)
                if class_name in visitor.classes:
                    classes[class_path] = visitor.classes[class_name].get("body", "")
                else:
                    classes[class_path] = f"Class {class_name} not found in {file_path}"
            except FileNotFoundError:
                classes[class_path] = f"File {file_path} not found"
            except Exception as e:
                classes[class_path] = f"Error processing {file_path}: {str(e)}"

        return classes

    @EnhancedToolManager.tool
    def run_repo_tests_for_fixing(self,file_paths:List[str])->str:
        '''
        Runs the tests for the repository. This tool will only run the tests for the files provided.
        Arguments:
            file_paths: path of the files to run the tests for.
        Output:
            Returns the stdout/stderr from the executed files.
        '''
        if self.test_runner == "pytest":
            print("CMD: pytest ", file_paths)
            result = subprocess.run(["pytest"] + file_paths, shell=True, capture_output=True, text=True, timeout=90)
            output = (result.stdout or "") + (result.stderr or "")
        elif self.test_runner == "unittest":
            print("CMD: python ", file_paths)
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
                # cmd = f"{self.test_runner} {' '.join(modules)}"
                cmd = f"{self.test_runner} {' '.join(modules)} -v 2"
                print("CMD: ", cmd)
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                output = (result.stdout or "") + (result.stderr or "")
            else:
                files_to_test = [clean_filepath(f, os.getcwd(), self.test_runner) for f in file_paths]
                # cmd = f"{self.test_runner} {' '.join(files_to_test)}"
                cmd = f"{self.test_runner} {' '.join(files_to_test)} -v 2"
                print("CMD: ", cmd)
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                output = (result.stdout or "") + (result.stderr or "")
        
        enhanced_output = self._enhance_test_output(output, file_paths)
        has_failures = any(indicator in output for indicator in ["FAIL:", "expected failures"])
        failure_count = 0
        if has_failures:
            import re
            
            fail_patterns = [
                r'(\d+)\s+failed',
                r'FAILED.*\(failures=(\d+)\)',
            ]
            for pattern in fail_patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    failure_count = int(match.group(1))
                    break
            
            if failure_count == 0:
                failure_count = output.count("FAIL:")
            
            expected_failure_match = re.search(r'expected failures[=\s]+(\d+)', output, re.IGNORECASE)
            if expected_failure_match:
                failure_count += int(expected_failure_match.group(1))
        
        self.last_test_results = {
            'has_failures': has_failures,
            'failure_count': failure_count,
            'file_paths': file_paths
        }
        
        return enhanced_output
    
    @EnhancedToolManager.tool
    def search_in_all_files_content(self, search_term: str, case_sensitive: bool = False) -> str:
        '''
        Search for a text pattern across all .py files in the project, excluding any file with "test" in its path.
        Use at the beginning of the workflow to locate all possible references to a function, class, or variable.
        If more context is needed (e.g., surrounding functions, classes, etc.), follow up with get_classes or get_functions.

        Arguments:
            search_term: text pattern to locate (e.g., "def test_function", "*SomeClass*")
            case_sensitive: flag to determine if the search should be case-sensitive
        Output:
            locations where pattern was found with file paths and line numbers
        '''
        output = []
        search_flags = 0 if case_sensitive else re.IGNORECASE

        for root, _, files in os.walk("."):
            # Skip .git and docs directories
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
                        logger.error(f"Error searching in file {file_path} with search term {search_term}: {e}")

        output = Utils.limit_strings("\n".join(output), n=100)
        if not output:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name, f"'{search_term}' not found in the codebase.")
        return output

    @EnhancedToolManager.tool
    def submit_solution(self, investigation_summary: str):
        '''
        Submit your solution after ALL repository tests pass. This is the ONLY way to finish.
        This tool will automatically verify that all tests pass before accepting your solution.
        Arguments:
            investigation_summary: Detailed summary of your investigation and solution. Format:
                Problem: <problem_statement>
                Investigation: <investigation_summary>
                Solution: <your solution>
        Output:
            Success message if all tests pass, or error if tests are still failing.
        '''
        # Check if we have recent test results stored
        if not hasattr(self, 'last_test_results'):
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,
                "❌ ERROR: You must run run_repo_tests_for_fixing() before submitting your solution!\n\n"
                "You cannot submit without verifying that all repository tests pass."
            )
        
        if self.last_test_results.get('has_failures', True):
            failing_count = self.last_test_results.get('failure_count', 'unknown')
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,
                f"❌ ERROR: Cannot submit solution - {failing_count} test(s) are still FAILING!\n\n"
                f"Last test run showed failures. You MUST fix all failing tests before submitting.\n"
                f"Run run_repo_tests_for_fixing() again after fixing the issues to verify all tests pass.\n\n"
                f"DO NOT try to submit with failing tests - this will always be rejected!"
            )
        
        # All tests passed! Allow submission
        return "finish_for_fixing"
    
    def get_function_ranges(self,file_path: str)->list[tuple[int, int, str]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,f"Error reading '{file_path}': {e}")
        try:
            tree = ast.parse("\n".join(source_lines), filename=file_path)
        except SyntaxError as e:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Error parsing '{file_path}': {e}, {traceback.format_exc()}")
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

    def _extract_function_matches(self,file_path: str, search_term: str, *, max_output_lines: int = 1000) -> str:
        '''
        Return the source code of any function definitions that contain `search_term`.
        If a match occurs outside of a function, only that line is returned. The final
        output is truncated with `limit_strings` to avoid excessive verbosity.
        '''
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            logger.error(f"Error reading '{file_path}': {e}")
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,f"Error reading '{file_path}': {e}")

        match_lines = [idx + 1 for idx, line in enumerate(source_lines) if search_term in line]
        if not match_lines:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name,f"'{search_term}' not found in file '{file_path}'")

        func_ranges=self.get_function_ranges(file_path)

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

    def _extract_failing_test_names(self, test_output: str) -> List[Dict[str, str]]:
        failing_tests = []
        seen = set()
        
        patterns = [
            r'FAIL:\s+(\w+)',      
            r'ERROR:\s+(\w+)',     
            r'FAILED.*::(\w+)',    
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, test_output):
                test_name = match.group(1)
                if test_name and test_name not in seen:
                    failing_tests.append({'name': test_name})
                    seen.add(test_name)
        
        return failing_tests
    
    @EnhancedToolManager.tool
    def search_in_specified_file_v2(self,file_path: str, search_term: str)->str:
        '''
        Locates text patterns within a specific file
        Arguments:
            file_path: target file for pattern matching. This file must be python file.
            search_term: text pattern to find (e.g., "def test_function", "*SomeClass*")
        Output:
            matching locations with line numbers, or error description
        '''
        if not file_path.endswith(".py"):
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_FILE_PATH.name,f"Error: file '{file_path}' is not a python file.")
        return self._extract_function_matches(file_path, search_term)
    
    @EnhancedToolManager.tool
    def start_over(self,problem_with_old_approach:str,new_apprach_to_try:str):
        '''
        This will revert any changes made to the codebase and let's you start over. Only use this tool when you have concluded that current changes you made to the codebase are not relevant and you want to start again with new approach.
        Arguments:
            problem_with_old_approach: What you tried and what was the key issues you faced with this approach.
            new_apprach_to_try: What is the new approach you want to try and how it will fix the issues you faced earlier.
        '''    
        logger.info("============Start Over============")
        os.system("git reset --hard")
        logger.info(f"problem_with_old_approach: {problem_with_old_approach}")
        logger.info(f"new_apprach_to_try: {new_apprach_to_try}")
        logger.info("===========================")
        return "Done, codebase reverted to initial state. You can start over with new approach."
        
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

            if diff.stderr:
                logger.warning("git diff (stderr): %s", diff.stderr.strip())

            patch_text = diff.stdout or ""
            return patch_text
        except Exception as e:
            logger.exception("Error generating git patch")
            return f"Error generating git patch: {e}"
    
    def _extract_class_attributes(self, class_node, lines: List[str]) -> Optional[str]:
        try:
            attrs = []
            for item in class_node.body:
                if isinstance(item, ast.Assign):
                    line_num = item.lineno - 1
                    if line_num < len(lines):
                        line_content = lines[line_num].strip()
                        if len(line_content) < 200:
                            attrs.append(line_content)
                elif isinstance(item, ast.AnnAssign):
                    line_num = item.lineno - 1
                    if line_num < len(lines):
                        line_content = lines[line_num].strip()
                        if len(line_content) < 200:
                            attrs.append(line_content)
            
            return '\n'.join(attrs) if attrs else None
        except Exception:
            return None
        
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
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_FILE_PATH.name,f"Error: file '{file_path}' is not a python file.")

        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        test_fn = (test_function_code or "").strip()
        if not test_fn:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,"Error: test_function_code cannot be empty.")

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
            lines = lines[:insert_idx] + (["", block, ""] if insert_idx < len(lines) else ["", block]) + lines[insert_idx:]
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
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Error: generated test function has syntax error: {err}")
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
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: invalid position '{position}'. Use 'append', 'top', 'after_imports', 'before_main', or 'auto'.")

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
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Error: inserting test caused syntax error. First error: {first_error}")

        self._save(file_path, new_content)

        rel = os.path.relpath(file_path)
        if rel not in self.generated_test_files:
            self.generated_test_files.append(rel)

        return f"Test {'created' if is_new_file else 'updated'} in '{rel}' (position={position})."

    def _extract_test_data(self, file_path: str, test_name: str) -> Optional[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=file_path)
            lines = content.splitlines()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    has_test = any(
                        isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == test_name
                        for item in node.body
                    )
                    
                    if has_test:
                        setup_methods = []
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                if item.name in ['setUp', 'setUpTestData', 'setUpClass', 'setup_method', 'setup_class']:
                                    start_line = item.lineno - 1
                                    end_line = self._get_function_end_line(item, lines)
                                    setup_code = '\n'.join(lines[start_line:end_line])
                                    setup_methods.append(f"# {item.name}\n{setup_code}")
                        
                        class_attrs = self._extract_class_attributes(node, lines)
                        if class_attrs:
                            setup_methods.insert(0, f"# Class attributes\n{class_attrs}")
                        
                        if setup_methods:
                            return '\n\n'.join(setup_methods)
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting test data: {e}")
            return None
        
    @EnhancedToolManager.tool
    def run_repo_tests(self,file_paths:List[str])->str:
        '''
        Runs the tests for the repository. This tool will only run the tests for the files provided.
        Arguments:
            file_paths: path of the files to run the tests for.
        Output:
            Returns the stdout/stderr from the executed files.
        '''
        if self.test_runner == "pytest":
            print("CMD: pytest ", file_paths)
            result = subprocess.run(["pytest"] + file_paths, shell=True, capture_output=True, text=True, timeout=90)
            output = (result.stdout or "") + (result.stderr or "")
        elif self.test_runner == "unittest":
            print("CMD: python ", file_paths)
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
                print("CMD: ", cmd)
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                output = (result.stdout or "") + (result.stderr or "")
            else:
                files_to_test = [clean_filepath(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(files_to_test)}"
                print("CMD: ", cmd)
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                output = (result.stdout or "") + (result.stderr or "")
        return output

    def _enhance_test_output(self, test_output: str, file_paths: List[str]) -> str:
        try:
            has_failures = any(indicator in test_output for indicator in 
                             ["FAIL:", "expected failures"])
            if not has_failures:
                return test_output
            failing_tests = self._extract_failing_test_names(test_output)
            logger.info(f'failing_tests: {failing_tests}')
            if not failing_tests:
                return test_output
            enhanced_parts = [
                "=" * 80,
                "TEST RESULTS:",
                "=" * 80,
                test_output,
                "",
                "=" * 80,
                "ENHANCED FAILURE INFORMATION:",
                "=" * 80,
                ""
            ]
            
            actual_file_paths = file_paths.copy()
            for file_path in file_paths:
                if os.path.isdir(file_path):
                    logger.info(f"Directory path detected: {file_path}, extracting actual file paths from test output")
                    extracted_paths = self._extract_test_file_paths_from_output(test_output)
                    if extracted_paths:
                        actual_file_paths.extend(extracted_paths)
                        logger.info(f"Extracted file paths from test output: {extracted_paths}")
            
            for test_info in failing_tests:
                logger.info(f'test_info: {test_info}')
                test_name = test_info.get('name')
                is_expected_failure = test_info.get('expected_failure', False)
                if not test_name:
                    continue
                    
                found_test = False
                for file_path in actual_file_paths:
                    actual_file_path = file_path.split('::')[0]
                    
                    if not os.path.exists(actual_file_path):
                        continue
                    test_code = self._extract_test_function_code(actual_file_path, test_name)
                    
                    if not test_code:
                        continue
                        
                    found_test = True  
                    test_data = self._extract_test_data(actual_file_path, test_name)
                    
                    enhanced_parts.append(f"{'─' * 80}")
                    if is_expected_failure:
                        enhanced_parts.append(f"❌ Expected Failure Test (MUST FIX): {test_name}")
                        enhanced_parts.append(f"⚠️  This test is marked as @expectedFailure - you MUST fix it and remove the decorator!")
                    else:
                        enhanced_parts.append(f"❌ Failing Test: {test_name}")
                    enhanced_parts.append(f"File: {actual_file_path}")
                    enhanced_parts.append(f"{'─' * 80}")
                    
                    enhanced_parts.append("\nTest Code:")
                    enhanced_parts.append("```python")
                    enhanced_parts.append(test_code)
                    enhanced_parts.append("```")
                    
                    if test_data:
                        enhanced_parts.append("\nTest Data (setUp/setUpTestData):")
                        enhanced_parts.append("```python")
                        enhanced_parts.append(test_data)
                        enhanced_parts.append("```")
                    
                    enhanced_parts.append("")
                    break
                
                if not found_test:
                    enhanced_parts.append(f"{'─' * 80}")
                    enhanced_parts.append(f"⚠️  WARNING: Could not extract test function '{test_name}'")
                    enhanced_parts.append(f"{'─' * 80}")
                    enhanced_parts.append("")
                    enhanced_parts.append("⚠️  ISSUE: Failed to automatically extract the test function code.")
                    enhanced_parts.append("📋 ACTION REQUIRED: Please read the actual test file(s) to understand the test:")
                    for file_path in actual_file_paths:
                        if os.path.exists(file_path):
                            enhanced_parts.append(f"   - {file_path}")
                    enhanced_parts.append("")
                    enhanced_parts.append("💡 TIP: Use read_file tool to examine the test file and locate the test function.")
                    enhanced_parts.append("     Then analyze the test logic to determine why it's failing and how to fix it.")
                    enhanced_parts.append("")
                    
            return "\n".join(enhanced_parts)
            
        except Exception as e:
            logger.error(f"Error enhancing test output: {e}")
            return test_output               
        
    @EnhancedToolManager.tool
    def run_code(self,content:str,file_path:str)->str:
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
        # Parse the file's AST to collect import statements
        
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

                # Skip relative imports ("from . import foo") which have level > 0
                if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                    continue

                # --- Additional check: allow local modules/packages in CWD ---
                cwd = os.getcwd()
                local_file = os.path.join(cwd, f"{mod}.py")
                local_pkg_init = os.path.join(cwd, mod, "__init__.py")
                local_pkg_dir = os.path.join(cwd, mod)
                # Also check inside a conventional 'lib' folder within cwd
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
                    # Treat as local dependency, allow it
                    continue

                # Any other module is considered disallowed
                disallowed_modules.add(mod)

        if disallowed_modules and False:
            logger.error(f"Cannot run, third party dependencies detected: {sorted(disallowed_modules)}\n")
            raise ToolManager.Error(ToolManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES.name,f"Error:Cannot run, third party dependencies detected: {sorted(disallowed_modules)}\n")

        result = subprocess.run(["python", file_path], capture_output=True, text=True, check=False, timeout=60)
        if result.returncode!=0:
            
            error_type=EnhancedToolManager.Error.ErrorType.RUNTIME_ERROR
            if "ImportError" in result.stderr:
                error_type=EnhancedToolManager.Error.ErrorType.IMPORT_ERROR
            if "ModuleNotFoundError" in result.stderr:
                error_type=EnhancedToolManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES
            raise EnhancedToolManager.Error(error_type,f"Error running code: {result.stderr}\n")
        observation = f"{result.stdout}\n"
    
        return observation
    
    def _extract_test_function_code(self, file_path: str, test_name: str) -> Optional[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=file_path)
            lines = content.splitlines()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if item.name == test_name:
                                start_line = item.lineno - 1
                                end_line = self._get_function_end_line(item, lines)
                                return '\n'.join(lines[start_line:end_line])
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == test_name:
                        start_line = node.lineno - 1
                        end_line = self._get_function_end_line(node, lines)
                        return '\n'.join(lines[start_line:end_line])
            
            return None
        except Exception as e:
            logger.error(f"Error extracting test function code: {e}")
            return None
        
    @EnhancedToolManager.tool
    def apply_code_edit(self,file_path:str, search:str, replace:str)->str:
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
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,f"Error: You cannot use this tool before you have approval from user on your proposed solution. Please call get_approval_for_solution tool first with list of proposed solutions.")
        if not os.path.exists(file_path):
            logger.error(f"file '{file_path}' does not exist.")
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,f"Error: file '{file_path}' does not exist.")
        
        original=self._get_file_content(file_path,limit=-1)

        match original.count(search):
            case 0:
                logger.error(f"search string not found in file {file_path}. You need to share the exact code you want to replace.")
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name,f"Error: search string not found in file {file_path}. You need to share the exact code you want to replace.")
            case 1:
                
                new_content = original.replace(search, replace)
                try:
                        is_error,error=self.check_syntax_error(new_content)
                        if not is_error:
                            self.save_file(file_path, new_content)
                                
                            return "ok, code edit applied successfully"
                        else:
                            error.message="code edit failed. "+error.message
                            raise error
                except EnhancedToolManager.Error as e:
                    raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,f"Error: syntax error in file {file_path}. {e.message}")
            case num_hits:
                logger.error(f"search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change.")
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.MULTIPLE_SEARCH_RESULTS_FOUND.name,f"Error: search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change.")
    
    @EnhancedToolManager.tool
    def run_validation_test(self, file_path: str) -> str:
        '''
        Run a validation test file that you created with generate_validation_test.
        Arguments:
            file_path: path to the validation test file
        Output:
            Test execution results (stdout/stderr)
        '''
        if not os.path.exists(file_path):
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,
                f"Error: validation test file '{file_path}' not found."
            )
        
        import subprocess
        try:
            result = subprocess.run(
                ['python', file_path],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.repo_path if hasattr(self, 'repo_path') else os.getcwd()
            )
            
            output = f"=== VALIDATION TEST RESULTS ===\n"
            output += f"File: {file_path}\n"
            output += f"Exit Code: {result.returncode}\n\n"
            
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n\n"
            
            if result.returncode == 0:
                output += "✅ Validation test PASSED\n\n"
                output += "=" * 80 + "\n"
                output += "⛔ CRITICAL WARNING - DO NOT SKIP THIS ⛔\n"
                output += "=" * 80 + "\n"
                output += "This validation test passing does NOT mean you are done!\n"
                output += "This is NOT your success criteria!\n\n"
                output += "⚠️ YOU MUST NOW:\n"
                output += "1. Run run_repo_tests_for_fixing() on the relevant existing test files\n"
                output += "2. Ensure ALL existing repository tests PASS (ZERO failures)\n"
                output += "3. If ANY test fails, you MUST fix it or revise your entire approach\n"
                output += "4. ONLY call finish_for_fixing after run_repo_tests_for_fixing shows ZERO failures\n\n"
                output += "❌ DO NOT call finish_for_fixing just because this validation test passed!\n"
                output += "❌ DO NOT think 'the original issue is fixed, so I'm done'!\n"
                output += "❌ DO NOT ignore failing repository tests with excuses like 'edge cases' or 'unrelated'!\n"
                output += "❌ DO NOT say 'these don't directly involve what I fixed' - if they fail after your changes, YOU broke them!\n"
                output += "=" * 80 + "\n"
            else:
                output += "❌ Validation test FAILED\n\n"
                output += "This means your fix has issues. Debug and fix them.\n"
                output += "After fixing, you MUST ALSO run run_repo_tests_for_fixing() to verify existing tests pass!\n"
            
            return output
            
        except subprocess.TimeoutExpired:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.TIMEOUT.name,
                f"Error: validation test timed out after 30 seconds."
            )
        except Exception as e:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.RUNTIME_ERROR.name,
                f"Error running validation test: {str(e)}"
            )
    
    @EnhancedToolManager.tool
    def finish(self,investigation_summary: str):
        '''
        Signals completion of the current workflow execution
        Arguments:
            investigation_summary: Please provide a detailed summary of the findings from your investigation and detailed solution to the problem.Use the following format:
                Problem: <problem_statement>
                Investigation: <investigation_summary>
                Solution: <your solution>
        '''
        qa_response={"is_patch_correct":"yes"}
        if qa_response.get("is_patch_correct","no").lower()=="yes":
            return "finish"
        else: 
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.BUG_REPORT_REQUIRED.name,qa_response.get("analysis",""))

    def _get_function_end_line(self, node, lines: List[str]) -> int:
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno
        
        start_line = node.lineno - 1
        if start_line >= len(lines):
            return len(lines)
        
        base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
        
        end_line = start_line + 1
        while end_line < len(lines):
            line = lines[end_line]
            if line.strip():
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= base_indent:
                    break
            end_line += 1
        
        return end_line
    
    @EnhancedToolManager.tool
    def finish_for_fixing(self,investigation_summary: str):
        '''
        Signals completion of the current workflow execution
        Arguments:
            investigation_summary: Please provide a detailed summary of the findings from your investigation and detailed solution to the problem.Use the following format:
                Problem: <problem_statement>
                Investigation: <investigation_summary>
                Solution: <your solution>
        '''
        qa_response={"is_patch_correct":"yes"}
        if qa_response.get("is_patch_correct","no").lower()=="yes":
            return self.submit_solution(investigation_summary)
        else: 
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.BUG_REPORT_REQUIRED.name,qa_response.get("analysis",""))

def ensure_git_initialized():
    """Initialize git repository if not already initialized, with temporary config."""
    print("[DEBUG] Starting git initialization check...")
    
    work_dir = os.getcwd()
    original_cwd = os.getcwd()
    
    try:
        print(f"[DEBUG] Work directory: {work_dir}")
        print(f"[DEBUG] Before chdir - pwd shows: {subprocess.run(['pwd'], capture_output=True, text=True).stdout.strip()}")
        
        os.chdir(work_dir)
        print(f"[DEBUG] After chdir - pwd shows: {subprocess.run(['pwd'], capture_output=True, text=True).stdout.strip()}")
        
        # Initialize git repo if not already initialized
        if not os.path.exists(".git"):
            print("[DEBUG] Initializing git repository...")
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
            
            # Verify .git was created in current directory
            print(f"[DEBUG] .git exists: {os.path.exists('.git')}")
            print(f"[DEBUG] Files in current dir: {os.listdir('.')[:10]}")  # Show first 10 files
            
            # Set local git config (only for this repo)
            print("[DEBUG] Setting git config...")
            subprocess.run(["git", "config", "--global", "user.email", "agent@sandbox.local"], check=True)
            subprocess.run(["git", "config", "--global", "user.name", "sandbox_agent"], check=True)

            # Add all files
            print("[DEBUG] Adding all files...")
            subprocess.run(["git", "add", "."], check=True)
            
            # Commit (ignore error if nothing to commit)
            print("[DEBUG] Creating initial commit...")
            result = subprocess.run(["git", "commit", "-m", "Initial commit"], check=False, capture_output=True, text=True)
            if result.returncode == 0:
                print("[DEBUG] Initial commit created successfully")
            else:
                print(f"[DEBUG] Commit result: {result.stderr.strip()}")
                
            print("[DEBUG] Git initialization completed successfully")
        else:
            print("[DEBUG] Git repository already exists")
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
        
    except Exception as e:
        print(f"[DEBUG] ERROR: Could not initialize git repository: {e}")
    finally:
        os.chdir(original_cwd)

def set_env_for_agent():
    
    if os.getcwd() not in os.environ.get("PYTHONPATH",""):
        os.environ["PYTHONPATH"]=os.environ.get("PYTHONPATH","")+":"+os.getcwd()
    if Path(os.getcwd()+"/lib").exists() and os.getcwd()+"/lib" not in os.environ.get("PYTHONPATH",""):
        os.environ["PYTHONPATH"]=os.environ["PYTHONPATH"]+":"+os.getcwd()+"/lib"

def process_fix_task(input_dict: Dict[str, Any]):
    """Main entry point for task processing and code modification.

    Parameters
    ----------
    input_dict : dict
        Configuration dictionary containing the task specification.
        Required key: 'problem_statement' with task details.
        Optional keys: 'run_id', 'instance_id' for tracking purposes.
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
    repod_path = repo_path[:-len(repod_dir)-1]
    if os.path.exists(repod_dir):
        os.chdir(repod_dir)

    set_env_for_agent()
    cwd = os.getcwd()
    logger.info(f"Current working directory: {cwd} and environ:{os.environ}")
    
    test_runner, test_runner_mode = get_test_runner_and_mode()
    print(f"test_runner: {test_runner}, test_runner_mode: {test_runner_mode}")

    try:
        logger.info(f"current files:{os.listdir()}")
        logger.info(f"packages installed:{subprocess.check_output(['pip','list']).decode('utf-8')}")
        logger.info(f"About to execute workflow...")
        patch_text= fix_task_solve_for_fixing_workflow(
            problem_text,
            timeout=timeout,
            run_id_1=run_id,
            test_runner=test_runner,
            test_runner_mode=test_runner_mode
        )
        logger.info(f"workflow execution completed, patch length: {len(patch_text)}")

        os.system("git reset --hard")

    except Exception as e:
        import traceback  # Ensure traceback is accessible
        error_info = f"Error: {e}, {traceback.format_exc()}"
        logger.error(f"[CRITICAL] Exception in task processing: {error_info}")
        logs.append(error_info)
    finally:
        os.chdir(cwd)

    print(f"[CRITICAL] task processor returning patch length: {len(patch_text)}")
    print(f"[CRITICAL] patch: {patch_text}")
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
            logger.error(f"Error: {e}")
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
            
        if should_apply_markup == False:
            return text_block

        for i, line in enumerate(lines):
            if line.strip() == '':                
                processed_line = '[EMPTY_LINE]'
            else:
                # Mark trailing and leading spaces
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

        # Normalize potential fenced responses
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

        logger.info(f"[MODEL-ROUTER] Selected model order via LLM: {mapped}")
        return mapped[:2]
    except Exception as e:
        logger.warning(f"[MODEL-ROUTER] Routing failed ({e}); using safe default order")
        return [QWEN_MODEL_NAME, DEEPSEEK_MODEL_NAME]

def generate_test_files(problem_statement: str, files_to_test: str, code_skeleton: str) -> str:
    retry = 0
    while retry < 10:
        try:
            logger.info("Starting test cases generation")
            testcases = generate_testcases_with_multi_step_reasoning(problem_statement, files_to_test, code_skeleton)
            
            if testcases:
                logger.info("Generated testcases successfully using multi-step reasoning")
                return testcases
            else:
                logger.warning("Multi-step reasoning failed, falling back to single-step approach")
                messages = [
                    {
                        "role": "system",
                        "content": GENERATE_INITIAL_TESTCASES_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"""Problem Statement:\n{problem_statement}\n\nPython files to test:\n{files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the ground truth and edge case coveraging testcases."""
                    }
                ]
                
                response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
                
                testcases = response.strip()
                if testcases.startswith('```python'):
                    testcases = testcases[9:]
                if testcases.startswith('```'):
                    testcases = testcases[3:]
                if testcases.endswith('```'):
                    testcases = testcases[:-3]
                testcases = testcases.strip()
                logger.info("Generated testcases successfully using fallback approach")
                return testcases
            
        except Exception as e:
            logger.error(f"Error generating initial solution: {str(e)}")
            retry += 1
            time.sleep(2)
    
    if retry >= 10:
        logger.error("Failed to generate initial solution")
        return ""
    return ""

def generate_testcases_with_multi_step_reasoning(problem_statement: str, files_to_test: str, code_skeleton: str) -> str:
    retry = 0
    test_generation_messages = [
        {
            "role": "system",
            "content": GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT
        },
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the complete and correct testcases in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"
        }
    ]
    while retry < 10:
        try:
            testcode_response = EnhancedNetwork.make_request(test_generation_messages, model=QWEN_MODEL_NAME)
            logger.info("Step 1 - Testcase Generation completed")
            
            testcases_check_messages = [
                {
                    "role": "system",
                    "content": TESTCASES_CHECK_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Problem statement: {problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerated Test Code:\n{testcode_response}\n\nAnalyze this code for invalid testcases. Return ONLY the final Python test code."
                }   
            ]
            
            testcode_checked_response = EnhancedNetwork.make_request(testcases_check_messages, model=QWEN_MODEL_NAME)
            logger.info("Step 2 - Testcase check completed")

            testcases = testcode_checked_response.strip()
            if testcases.startswith('```python'):
                testcases = testcases[9:]
            if testcases.startswith('```'):
                testcases = testcases[3:]
            if testcases.endswith('```'):
                testcases = testcases[:-3]
            testcases = testcases.strip()
            
            lines = testcases.split("\n")
            if lines[0].endswith(".py") == False:
                retry += 1
                test_generation_messages.append({"role": "assistant", "content": testcode_checked_response})
                test_generation_messages.append({"role": "user", "content": f"Include file name in the response. example:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"})
                print(f"Retrying because the first line is not a python test file name:\n {testcases}")
                continue

            logger.info("Multi-step reasoning solution generation completed successfully with infinite loop validation")
            return testcases
        except Exception as e:
            retry += 1
            print(f"Exception in generate_testcases_with_multi_step_reasoning: {e}")
            time.sleep(2)
    
    if retry >= 10:
        logger.error("Multi-step reasoning testcase generation failed")
        return ""
    
    return ""

def generate_solution_with_multi_step_reasoning(problem_statement: str, code_skeleton: str) -> str:
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
            code_response = EnhancedNetwork.make_request(code_generation_messages, model=QWEN_MODEL_NAME)
            logger.info("Step 1 - Code Generation completed")
            
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
            logger.info("Step 2 - Infinite Loop Check completed")

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
                code_generation_messages.append({"role": "user", "content": f"Include file name in the response. example:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```"})
                print(f"Retrying because the first line is not a python file name:\n {solution}")
                continue

            logger.info("Multi-step reasoning solution generation completed successfully with infinite loop validation")
            return solution
        except Exception as e:
            retry += 1
            print(f"Exception in generate_solution_with_multi_step_reasoning: {e}")
            time.sleep(2)
    
    if retry >= 10:
        logger.error("Multi-step reasoning solution generation failed")
        return ""
    
    return ""

def extract_and_write_files(initial_solution: str, base_dir: str = ".") -> list:
    import os
    
    created_files = []
    
    if not initial_solution.strip():
        print("No solution content to process")
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
            not stripped_line.startswith('#')):  # Not a comment
            
            if current_filename and current_content:
                file_path = os.path.join(base_dir, current_filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                content = '\n'.join(current_content).strip()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                created_files.append(file_path)
                print(f"Created file: {file_path}")
            
            current_filename = stripped_line
            current_content = []
        else:
            if current_filename:  # Only collect content if we have a filename
                current_content.append(line)
    
    # Write the last file
    if current_filename and current_content:
        file_path = os.path.join(base_dir, current_filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        content = '\n'.join(current_content).strip()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        created_files.append(file_path)
        print(f"Created file: {file_path}")
    
    return created_files

def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo"):
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
            result = process_fix_task(input_dict)
        else:
            result = process_create_task(input_dict)
    except Exception as e:
        result = process_fix_task(input_dict)
    os.system("git reset --hard")

    return result

def process_create_task(input_dict):
    problem_statement = input_dict.get("problem_statement", "")
    problem_statement = post_process_instruction(problem_statement)
    print(problem_statement)

    code_skeleton = get_code_skeleton()
    start_time = time.time()
    initial_solution = generate_initial_solution(problem_statement, code_skeleton)
    print(initial_solution)
    
    # Extract and write files from the solution
    created_files = extract_and_write_files(initial_solution)
    print(f"Created or Updated {len(created_files)} files: {created_files}")

    
    test_cases = generate_test_files(problem_statement, created_files, code_skeleton)
    print(test_cases)
    # Extract and write files from test cases
    test_files = extract_and_write_files(test_cases)
    print(f"Created or Updated {len(test_files)} files: {test_files}")

    timeout = DEFAULT_TIMEOUT - (time.time()-start_time) - 60
    
    patch = fix_task_solve_workflow(
        problem_statement,
        timeout=timeout,
        run_id_1=run_id,
        test_runner=f"unittest",
        test_runner_mode="FILE",
        n_max_steps=30
    )

    if patch is None:
        extract_and_write_files(initial_solution)

    tool_manager = EnhancedToolManager()
    patch = tool_manager.get_final_git_patch()
    return patch

def fix_task_solve_for_fixing_workflow(problem_statement: str, *, timeout: int, run_id_1: str,\
    test_runner: str = "pytest", test_runner_mode: str = "FILE", n_max_steps = MAX_FIX_TASK_STEPS) -> tuple[str, List[str], List[str]]:
    global run_id
    run_id=run_id_1
    cot=EnhancedCOT(latest_observations_to_keep=30)
    tool_manager=FixTaskEnhancedToolManager(
        available_tools=[
            "get_file_content",
            "save_file",
            "get_approval_for_solution",
            "get_functions",
            "get_classes",
            "search_in_all_files_content",
            "search_in_specified_file_v2",
            "start_over",
            "run_repo_tests_for_fixing",
            "run_code",
            "generate_validation_test",
            "run_validation_test",
            "apply_code_edit",
            "generate_test_function",
            "submit_solution",
            "finish_for_fixing"
        ],
        test_runner=test_runner,
        test_runner_mode=test_runner_mode
    )
    logger.info(f"Starting main agent execution...")
    system_prompt = FIX_TASK_FOR_FIXING_SYSTEM_PROMPT.format(tools_docs=tool_manager.get_tool_docs(),format_prompt=FORMAT_PROMPT_V0)
    instance_prompt = FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=problem_statement)
    
    start_time = time.time()
    logs: List[str] = []
    logs.append(f"cwd: {os.getcwd()}")
    logger.info(f"Starting workflow execution with {n_max_steps} max steps: timeout: {timeout} seconds : run_id: {run_id}")
    
    for step in range(n_max_steps):
        logger.info(f"Execution step {step + 1}/{n_max_steps}")
        
        if time.time() - start_time > timeout:
            cot.add_action(EnhancedCOT.Action(next_thought="global timeout reached",next_tool_name="",next_tool_args={},observation="",is_error=True,inference_error_counter={},request_data=[]))
            break

        messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instance_prompt},
            ]
        
        messages.extend(cot.to_str())
        messages.append({"role": "system", "content": STOP_INSTRUCTION})
    
        if cot.is_thought_repeated():
            logger.info(f"[TEST_PATCH_FIND] Thought repeated, adding DO NOT REPEAT TOOL CALLS instruction")
            last_thought = cot.thoughts[-1]
            messages.append({"role": "user", "content": DO_NOT_REPEAT_TOOL_CALLS.format(previous_response=f"next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}")})
    
        try:
            next_thought, next_tool_name, next_tool_args,raw_text,total_attempts,error_counter,messages = EnhancedNetwork.inference(messages, model=GLM_MODEL_NAME, run_id=run_id)
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_msg=f"\n\nERROR: {repr(e)} {traceback.format_exc()}"
            logger.error(f"Inference error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(next_thought=error_msg,next_tool_name="",next_tool_args={},observation="",is_error=True,raw_response=raw_text,total_attempts=total_attempts),inference_error_counter=error_counter,request_data=messages)
            break
        
        logger.info(f"About to execute operation: {next_tool_name}")
       
        try:
            logger.info(f"next_thought: {next_thought}\nnext_tool_name: {next_tool_name}\nnext_tool_args: {next_tool_args}\n")
            if '"' in next_tool_name or "'" in next_tool_name:
                next_tool_name=next_tool_name.replace('"','')
                next_tool_name=next_tool_name.replace("'","")
                
            next_observation = tool_manager.get_tool(next_tool_name)(**next_tool_args) if next_tool_args else tool_manager.get_tool(next_tool_name)()
            logger.info(f"next_observation: {next_observation}")
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=next_observation,is_error=False,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
        except EnhancedToolManager.Error as e:
            import traceback  
            error_msg=f"observation: {e.message}"
            logger.error(f"Tool error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
            continue
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_traceback=traceback.format_exc()
            if isinstance(e,TypeError):
                error_msg=f"observation: {str(e)}"
            else:
                error_msg=f"observation: {repr(e)} {error_traceback}"
            logger.error(f"Tool error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
            continue
        
        if next_tool_name == "finish_for_fixing":
            logger.info('[CRITICAL] Workflow called finish_for_fixing operation')
            if next_observation == "finish_for_fixing":  # Only break if validation passed
                break
            
        print(f"[CRITICAL] Completed step {step + 1}, continuing to next step")
    else:
        cot.add_action(EnhancedCOT.Action(next_thought="global timeout reached",next_tool_name="",next_tool_args={},observation="",is_error=True))
        logger.info(f"[CRITICAL] Workflow completed after reaching MAX_STEPS ({n_max_steps})")
        if n_max_steps < MAX_FIX_TASK_STEPS:
            return None
    
    logger.info(f"[CRITICAL] Workflow execution completed after {step + 1} steps")
    logger.info(f"[CRITICAL] About to generate final patch...")
    patch = tool_manager.get_final_git_patch()
    logger.info(f"Final Patch Generated..: Length: {len(patch)}")

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
                    tree_lines.append(f"{prefix}{'' if is_root else ('    ' if is_last else '│   ')}{connector}{file_name}")
                    
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
        
        response = EnhancedNetwork.make_request([
            {"role": "system", "content": FIND_TEST_RUNNER_PROMPT},
            {"role": "user", "content": readme_content}
        ], model=DEEPSEEK_MODEL_NAME)
        return response.strip() or "pytest"
    except Exception as e:
        logger.error(f"Error finding test runner: {e}")
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
        
        response = EnhancedNetwork.make_request([
            {"role": "system", "content": TEST_RUNNER_MODE_PROMPT},
            {"role": "user", "content": runner_content}
        ], model=DEEPSEEK_MODEL_NAME)
        return response.strip() or "FILE"
    except Exception as e:
        logger.error(f"Error determining test runner mode: {e}")
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
        print(f"no test file found")
        return "pytest", "FILE"

    print(f"test_file_path: {test_file_path}")
    readme_file_path = find_readme(test_file_path, '.')
    if readme_file_path:
        print(f"README found: {readme_file_path}")
        test_runner = find_test_runner(readme_file_path)
        test_runner_mode = get_test_runner_mode(test_runner)
    else:
        print("No README found, using default pytest")
    return test_runner, test_runner_mode

def fix_task_solve_workflow(problem_statement: str, *, timeout: int, run_id_1: str,\
    test_runner: str = "pytest", test_runner_mode: str = "FILE", n_max_steps = MAX_FIX_TASK_STEPS) -> tuple[str, List[str], List[str]]:
    global run_id
    run_id=run_id_1
    cot=EnhancedCOT(latest_observations_to_keep=30)
    tool_manager=FixTaskEnhancedToolManager(
        available_tools=[
            "get_file_content",
            "save_file",
            "get_approval_for_solution",
            "get_functions",
            "get_classes",
            "search_in_all_files_content",
            "search_in_specified_file_v2",
            "start_over",
            "run_repo_tests",
            "run_code",
            "apply_code_edit",
            "generate_test_function",
            "finish"
        ],
        test_runner=test_runner,
        test_runner_mode=test_runner_mode
    )
    logger.info(f"Starting main agent execution...")
    system_prompt = FIX_TASK_SYSTEM_PROMPT.format(tools_docs=tool_manager.get_tool_docs(),format_prompt=FORMAT_PROMPT_V0)
    instance_prompt = FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=problem_statement)
    
    start_time = time.time()
    logs: List[str] = []
    logs.append(f"cwd: {os.getcwd()}")
    logger.info(f"Starting workflow execution with {n_max_steps} max steps: timeout: {timeout} seconds : run_id: {run_id}")
    
    for step in range(n_max_steps):
        logger.info(f"Execution step {step + 1}/{n_max_steps}")
        
        if time.time() - start_time > timeout:
            cot.add_action(EnhancedCOT.Action(next_thought="global timeout reached",next_tool_name="",next_tool_args={},observation="",is_error=True,inference_error_counter={},request_data=[]))
            break

        messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instance_prompt},
            ]
        
        messages.extend(cot.to_str())

        messages.append({"role": "system", "content": STOP_INSTRUCTION})
    
        if cot.is_thought_repeated():
            logger.info(f"[TEST_PATCH_FIND] Thought repeated, adding DO NOT REPEAT TOOL CALLS instruction")
            last_thought = cot.thoughts[-1]
            messages.append({"role": "user", "content": DO_NOT_REPEAT_TOOL_CALLS.format(previous_response=f"next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}")})
    
        try:
            next_thought, next_tool_name, next_tool_args,raw_text,total_attempts,error_counter,messages = EnhancedNetwork.inference(messages, model=GLM_MODEL_NAME, run_id=run_id)
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_msg=f"\n\nERROR: {repr(e)} {traceback.format_exc()}"
            logger.error(f"Inference error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(next_thought=error_msg,next_tool_name="",next_tool_args={},observation="",is_error=True,raw_response=raw_text,total_attempts=total_attempts),inference_error_counter=error_counter,request_data=messages)
            break
        
        logger.info(f"About to execute operation: {next_tool_name}")
       
        try:
            logger.info(f"next_thought: {next_thought}\nnext_tool_name: {next_tool_name}\nnext_tool_args: {next_tool_args}\n")
            if '"' in next_tool_name or "'" in next_tool_name:
                next_tool_name=next_tool_name.replace('"','')
                next_tool_name=next_tool_name.replace("'","")
                
            next_observation = tool_manager.get_tool(next_tool_name)(**next_tool_args) if next_tool_args else tool_manager.get_tool(next_tool_name)()
            logger.info(f"next_observation: {next_observation}")
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=next_observation,is_error=False,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
        except EnhancedToolManager.Error as e:
            import traceback  # Ensure traceback is accessible
            error_msg=f"observation: {e.message}"
            logger.error(f"Tool error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
            continue
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_traceback=traceback.format_exc()
            if isinstance(e,TypeError):
                error_msg=f"observation: {str(e)}"
            else:
                error_msg=f"observation: {repr(e)} {error_traceback}"
            logger.error(f"Tool error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
            continue
        
        if next_tool_name == "finish":
            logger.info('[CRITICAL] Workflow called finish operation')
            break
        print(f"[CRITICAL] Completed step {step + 1}, continuing to next step")
    else:
        cot.add_action(EnhancedCOT.Action(next_thought="global timeout reached",next_tool_name="",next_tool_args={},observation="",is_error=True))
        logger.info(f"[CRITICAL] Workflow completed after reaching MAX_STEPS ({n_max_steps})")
        if n_max_steps < MAX_FIX_TASK_STEPS:
            return None
    
    logger.info(f"[CRITICAL] Workflow execution completed after {step + 1} steps")
    logger.info(f"[CRITICAL] About to generate final patch...")
    patch = tool_manager.get_final_git_patch()
    logger.info(f"Final Patch Generated..: Length: {len(patch)}")

    return patch

def generate_initial_solution(problem_statement: str, code_skeleton: str) -> str:
    models = determine_model_order(problem_statement)

    retry = 0
    while retry < 10:
        try:
            logger.info("Starting multi-step reasoning solution generation")
            
            solution = generate_solution_with_multi_step_reasoning(problem_statement, code_skeleton)
            
            if solution:
                logger.info("Generated initial solution successfully using multi-step reasoning")
                return solution
            else:
                logger.warning("Multi-step reasoning failed, falling back to single-step approach")
                
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
                
                response = EnhancedNetwork.make_request(messages, model=models[0])
                
                solution = response.strip()
                if solution.startswith('```python'):
                    solution = solution[9:]
                if solution.startswith('```'):
                    solution = solution[3:]
                if solution.endswith('```'):
                    solution = solution[:-3]
                solution = solution.strip()
                
                logger.info("Generated initial solution successfully using fallback approach")
                return solution
            
        except Exception as e:
            logger.error(f"Error generating initial solution: {str(e)}")
            retry += 1
            time.sleep(2)
    
    if retry >= 10:
        logger.error("Failed to generate initial solution")
        return ""
    return ""