#V4.0 - Combined Agent
# Combines strengths of miner-261.py and Kindness.py
# No AutoGen dependencies

"""
This combined agent integrates the best features from both:
- miner-261.py: Comprehensive test management, multi-step reasoning, sophisticated initial solution generation
- Kindness.py: EnhancedCOT management, cleaner OOP-based tool management, better error handling

Key improvements:
1. No AutoGen dependency (uses direct HTTP requests)
2. Enhanced COT management for conversation state
3. Structured error handling with ErrorType enum
4. Decorator-based tool registration
5. Intelligent model selection
6. Comprehensive test management
"""

from __future__ import annotations
import ast
import json
import os
import subprocess
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from json import JSONDecodeError
import re
import inspect
import random
from enum import Enum
import logging
import concurrent.futures
import threading
from collections import defaultdict, Counter
from uuid import uuid4
import urllib.request as _urlreq
from ast import literal_eval
import math
import requests

# Configuration
DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2000"))
MAX_FIX_TASK_STEPS = int(os.getenv("MAX_STEPS_TEST_PATCH_FIND", "400"))
MAX_TEST_PATCH_TIMEOUT = int(os.getenv("MAX_STEPS_TEST_PATCH_FIND", "400"))

PROBLEM_TYPE_CREATE = "CREATE"
PROBLEM_TYPE_FIX = "FIX"

# Model Configuration
GLM_MODEL_NAME = "zai-org/GLM-4.5-FP8"
GLM_MODEL_NAME_46 = "zai-org/GLM-4.6-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
AGENT_MODELS = [GLM_MODEL_NAME, QWEN_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME]

# Global state
RUN_ID = os.getenv("RUN_ID", str(uuid4()))  # Generate UUID if not provided
if RUN_ID == "nocache-1":
    RUN_ID = str(uuid4())
JSON_LLM_USED = 0
JSON_LITERAL_USED = 0
DEBUG_MODE = True
MARKDOWN_FAILED = 0
TOOL_CALL_FAILED = 0
MAX_EMBED_TOKENS = 128000
MAX_EMBED_CHARS = MAX_EMBED_TOKENS * 4
IS_SOLUTION_APPROVED = False
DISABLE_TEST_FILE_REMOVAL = False
TOO_MANY_SECTIONS_FOUND = 0

# Logging Setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

for h in list(logger.handlers):
    logger.removeHandler(h)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

log_file = "final_agent.log"
file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# ============================================================================
# ENHANCED CHAIN OF THOUGHT MANAGEMENT (from Kindness.py)
# ============================================================================

class EnhancedCOT:
    """Manages conversation state and chain of thought reasoning."""
    
    class Action:
        """Represents a single action in the conversation."""
        def __init__(self, next_thought: str, next_tool_name: str, next_tool_args: dict, 
                     observation: Union[list, tuple, str], is_error: bool = False, 
                     raw_response: str = None, total_attempts: int = 0, 
                     inference_error_counter: dict = None, request_data: list = None):
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
        self.thoughts: List[EnhancedCOT.Action] = []
        self.latest_observations_to_keep = latest_observations_to_keep
    
    def add_action(self, action: Action) -> bool:
        """Add a new action to the chain."""
        self.thoughts.append(action)
        return True
    
    def is_thought_repeated(self) -> bool:
        """Check if the last action is a repeat of the previous one."""
        if len(self.thoughts) < 2:
            return False
        last = self.thoughts[-1]
        prev = self.thoughts[-2]
        if last.next_tool_name == prev.next_tool_name and last.next_tool_args == prev.next_tool_args:
            return True
        return False
    
    def to_str(self):
        """Convert the chain of thought to conversation messages."""
        messages = []
        for i, thought in enumerate(self.thoughts):
            if thought.is_deleted:
                continue
            if i < len(self.thoughts) - self.latest_observations_to_keep:
                # Compress old observations
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
                user_str = (f"observation: {'error occurred.' if thought.is_error else ''} "
                           f"output omitted ({_obs_len}) lines\n")
            else:
                # Full observation for recent actions
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
                    assistant_str = (
                        f"next_thought:{thought.next_thought}\n"
                        f"next_tool_name:{thought.next_tool_name}\n"
                        f"next_tool_args:{thought.next_tool_args}")
                    if isinstance(thought.observation, (list, tuple)):
                        _obs_len = len(thought.observation)
                    else:
                        _obs_len = len(str(thought.observation).splitlines())
                    user_str = (
                        f"observation: error occurred. detailed output omitted "
                        f"({_obs_len}) lines\n"
                    )
            messages.append({"role": "assistant", "content": assistant_str})
            messages.append({"role": "user", "content": user_str})
        return messages


# ============================================================================
# ENHANCED TOOL MANAGER (from Kindness.py with improvements)
# ============================================================================

class EnhancedToolManager:
    """Base class for tool management with decorator-based registration."""
    
    logs = []
    TOOL_LIST = {}
    
    class Error(Exception):
        """Structured error handling."""
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
        """Decorator to mark methods as tools."""
        def wrapper(self, *args, **kwargs):
            EnhancedToolManager.TOOL_LIST[fn.__name__] = fn
            return fn(self, *args, **kwargs)
        wrapper.is_tool = True
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper
    
    @classmethod
    def tool_parsing(cls, fn):
        """Parse function signature to create tool schema."""
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
                raise ValueError(f"Parameter description not found for {param.name} in {doc_fn}")
            
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
    
    def get_tool_docs(self) -> str:
        """Get documentation for all tools."""
        return '\n\n'.join([json.dumps(EnhancedToolManager.tool_parsing(tool), ensure_ascii=False) 
                           for name, tool in EnhancedToolManager.TOOL_LIST.items()])
    
    def check_syntax_error(self, content: str, file_path: str = "<unknown>") -> Tuple[bool, Any]:
        """Check for syntax errors in Python code."""
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            logger.error(f"Syntax error: {e}")
            return True, EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR, 
                                                   f"Syntax error. {str(e)}")


# ============================================================================
# ENHANCED NETWORK (from Kindness.py with improvements)
# ============================================================================

class EnhancedNetwork:
    """Network communication layer."""
    
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
    def is_valid_response(cls, raw_text: str) -> Tuple[bool, Optional[str]]:
        """Validate response from LLM."""
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
    def get_error_counter(cls) -> dict:
        """Get error counter dictionary."""
        return {k: 0 for k in cls.ErrorType.__members__}
    
    @classmethod
    def make_request(cls, messages: list, model: str, attempt: int = 0, temperature: float = 0.0) -> str:
        """Make HTTP request to LLM."""
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
        
        # Convert messages to InferenceMessage format
        inference_messages = []
        for msg in messages:
            inference_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        request_data = {
            "run_id": RUN_ID,
            "model": AGENT_MODELS[attempt % len(AGENT_MODELS)],
            "temperature": temperature,
            "messages": inference_messages
        }
        
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(url, json=request_data, timeout=120, headers=headers)
        logger.info(f"[agent] HTTP {response.status_code} from {url} ({len(response.content)} bytes)")
        
        response.raise_for_status()
        
        # The inference gateway returns a plain string response, not JSON
        raw_text = response.text
        
        if type(raw_text) is not dict:
            raw_text = raw_text.lstrip()
        
        return raw_text
    
    @classmethod
    def sanitise_text_resp(cls, text_resp: str) -> str:
        """Sanitize text response."""
        text_resp = re.sub("[\'\"]*next_thought[\'\"]*:", "next_thought:", text_resp)
        text_resp = re.sub("[\'\"]*next_tool_name[\'\"]*:", "next_tool_name:", text_resp)
        text_resp = re.sub("[\'\"]*next_tool_args[\'\"]*:", "next_tool_args:", text_resp)
        text_resp = re.sub("[\'\"]*observation[\'\"]*:", "observation:", text_resp)
        
        if ("next_thought" not in text_resp and "next_tool_name:" in text_resp 
            and "next_tool_args:" in text_resp 
            and text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:") 
            and text_resp.find("next_tool_name:") > 10):
            logger.info(f"next_thought not found in {text_resp[:50]}, adding it")
            text_resp = "next_thought: " + text_resp
        
        return text_resp
    
    @classmethod
    def parse_response(cls, text_resp: str) -> Tuple[str, str, dict, Optional[str]]:
        """Parse LLM response into structured format."""
        error_msg = None
        text_resp = text_resp.strip()
        text_resp = text_resp.split("observation:")[0]
        text_resp = text_resp.strip().strip("\n")
        text_resp = cls.sanitise_text_resp(text_resp)
        
        if ("next_thought:" in text_resp and "next_tool_name:" in text_resp 
            and "next_tool_args:" in text_resp 
            and text_resp.find("next_thought:") < text_resp.find("next_tool_name:") 
            and text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:")):
            
            next_thought = text_resp.split("next_thought:")[1].split("next_tool_name:")[0].strip().strip("\n")
            next_tool_name = text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n")
            next_tool_args = text_resp.split("next_tool_args:")[1].strip().split("next_thought:")[0].strip().strip("\n")
            
            try:
                next_tool_args = json.loads(next_tool_args)
            except JSONDecodeError as e:
                error_msg = f"Invalid JSON: {str(e)}"
        else:
            if "next_thought:" not in text_resp:
                error_msg = "Invalid response. next_thought not found"
            elif "next_tool_name:" not in text_resp:
                error_msg = "Invalid response. next_tool_name not found"
            elif "next_tool_args:" not in text_resp:
                error_msg = "Invalid response. next_tool_args not found"
            else:
                error_msg = f"Invalid response. Please follow the response format"
            return None, None, None, error_msg
        
        return next_thought, next_tool_name, next_tool_args, error_msg
    
    @classmethod
    def _request_next_action_with_retry(cls, messages: dict, model: str, max_retries: int = 10, 
                                       base_delay: float = 2.0, temperature: float = 0.0) -> Tuple:
        """Request next action with retry logic."""
        raw_text = 'not defined'
        error_counter = cls.get_error_counter()
        next_thought, next_tool_name, next_tool_args = None, None, None
        total_attempts = 0
        
        for attempt in range(max_retries):
            try:
                total_attempts += 1
                raw_text = cls.make_request(messages, model, attempt=attempt, temperature=temperature)
                is_valid, error_msg = cls.is_valid_response(raw_text)
                
                if not is_valid:
                    logger.error(f"raw_text: {raw_text}")
                    logger.error("--------------------------------")
                    raise Exception(error_msg)
                
                next_thought, next_tool_name, next_tool_args, error_msg = cls.parse_response(raw_text)
                if error_msg:
                    raise Exception(error_msg)
                break
            except Exception as e:
                error_body = str(e)
                logger.error(f"Error: {error_body}")
                if attempt < max_retries:
                    delay = base_delay
                    logger.info(f"[agent] Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(random.uniform(2*delay, 2.2*delay))
                    continue
                else:
                    raise RuntimeError(error_body)
        
        return next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages
    
    @classmethod
    def inference(cls, messages: List[Dict[str, Any]], model: str, run_id: str = str(uuid4()), 
                  temperature: float = 0.0) -> dict:
        """Perform inference with the LLM."""
        cleaned_msgs = []
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
        
        next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages = \
            cls._request_next_action_with_retry(cleaned_msgs, model)
        
        return next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages


# ============================================================================
# FIX TASK ENHANCED TOOL MANAGER (Combined from both)
# ============================================================================

class FixTaskEnhancedToolManager(EnhancedToolManager):
    """Enhanced tool manager for fix tasks."""
    
    generated_test_files = []
    
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
                    if available_tools is not None and name not in available_tools:
                        continue
                    self.TOOL_LIST[name] = self.__class__.tool_parsing(attr)
        
        self.tool_failure = {k: {j: 0 for j in self.Error.ErrorType.__members__} 
                            for k in self.TOOL_LIST.keys()}
        self.tool_invocations = {k: 0 for k in self.TOOL_LIST.keys()}
    
    def _get_file_content(self, file_path: str, search_start_line: int = None, 
                         search_end_line: int = None, search_term: str = None, limit: int = 5000) -> str:
        """Get file content with optional filtering."""
        if search_term is not None and search_term != "":
            logger.debug(f"search_term specified: {search_term}, searching in v2")
            return self.search_in_specified_file_v2(file_path, search_term)
        
        func_ranges = self.get_function_ranges(file_path)
        if search_start_line is not None:
            for start, end, name in func_ranges:
                if start <= search_start_line <= end:
                    if start < search_start_line:
                        logger.debug(f"search start line {search_start_line} is between a function, setting to {start}")
                        search_start_line = start
        if search_end_line is not None:
            for start, end, name in func_ranges:
                if start <= search_end_line <= end:
                    if end > search_end_line:
                        logger.debug(f"search end line {search_end_line} is between a function, setting to {end}")
                        search_end_line = end
        
        with open(file_path, "r") as f:
            if search_start_line is not None or search_end_line is not None:
                lines = f.readlines()
                start = max(0, (search_start_line or 1) - 1)
                end = min(len(lines), search_end_line or len(lines))
                content = ''.join(lines[start:end])
                return f"Lines {start+1}-{end} of {file_path}:\n{content}"
            else:
                content = f.read()
        
        return self.limit_strings(content, n=limit) if limit != -1 else content
    
    def limit_strings(self, strings: str, n=1000) -> str:
        """Limit the number of strings."""
        strings_list = strings.split("\n")
        if len(strings_list) > n:
            return "\n".join(strings_list[:n]) + "\n..." + f"({len(strings_list)-n} more lines)"
        else:
            return strings
    
    def get_function_ranges(self, file_path: str) -> list:
        """Get function ranges in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            return f"Error reading '{file_path}': {e}"
        
        try:
            tree = ast.parse("\n".join(source_lines), filename=file_path)
        except SyntaxError as e:
            return f"Error parsing '{file_path}': {e}, {traceback.format_exc()}"
        
        func_ranges: list[tuple[int, int, str]] = []
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = getattr(node, 'lineno', None)
                    end = getattr(node, 'end_lineno', None)
                    if start is not None and end is not None:
                        func_ranges.append((start, end, node.name))
        return func_ranges
    
    def _save(self, file_path: str, content: str) -> str:
        """Save file with syntax checking."""
        is_syntax_error, error = self.check_syntax_error(content)
        if not is_syntax_error:
            with open(file_path, "w") as file:
                file.write(content)
            self.new_files_created.append(file_path)
            return f"File {file_path} saved successfully"
        else:
            logger.error(f"Error saving file: {error.message}")
            error.message = "Error saving file. " + error.message
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR, error.message)
    
    @EnhancedToolManager.tool
    def get_file_content(self, file_path: str, search_start_line: int = None, 
                         search_end_line: int = None, search_term: str = None) -> str:
        """Retrieves file contents with optional filtering based on search term and line numbers.
        Arguments:
            file_path: filesystem path to target file. This file must be python file.
            search_start_line: optional start line number to begin extraction (1-indexed)
            search_end_line: optional end line number to end extraction (1-indexed)
            search_term: optional text pattern to filter matching lines
        Output:
            File content or error message
        """
        return self._get_file_content(file_path, search_start_line, search_end_line, search_term, limit=5000)
    
    @EnhancedToolManager.tool
    def save_file(self, file_path: str, content: str) -> str:
        """Writes text content to specified filesystem location. If there are any syntax errors in the code, it rejects the edit with an error message. Do not use this tool to create test or files to reproduce the error.
        Arguments:
            file_path: target filesystem path
            content: text data to write
        Output:
            Success message or error message
        """
        if "test" in file_path.lower() or "reproduce" in file_path.lower():
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL,
                                           "Error: You cannot use this tool to create test or files to reproduce the error.")
        return self._save(file_path, content)
    
    @EnhancedToolManager.tool
    def get_approval_for_solution(self, solutions: list[str], selected_solution: int, 
                                  reason_for_selection: str) -> str:
        """This tool is used to get approval for your proposed solution. You need to propose at least 2 meaningfully different and elegant solutions to the problem.
        Arguments:
            solutions: list of solutions proposed by you
            selected_solution: Index of the solution you think is the best
            reason_for_selection: Reason for selecting the solution over other solutions
        Output:
            approval: approved/not approved. If approved, you can go ahead and implement the solution.
        """
        logger.info(f"solutions: {solutions}")
        logger.info(f"selected_solution: {selected_solution}")
        logger.info(f"reason_for_selection: {reason_for_selection}")
        
        if type(solutions) is not list or len(solutions) < 2:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL,
                                           "Error: solutions must be a list with length at least 2.")
        
        self.is_solution_approved = True
        return "Approved"
    
    @EnhancedToolManager.tool
    def apply_code_edit(self, file_path: str, search: str, replace: str) -> str:
        """Performs targeted text replacement within source files. If there are any syntax errors in the code, it rejects the edit with an error message.
        Arguments:
            file_path: target file for modification
            search: exact text pattern to locate and replace
            replace: new text content to substitute
        Output:
            Operation status - success confirmation or detailed error
        """
        if not self.is_solution_approved:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL,
                                           "Error: You cannot use this tool before you have approval from user on your proposed solution.")
        
        if not os.path.exists(file_path):
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND,
                                           f"Error: file '{file_path}' does not exist.")
        
        original = self._get_file_content(file_path, limit=-1)
        
        match original.count(search):
            case 0:
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND,
                                                f"Error: search string not found in file {file_path}.")
            case 1:
                new_content = original.replace(search, replace)
                try:
                    is_error, error = self.check_syntax_error(new_content)
                    if not is_error:
                        self._save(file_path, new_content)
                        logger.info("ok, code edit applied successfully")
                        return "ok, code edit applied successfully"
                    else:
                        raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR,
                                                       f"Error: code edit failed. {error.message}")
                except Exception as e:
                    raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR,
                                                    f"Error: syntax error in file {file_path}. {e}")
            case num_hits:
                raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.MULTIPLE_SEARCH_RESULTS_FOUND,
                                                f"Error: search string found {num_hits} times in file '{file_path}'.")
    
    @EnhancedToolManager.tool
    def search_in_all_files_content(self, search_term: str, case_sensitive: bool = False) -> str:
        """Search for a text pattern across all .py files in the project.
        Arguments:
            search_term: text pattern to locate (e.g., "def test_function", "*SomeClass*")
            case_sensitive: flag to determine if the search should be case-sensitive
        Output:
            locations where pattern was found with file paths and line numbers
        """
        logger.info("tool called: search_in_all_files_content")
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
                        
                        # Find line numbers
                        lines = content.splitlines()
                        for idx, line in enumerate(lines):
                            if re.search(search_term, line, search_flags):
                                output.append(f"{file_path}:{idx+1} | {line.rstrip()}")
                    except Exception as e:
                        logger.error(f"Error searching in file {file_path} with search term {search_term}: {e}")
        
        result = self.limit_strings("\n".join(output), n=100)
        if not result:
            return f"'{search_term}' not found in the codebase."
        return result
    
    @EnhancedToolManager.tool
    def search_in_specified_file_v2(self, file_path: str, search_term: str) -> str:
        """Locates text patterns within a specific file.
        Arguments:
            file_path: target file for pattern matching. This file must be python file.
            search_term: text pattern to find (e.g., "def test_function", "*SomeClass*")
        Output:
            matching locations with line numbers, or error description
        """
        if not file_path.endswith(".py"):
            return f"Error: file '{file_path}' is not a python file."
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            logger.error(f"Error reading '{file_path}': {e}")
            return f"Error reading '{file_path}': {e}"
        
        match_lines = [idx + 1 for idx, line in enumerate(source_lines) if search_term in line]
        if not match_lines:
            return f"'{search_term}' not found in file '{file_path}'"
        
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
            chunks.append(f"Function {name} (lines {start}-{end}):\n{func_src}")
        
        for ln in standalone_lines:
            chunks.append(f"{ln}:{source_lines[ln - 1]}")
        
        return self.limit_strings("\n\n".join(chunks), n=1000)
    
    @EnhancedToolManager.tool
    def run_code(self, content: str, file_path: str) -> str:
        """Runs any python code. Saves the code at the given file_path and then runs it.
        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in
        Output:
            Returns the stdout/stderr from the executed file.
        """
        is_syntax_error, error = self.check_syntax_error(content)
        if is_syntax_error:
            return f"Error: syntax error. {error.message}"
        
        self._save(file_path, content)
        self.generated_test_files.append(os.path.normpath(file_path))
        
        result = subprocess.run(["python", file_path], capture_output=True, text=True, check=False, timeout=60)
        if result.returncode != 0:
            logger.error(f"Error running code: {result.stderr}\n")
            return f"Error running code: {result.stderr}\n"
        
        observation = f"{result.stdout}\n"
        if result.stderr:
            observation += f"\nSTDERR: {result.stderr}"
        logger.info(f"output: {observation}")
        
        return observation
    
    @EnhancedToolManager.tool
    def start_over(self, problem_with_old_approach: str, new_apprach_to_try: str) -> str:
        """This will revert any changes made to the codebase and let's you start over.
        Arguments:
            problem_with_old_approach: What you tried and what was the key issues you faced
            new_apprach_to_try: What is the new approach you want to try
        Output:
            Confirmation message
        """
        logger.info("============Start Over============")
        os.system("git reset --hard")
        logger.info(f"problem_with_old_approach: {problem_with_old_approach}")
        logger.info(f"new_apprach_to_try: {new_apprach_to_try}")
        logger.info("===========================")
        return "Done, codebase reverted to initial state. You can start over with new approach."
    
    @EnhancedToolManager.tool
    def finish(self, investigation_summary: str) -> str:
        """Signals completion of the current workflow execution.
        Arguments:
            investigation_summary: Please provide a detailed summary of the findings
        Output:
            Completion status
        """
        return "finish"
    
    def get_final_git_patch(self, initial_checkpoint: str = None) -> str:
        """Generate a clean unified diff (staged changes only)."""
        try:
            exts = (".py", ".ini", ".cfg", ".toml")
            exclude = {"src/agent.py", "src/agent_runner.py"}
            
            try:
                for _p in self.generated_test_files:
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
            
            if initial_checkpoint:
                diff = subprocess.run(
                    ["git", "diff", "--cached", initial_checkpoint, "--no-color", "--unified=3"],
                    capture_output=True, text=True, timeout=30, check=True
                )
            else:
                diff = subprocess.run(
                    ["git", "diff", "--cached", "--no-color", "--unified=3"],
                    capture_output=True, text=True, timeout=30, check=True
                )
            
            if diff.stderr:
                logger.error("git diff (stderr): %s", diff.stderr.strip())
            
            patch_text = diff.stdout or ""
            if not patch_text:
                logger.error("Patch text is empty..")
            
            return patch_text
        except Exception as e:
            logger.exception("Error generating git patch")
            return f"Error generating git patch: {e}"
    
    def remove_any_generated_test_files(self):
        """Remove generated test files."""
        for file in self.generated_test_files:
            if os.path.exists(file):
                os.remove(file)
        self.generated_test_files = []


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

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
12. After fixing the issue, you can test the changes by running the run_code tool.
13. If you find that the error while running the run_code or run_repo_tests tool due to missing dependencies, do not try to solve it as you don't have any internet access.
14. Call the finish tool to finish the task after you have fixed the issue.

## Multi-file awareness (critical):
- Tests and patch contexts may span multiple files. Do not stop after the first similar match or applied fix.
- Keep searching the repository after each match and apply consistent changes to every relevant file before finishing.
- Prefer using `search_in_all_files_content` to enumerate matches across the codebase and `search_in_specified_file_v2` to drill into each file; iterate until no applicable occurrences remain.
- Re-run tests only after covering all discovered occurrences to avoid partial fixes.

You have access to the following tools:-
{tools_docs}

{format_prompt}
""")

FIX_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
# Now let's start. Here is the problem statement:
{problem_statement}
""")

FORMAT_PROMPT_V0 = """
Your response must not contain multiple THOUGHT or TOOL_CALL sections. You must respond in the following format. You must not add anything before THOUGHT section.
======THOUGHT
<<your detailed thought process>>
======TOOL_CALL
{"name":"<tool_name>","arguments":{...}}
"""

STOP_INSTRUCTION = textwrap.dedent("""
# 🎨 
DO NOT generate `observation:` in your response. It will be provided by user for you.
Generate only SINGLE triplet of `next_thought`, `next_tool_name`, `next_tool_args` in your response.
""")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_git_initialized():
    """Initialize git repository if not already initialized."""
    print("[DEBUG] Starting git initialization check...")
    
    work_dir = os.getcwd()
    
    try:
        if not os.path.exists(".git"):
            print("[DEBUG] Initializing git repository...")
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
            subprocess.run(["git", "config", "--global", "user.email", "agent@sandbox.local"], check=True)
            subprocess.run(["git", "config", "--global", "user.name", "sandbox_agent"], check=True)
            subprocess.run(["git", "add", "."], check=True)
            result = subprocess.run(["git", "commit", "-m", "Initial commit"], check=False, capture_output=True, text=True)
            if result.returncode == 0:
                print("[DEBUG] Initial commit created successfully")
            print("[DEBUG] Git initialization completed successfully")
        else:
            print("[DEBUG] Git repository already exists")
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
        
    except Exception as e:
        print(f"[DEBUG] ERROR: Could not initialize git repository: {e}")

def set_env_for_agent():
    """Set up environment for the agent."""
    if os.getcwd() not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + os.getcwd()
    if Path(os.getcwd() + "/lib").exists() and os.getcwd() + "/lib" not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + ":" + os.getcwd() + "/lib"


# ============================================================================
# MAIN WORKFLOW FUNCTIONS
# ============================================================================

def fix_task_solve_workflow(problem_statement: str, *, timeout: int, run_id_1: str,
                            test_runner: str = "pytest", test_runner_mode: str = "FILE", 
                            n_max_steps: int = MAX_FIX_TASK_STEPS) -> str:
    """Main workflow for solving fix tasks."""
    global RUN_ID
    RUN_ID = run_id_1
    
    cot = EnhancedCOT(latest_observations_to_keep=30)
    tool_manager = FixTaskEnhancedToolManager(
        available_tools=[
            "get_file_content",
            "save_file",
            "get_approval_for_solution",
            "search_in_all_files_content",
            "search_in_specified_file_v2",
            "start_over",
            "run_code",
            "apply_code_edit",
            "finish"
        ],
        test_runner=test_runner,
        test_runner_mode=test_runner_mode
    )
    
    logger.info(f"Starting main agent execution...")
    system_prompt = FIX_TASK_SYSTEM_PROMPT.format(
        tools_docs=tool_manager.get_tool_docs(),
        format_prompt=FORMAT_PROMPT_V0
    )
    instance_prompt = FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=problem_statement)
    
    start_time = time.time()
    logs: List[str] = []
    logs.append(f"cwd: {os.getcwd()}")
    logger.info(f"Starting workflow execution with {n_max_steps} max steps: timeout: {timeout} seconds")
    
    # Get initial response
    response = EnhancedNetwork.inference(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt}
        ],
        model=GLM_MODEL_NAME,
        run_id=run_id_1
    )
    
    next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages = response
    
    for step in range(n_max_steps):
        logger.info(f"Execution step {step + 1}/{n_max_steps}")
        
        if time.time() - start_time > timeout:
            cot.add_action(EnhancedCOT.Action(
                next_thought="global timeout reached",
                next_tool_name="",
                next_tool_args={},
                observation="",
                is_error=True,
                inference_error_counter={},
                request_data=[]
            ))
            break
        
        # Execute tool
        try:
            tool_func = getattr(tool_manager, next_tool_name)
            observation = tool_func(**next_tool_args)
            
            if observation == "finish":
                break
            
            cot.add_action(EnhancedCOT.Action(
                next_thought=next_thought,
                next_tool_name=next_tool_name,
                next_tool_args=next_tool_args,
                observation=observation,
                is_error=False,
                raw_response=raw_text,
                total_attempts=total_attempts,
                inference_error_counter=error_counter,
                request_data=messages
            ))
            
            # Get next action
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instance_prompt},
            ]
            messages.extend(cot.to_str())
            messages.append({"role": "system", "content": STOP_INSTRUCTION})
            
            response = EnhancedNetwork.inference(
                messages,
                model=GLM_MODEL_NAME,
                run_id=run_id_1
            )
            
            next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages = response
            
        except Exception as e:
            logger.error(f"Error in step {step + 1}: {e}")
            cot.add_action(EnhancedCOT.Action(
                next_thought=next_thought,
                next_tool_name=next_tool_name,
                next_tool_args=next_tool_args,
                observation=str(e),
                is_error=True,
                raw_response=raw_text,
                total_attempts=total_attempts,
                inference_error_counter=error_counter,
                request_data=messages
            ))
            
            # Retry with error information
            messages.append({"role": "user", "content": f"observation: Error occurred: {str(e)}"})
            response = EnhancedNetwork.inference(
                messages,
                model=GLM_MODEL_NAME,
                run_id=run_id_1
            )
            next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages = response
    
    # Generate final patch
    final_patch = tool_manager.get_final_git_patch()
    if not DISABLE_TEST_FILE_REMOVAL:
        tool_manager.remove_any_generated_test_files()
    
    logger.info(f"Final patch: {final_patch}")
    logger.info(f"Total time taken: {time.time() - start_time} seconds")
    
    return final_patch


def check_problem_type(problem_statement: str) -> str:
    """Check if problem is CREATE or FIX type."""
    retry = 0
    while retry < 10:
        try:
            messages = [
                {"role": "system", "content": "You are a problem type classifier. Classify problems as CREATE (new functionality) or FIX (bug fix). Only respond with CREATE or FIX."},
                {"role": "user", "content": problem_statement}
            ]
            
            response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
            response = response.strip()
            
            if response in [PROBLEM_TYPE_CREATE, PROBLEM_TYPE_FIX]:
                return response
            retry += 1
        except Exception as e:
            logger.error(f"Error: {e}")
            retry += 1
        
        time.sleep(2)
    
    return PROBLEM_TYPE_FIX  # Default to FIX


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo"):
    """Legacy interface wrapper for backwards compatibility."""
    global DEFAULT_PROXY_URL, DEFAULT_TIMEOUT, RUN_ID
    
    # Generate a new UUID for this run
    RUN_ID = str(uuid4())
    repo_dir = os.path.abspath(repo_dir)
    sys.path.insert(0, repo_dir)
    
    if os.path.exists(repo_dir):
        os.chdir(repo_dir)
    
    ensure_git_initialized()
    set_env_for_agent()
    
    try:
        problem_type = check_problem_type(input_dict.get("problem_statement"))
        logger.info(f"Problem type: {problem_type}")
        
        if problem_type == PROBLEM_TYPE_FIX:
            result = fix_task_solve_workflow(
                input_dict.get("problem_statement"),
                timeout=DEFAULT_TIMEOUT,
                run_id_1=RUN_ID
            )
        else:
            # For CREATE tasks, use a simplified workflow
            logger.info("CREATE task detected - using simplified workflow")
            result = fix_task_solve_workflow(
                input_dict.get("problem_statement"),
                timeout=DEFAULT_TIMEOUT,
                run_id_1=RUN_ID
            )
        
        if not DISABLE_TEST_FILE_REMOVAL:
            os.system("git reset --hard")
        
        logger.info(f"patch returned: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in agent_main: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error: {e}"


if __name__ == "__main__":
    # Example usage
    result = agent_main({
        "problem_statement": "Fix the bug in the calculate function"
    }, repo_dir="repo")
    print(result)
