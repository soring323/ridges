"""
Network communication module for LLM interactions.
"""

import json
import logging
import random
import re
import time
import traceback
from enum import Enum
from typing import Any, Dict, List
from json import JSONDecodeError
from uuid import uuid4

import requests

from kindness_refactored.constants import DEFAULT_PROXY_URL, DEEPSEEK_MODEL_NAME, AGENT_MODELS
from kindness_refactored.utils import Utils

logger = logging.getLogger(__name__)


class EnhancedNetwork:
    """Handles network communication with LLM providers."""
    
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
    def is_valid_response(cls, raw_text: str) -> tuple[bool, str]:
        """Validate LLM response for common issues."""
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
        """Get error counter dictionary."""
        return {k: 0 for k in cls.ErrorType.__members__}

    @classmethod
    def fix_json_string_with_llm(cls, json_string: str, attempt: int = 0) -> dict:
        """Fix malformed JSON using LLM."""
        messages = [
            {"role": "system", "content": "Fix the json string sent by the user.  Reply only with the json string and nothing else."},
            {"role": "user", "content": json_string}
        ]
        response = cls.make_request(messages, model=DEEPSEEK_MODEL_NAME)
        try:
            response = response.replace('```json', '').strip('```')
            response = json.loads(response)
            return response
        except JSONDecodeError as e:
            logger.error(f"Error fixing json string: {e},trying again..")
            logger.error(f"json string is :{json_string}")
            logger.error(f"LLM response is :{response}")
            return None
    
    @classmethod
    def make_request(cls, messages: list, model: str, attempt: int = 0, temperature: float = 0.0) -> str:
        """Make request to LLM provider."""
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
            is_oai_interface = (type(response_json) is dict and 
                              response_json.get('choices') is not None and 
                              len(response_json.get('choices')) > 0 and 
                              response_json.get('choices')[0].get('message') is not None)
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
                            temperature: float = 0.0) -> tuple:
        """Request next action with retry logic."""
        raw_text = 'not defined'
        error_counter = cls.get_error_counter()
        next_thought, next_tool_name, next_tool_args = None, None, None
        total_attempts = 0
        
        for attempt in range(max_retries):
            try:
                total_attempts += 1
                index = AGENT_MODELS.index(model) if model in AGENT_MODELS else -1
                raw_text = cls.make_request(messages, model=AGENT_MODELS[(index + attempt) % len(AGENT_MODELS)], temperature=temperature)
                is_valid, error_msg = cls.is_valid_response(raw_text)
                if not is_valid:
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
                    logger.info(error_body)
                    logger.error("--------------------------------")
                    logger.error(f"response: {raw_text}")
                    logger.error("--------------------------------")
                    logger.info(f"[agent] Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})") 
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
                    if ("RATE_LIMIT_EXCEEDED" not in error_body and 
                        "RESERVED_TOKEN_PRESENT" not in error_body and 
                        "EMPTY_RESPONSE" not in error_body and 
                        "TIMEOUT" not in error_body):
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
        """Parse malformed JSON using regex patterns."""
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
        """Parse tool arguments from string to JSON."""
        next_tool_args = next_tool_args.replace('```json', '').strip('```')
        error_msg = ''

        try:
            next_tool_args = Utils.load_json(next_tool_args.strip())
        except JSONDecodeError as e:
            error_msg = f"Invalid JSON: {next_tool_args}"    
            try:
                from kindness_refactored.enhanced_tool_manager import EnhancedToolManager
                next_tool_args = cls.parse_malformed_json(
                    EnhancedToolManager.get_tool_args_for_tool(tool_name, required=True), 
                    next_tool_args
                )
            except Exception as e:
                raise Exception(error_msg)
        return next_tool_args

    @classmethod
    def inference(cls, messages: List[Dict[str, Any]], model: str, run_id: str = str(uuid4()), temperature: float = 0.0) -> dict:
        """Production inference with caching."""
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
            cleaned_msgs, model=model, temperature=temperature
        )    
        return next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages
    
    @classmethod
    def sanitise_text_resp(cls, text_resp: str) -> str:
        """Sanitize text response."""
        text_resp = re.sub("[\'\"]*next_thought[\'\"]*:", "next_thought:", text_resp)
        text_resp = re.sub("[\'\"]*next_tool_name[\'\"]*:", "next_tool_name:", text_resp)
        text_resp = re.sub("[\'\"]*next_tool_args[\'\"]*:", "next_tool_args:", text_resp)
        text_resp = re.sub("[\'\"]*observation[\'\"]*:", "observation:", text_resp)
        if ("next_thought" not in text_resp and 
            "next_tool_name:" in text_resp and 
            "next_tool_args:" in text_resp and 
            text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:") and 
            text_resp.find("next_tool_name:") > 10):
            logger.info(f"next_thought not found in {text_resp[:50]}, adding it")
            text_resp = "next_thought: " + text_resp
        if ("next_tool_name:" in text_resp and 
            "next_tool_args:" in text_resp and 
            text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:")):
            next_tool_name = text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n").strip("\'").strip("\"").strip()
            text_resp = re.sub(f"next_tool_name:[\'\" ]*{next_tool_name}[\'\" ]*", "next_tool_name: " + next_tool_name, text_resp)    
        return text_resp

    @classmethod
    def parse_response(cls, text_resp: str) -> tuple[str, Any, Any]:
        """Parse LLM response into structured format."""
        error_msg = None
        text_resp = text_resp.strip()
        text_resp = text_resp.split("observation:")[0]
        text_resp = text_resp.strip().strip("\n")
        text_resp = cls.sanitise_text_resp(text_resp)
        
        if ("next_thought:" in text_resp and 
            "next_tool_name:" in text_resp and 
            "next_tool_args:" in text_resp and 
            text_resp.find("next_thought:") < text_resp.find("next_tool_name:") and 
            text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:")):
            
            next_thought = text_resp.split("next_thought:")[1].split("next_tool_name:")[0].strip().strip("\n")
            next_tool_name_raw = text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n")
            next_tool_args_raw = text_resp.split("next_tool_args:")[1].strip().split("next_thought:")[0].strip().strip("\n")
            
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
            else:
                logger.error(f"We have no clue why parsing failed. Please check this \n{text_resp}\n")
            Utils.log_to_failed_messages(text_resp)
            return None, None, None, error_msg

        if len(next_tool_name) == 1:
            return next_thought, next_tool_name[0], next_tool_args[0], error_msg
            
        return next_thought, next_tool_name, next_tool_args, error_msg


# Global run_id variable
run_id = None
