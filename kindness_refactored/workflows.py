"""
Workflow functions for the Kindness AI agent framework.
"""

import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

from kindness_refactored.constants import (
    DEFAULT_TIMEOUT, PROBLEM_TYPE_CREATE, PROBLEM_TYPE_FIX, 
    GLM_MODEL_NAME, QWEN_MODEL_NAME, DEEPSEEK_MODEL_NAME, 
    AGENT_MODELS, MAX_FIX_TASK_STEPS
)
from kindness_refactored.enhanced_cot import EnhancedCOT
from kindness_refactored.fix_tool_manager import FixTaskEnhancedToolManager
from kindness_refactored.network import EnhancedNetwork
from kindness_refactored.prompts import (
    FIX_TASK_SYSTEM_PROMPT, FIX_TASK_FOR_FIXING_SYSTEM_PROMPT,
    FIX_TASK_INSTANCE_PROMPT_TEMPLATE, STOP_INSTRUCTION,
    DO_NOT_REPEAT_TOOL_CALLS, FORMAT_PROMPT_V0,
    PROBLEM_TYPE_CHECK_PROMPT, GENERATE_INITIAL_SOLUTION_PROMPT,
    GENERATE_INITIAL_TESTCASES_PROMPT, GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT,
    GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT, INFINITE_LOOP_CHECK_PROMPT,
    TESTCASES_CHECK_PROMPT
)
from kindness_refactored.utils import (
    ensure_git_initialized, set_env_for_agent, get_directory_tree,
    get_code_skeleton, post_process_instruction, determine_model_order,
    get_test_runner_and_mode
)

logger = logging.getLogger(__name__)


def process_fix_task(input_dict: Dict[str, Any]):
    """Main entry point for task processing and code modification."""
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
        patch_text = fix_task_solve_for_fixing_workflow(
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
    """Check if problem is CREATE or FIX type."""
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


def generate_test_files(problem_statement: str, files_to_test: str, code_skeleton: str) -> str:
    """Generate test files for the problem."""
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
    """Generate test cases using multi-step reasoning."""
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
    """Generate solution using multi-step reasoning."""
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
    """Extract and write files from solution string."""
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
    """Process create task workflow."""
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

    from kindness_refactored.enhanced_tool_manager import EnhancedToolManager
    tool_manager = EnhancedToolManager()
    patch = tool_manager.get_final_git_patch()
    return patch


def fix_task_solve_for_fixing_workflow(problem_statement: str, *, timeout: int, run_id_1: str,
    test_runner: str = "pytest", test_runner_mode: str = "FILE", n_max_steps = MAX_FIX_TASK_STEPS) -> str:
    """Main workflow for fixing tasks."""
    global run_id
    run_id = run_id_1
    cot = EnhancedCOT(latest_observations_to_keep=30)
    tool_manager = FixTaskEnhancedToolManager(
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
    system_prompt = FIX_TASK_FOR_FIXING_SYSTEM_PROMPT.format(
        tools_docs=tool_manager.get_tool_docs(),
        format_prompt=FORMAT_PROMPT_V0
    )
    instance_prompt = FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=problem_statement)
    
    start_time = time.time()
    logs: List[str] = []
    logs.append(f"cwd: {os.getcwd()}")
    logger.info(f"Starting workflow execution with {n_max_steps} max steps: timeout: {timeout} seconds : run_id: {run_id}")
    
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

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]
        
        messages.extend(cot.to_str())
        messages.append({"role": "system", "content": STOP_INSTRUCTION})
    
        if cot.is_thought_repeated():
            logger.info(f"[TEST_PATCH_FIND] Thought repeated, adding DO NOT REPEAT TOOL CALLS instruction")
            last_thought = cot.thoughts[-1]
            messages.append({
                "role": "user", 
                "content": DO_NOT_REPEAT_TOOL_CALLS.format(
                    previous_response=f"next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}"
                )
            })
    
        try:
            next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages = EnhancedNetwork.inference(
                messages, model=GLM_MODEL_NAME, run_id=run_id
            )
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_msg = f"\n\nERROR: {repr(e)} {traceback.format_exc()}"
            logger.error(f"Inference error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(
                next_thought=error_msg,
                next_tool_name="",
                next_tool_args={},
                observation="",
                is_error=True,
                raw_response=raw_text,
                total_attempts=total_attempts
            ), inference_error_counter=error_counter, request_data=messages)
            break
        
        logger.info(f"About to execute operation: {next_tool_name}")
       
        try:
            logger.info(f"next_thought: {next_thought}\nnext_tool_name: {next_tool_name}\nnext_tool_args: {next_tool_args}\n")
            if '"' in next_tool_name or "'" in next_tool_name:
                next_tool_name = next_tool_name.replace('"', '')
                next_tool_name = next_tool_name.replace("'", "")
                
            next_observation = tool_manager.get_tool(next_tool_name)(**next_tool_args) if next_tool_args else tool_manager.get_tool(next_tool_name)()
            logger.info(f"next_observation: {next_observation}")
            cot.add_action(EnhancedCOT.Action(
                next_thought=next_thought,
                next_tool_name=next_tool_name,
                next_tool_args=next_tool_args,
                observation=next_observation,
                is_error=False,
                raw_response=raw_text,
                total_attempts=total_attempts,
                inference_error_counter=error_counter,
                request_data=messages
            ))
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_traceback = traceback.format_exc()
            if isinstance(e, TypeError):
                error_msg = f"observation: {str(e)}"
            else:
                error_msg = f"observation: {repr(e)} {error_traceback}"
            logger.error(f"Tool error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(
                next_thought=next_thought,
                next_tool_name=next_tool_name,
                next_tool_args=next_tool_args,
                observation=error_msg,
                is_error=True,
                raw_response=raw_text,
                total_attempts=total_attempts,
                inference_error_counter=error_counter,
                request_data=messages
            ))
            continue
        
        if next_tool_name == "finish_for_fixing":
            logger.info('[CRITICAL] Workflow called finish_for_fixing operation')
            if next_observation == "finish_for_fixing":  # Only break if validation passed
                break
            
        print(f"[CRITICAL] Completed step {step + 1}, continuing to next step")
    else:
        cot.add_action(EnhancedCOT.Action(
            next_thought="global timeout reached",
            next_tool_name="",
            next_tool_args={},
            observation="",
            is_error=True
        ))
        logger.info(f"[CRITICAL] Workflow completed after reaching MAX_STEPS ({n_max_steps})")
        if n_max_steps < MAX_FIX_TASK_STEPS:
            return None
    
    logger.info(f"[CRITICAL] Workflow execution completed after {step + 1} steps")
    logger.info(f"[CRITICAL] About to generate final patch...")
    patch = tool_manager.get_final_git_patch()
    logger.info(f"Final Patch Generated..: Length: {len(patch)}")

    return patch


def fix_task_solve_workflow(problem_statement: str, *, timeout: int, run_id_1: str,
    test_runner: str = "pytest", test_runner_mode: str = "FILE", n_max_steps = MAX_FIX_TASK_STEPS) -> str:
    """Workflow for general fix tasks."""
    global run_id
    run_id = run_id_1
    cot = EnhancedCOT(latest_observations_to_keep=30)
    tool_manager = FixTaskEnhancedToolManager(
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
    system_prompt = FIX_TASK_SYSTEM_PROMPT.format(
        tools_docs=tool_manager.get_tool_docs(),
        format_prompt=FORMAT_PROMPT_V0
    )
    instance_prompt = FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=problem_statement)
    
    start_time = time.time()
    logs: List[str] = []
    logs.append(f"cwd: {os.getcwd()}")
    logger.info(f"Starting workflow execution with {n_max_steps} max steps: timeout: {timeout} seconds : run_id: {run_id}")
    
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

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]
        
        messages.extend(cot.to_str())

        messages.append({"role": "system", "content": STOP_INSTRUCTION})
    
        if cot.is_thought_repeated():
            logger.info(f"[TEST_PATCH_FIND] Thought repeated, adding DO NOT REPEAT TOOL CALLS instruction")
            last_thought = cot.thoughts[-1]
            messages.append({
                "role": "user", 
                "content": DO_NOT_REPEAT_TOOL_CALLS.format(
                    previous_response=f"next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}"
                )
            })
    
        try:
            next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages = EnhancedNetwork.inference(
                messages, model=GLM_MODEL_NAME, run_id=run_id
            )
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_msg = f"\n\nERROR: {repr(e)} {traceback.format_exc()}"
            logger.error(f"Inference error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(
                next_thought=error_msg,
                next_tool_name="",
                next_tool_args={},
                observation="",
                is_error=True,
                raw_response=raw_text,
                total_attempts=total_attempts
            ), inference_error_counter=error_counter, request_data=messages)
            break
        
        logger.info(f"About to execute operation: {next_tool_name}")
       
        try:
            logger.info(f"next_thought: {next_thought}\nnext_tool_name: {next_tool_name}\nnext_tool_args: {next_tool_args}\n")
            if '"' in next_tool_name or "'" in next_tool_name:
                next_tool_name = next_tool_name.replace('"', '')
                next_tool_name = next_tool_name.replace("'", "")
                
            next_observation = tool_manager.get_tool(next_tool_name)(**next_tool_args) if next_tool_args else tool_manager.get_tool(next_tool_name)()
            logger.info(f"next_observation: {next_observation}")
            cot.add_action(EnhancedCOT.Action(
                next_thought=next_thought,
                next_tool_name=next_tool_name,
                next_tool_args=next_tool_args,
                observation=next_observation,
                is_error=False,
                raw_response=raw_text,
                total_attempts=total_attempts,
                inference_error_counter=error_counter,
                request_data=messages
            ))
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_traceback = traceback.format_exc()
            if isinstance(e, TypeError):
                error_msg = f"observation: {str(e)}"
            else:
                error_msg = f"observation: {repr(e)} {error_traceback}"
            logger.error(f"Tool error: {error_msg}")
            cot.add_action(EnhancedCOT.Action(
                next_thought=next_thought,
                next_tool_name=next_tool_name,
                next_tool_args=next_tool_args,
                observation=error_msg,
                is_error=True,
                raw_response=raw_text,
                total_attempts=total_attempts,
                inference_error_counter=error_counter,
                request_data=messages
            ))
            continue
        
        if next_tool_name == "finish":
            logger.info('[CRITICAL] Workflow called finish operation')
            break
        print(f"[CRITICAL] Completed step {step + 1}, continuing to next step")
    else:
        cot.add_action(EnhancedCOT.Action(
            next_thought="global timeout reached",
            next_tool_name="",
            next_tool_args={},
            observation="",
            is_error=True
        ))
        logger.info(f"[CRITICAL] Workflow completed after reaching MAX_STEPS ({n_max_steps})")
        if n_max_steps < MAX_FIX_TASK_STEPS:
            return None
    
    logger.info(f"[CRITICAL] Workflow execution completed after {step + 1} steps")
    logger.info(f"[CRITICAL] About to generate final patch...")
    patch = tool_manager.get_final_git_patch()
    logger.info(f"Final Patch Generated..: Length: {len(patch)}")

    return patch


def generate_initial_solution(problem_statement: str, code_skeleton: str) -> str:
    """Generate initial solution for the problem."""
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


# Global run_id variable
run_id = None
