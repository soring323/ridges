"""
Specialized tool manager for fix tasks.
"""

import ast
import hashlib
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

from kindness_refactored.enhanced_tool_manager import EnhancedToolManager
from kindness_refactored.utils import Utils
from kindness_refactored.visitors import FunctionVisitor, ClassVisitor

logger = logging.getLogger(__name__)


class FixTaskEnhancedToolManager(EnhancedToolManager):
    """Enhanced tool manager specifically for fix tasks."""

    def __init__(self, available_tools: Optional[list[str]] = [], test_runner: str = "pytest", test_runner_mode: str = "FILE"):
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
                
        self.tool_failure = {
            k: {j: 0 for j in self.Error.ErrorType.__members__} for k in self.TOOL_LIST.keys()
        }

        self.tool_invocations = {
            k: 0 for k in self.TOOL_LIST.keys()
        }

    def check_syntax_error(self, content: str, file_path: str = "<unknown>") -> tuple[bool, Optional[Exception]]:
        """Check for syntax errors in content."""
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            logger.error(f"Syntax error: {e}")
            return True, EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name,
                f"Syntax error. {str(e)}"
            )

    def _extract_test_file_paths_from_output(self, test_output: str) -> List[str]:
        """Extract test file paths from test output."""
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
    
    def _get_file_content(self, file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None, limit: int = 5000) -> str:
        """Get file content with optional filtering."""
        if search_term is not None and search_term != "":
            logger.debug(f"search_term specified: {search_term}, searching in v2")
            return self.search_in_specified_file_v2(file_path, search_term)
            
        func_ranges = self.get_function_ranges(file_path)
        if search_start_line is not None:
            for start, end, name in func_ranges:
                if start <= search_start_line <= end:
                    if start < search_start_line:
                        logger.debug(f"search start line {search_start_line} is between a function {start}-{end} for function {name}, setting to {start}")
                        search_start_line = start
        if search_end_line is not None:
            for start, end, name in func_ranges:
                if start <= search_end_line <= end:
                    if end > search_end_line:
                        logger.debug(f"search end line {search_end_line} is between a function {start}-{end} for function {name}, setting to {end}")
                        search_end_line = end
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

        return Utils.limit_strings(content, n=limit) if limit != -1 else content
    
    @EnhancedToolManager.tool
    def generate_validation_test(self, test_code: str, description: str = "validation test") -> str:
        """Generate a minimal validation test."""
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
    def get_file_content(self, file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None) -> str:
        """Retrieve file contents with optional filtering."""
        return self._get_file_content(file_path, search_start_line, search_end_line, search_term, limit=5000)
        
    @EnhancedToolManager.tool
    def save_file(self, file_path: str, content: str) -> str:
        """Write text content to specified filesystem location."""
        if "test" in file_path.lower() or "reproduce" in file_path.lower():
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,
                f"Error: You cannot use this tool to create test or files to reproduce the error."
            )
        return self._save(file_path, content)
    
    @EnhancedToolManager.tool   
    def get_approval_for_solution(self, solutions: list[str], selected_solution: int, reason_for_selection: str) -> str:
        """Get approval for proposed solution."""
        logger.info(f"solutions: {solutions}")
        logger.info(f"selected_solution: {selected_solution}")
        logger.info(f"reason_for_selection: {reason_for_selection}")
        parsed_solutions = []
        for solution in solutions:
            sols = re.split(r"(Solution \d+:)", solution)
            sols = [f"{sols[i]}{sols[i+1]}" for i in range(1, len(sols), 2)]  # Combine the split parts correctly
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
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name, error.message)
 
    @EnhancedToolManager.tool
    def get_functions(self, function_paths: List[str]) -> Dict[str, str]:
        """Get functions from a list of function paths."""
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
    def get_classes(self, class_paths: List[str]) -> Dict[str, str]:
        """Get classes from a list of class paths."""
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
    def run_repo_tests_for_fixing(self, file_paths: List[str]) -> str:
        """Run tests for the repository."""
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
                from kindness_refactored.utils import filepath_to_module
                modules = [filepath_to_module(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(modules)} -v 2"
                print("CMD: ", cmd)
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                output = (result.stdout or "") + (result.stderr or "")
            else:
                from kindness_refactored.utils import clean_filepath
                files_to_test = [clean_filepath(f, os.getcwd(), self.test_runner) for f in file_paths]
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
        """Search for text pattern across all .py files in the project."""
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
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name, 
                f"'{search_term}' not found in the codebase."
            )
        return output

    @EnhancedToolManager.tool
    def submit_solution(self, investigation_summary: str):
        """Submit solution after ALL repository tests pass."""
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
    
    def get_function_ranges(self, file_path: str) -> list[tuple[int, int, str]]:
        """Get function ranges from file."""
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
        """Return source code of function definitions containing search_term."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            logger.error(f"Error reading '{file_path}': {e}")
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

    def _extract_failing_test_names(self, test_output: str) -> List[Dict[str, str]]:
        """Extract failing test names from test output."""
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
    def search_in_specified_file_v2(self, file_path: str, search_term: str) -> str:
        """Locate text patterns within a specific file."""
        if not file_path.endswith(".py"):
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_FILE_PATH.name,
                f"Error: file '{file_path}' is not a python file."
            )
        return self._extract_function_matches(file_path, search_term)
    
    @EnhancedToolManager.tool
    def start_over(self, problem_with_old_approach: str, new_apprach_to_try: str):
        """Revert changes and start over with new approach."""
        logger.info("============Start Over============")
        os.system("git reset --hard")
        logger.info(f"problem_with_old_approach: {problem_with_old_approach}")
        logger.info(f"new_apprach_to_try: {new_apprach_to_try}")
        logger.info("===========================")
        return "Done, codebase reverted to initial state. You can start over with new approach."
        
    def get_final_git_patch(self) -> str:
        """Generate a clean unified diff (staged changes only)."""
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
        """Extract class attributes from AST node."""
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
        """Create or append a test function to the specified test file."""
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

    def _extract_test_data(self, file_path: str, test_name: str) -> Optional[str]:
        """Extract test data (setUp methods) from test file."""
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
    def run_repo_tests(self, file_paths: List[str]) -> str:
        """Run tests for the repository."""
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
                from kindness_refactored.utils import filepath_to_module
                modules = [filepath_to_module(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(modules)}"
                print("CMD: ", cmd)
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                output = (result.stdout or "") + (result.stderr or "")
            else:
                from kindness_refactored.utils import clean_filepath
                files_to_test = [clean_filepath(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(files_to_test)}"
                print("CMD: ", cmd)
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                output = (result.stdout or "") + (result.stderr or "")
        return output

    def _enhance_test_output(self, test_output: str, file_paths: List[str]) -> str:
        """Enhance test output with detailed failure information."""
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
    def run_code(self, content: str, file_path: str) -> str:
        """Run any python code."""
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
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES.name,
                f"Error:Cannot run, third party dependencies detected: {sorted(disallowed_modules)}\n"
            )

        result = subprocess.run(["python", file_path], capture_output=True, text=True, check=False, timeout=60)
        if result.returncode != 0:
            
            error_type = EnhancedToolManager.Error.ErrorType.RUNTIME_ERROR
            if "ImportError" in result.stderr:
                error_type = EnhancedToolManager.Error.ErrorType.IMPORT_ERROR
            if "ModuleNotFoundError" in result.stderr:
                error_type = EnhancedToolManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES
            raise EnhancedToolManager.Error(
                error_type,
                f"Error running code: {result.stderr}\n"
            )
        observation = f"{result.stdout}\n"
    
        return observation
    
    def _extract_test_function_code(self, file_path: str, test_name: str) -> Optional[str]:
        """Extract test function code from file."""
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
    def apply_code_edit(self, file_path: str, search: str, replace: str) -> str:
        """Perform targeted text replacement within source files."""
        if not self.is_solution_approved:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,
                f"Error: You cannot use this tool before you have approval from user on your proposed solution. Please call get_approval_for_solution tool first with list of proposed solutions."
            )
        if not os.path.exists(file_path):
            logger.error(f"file '{file_path}' does not exist.")
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name,
                f"Error: file '{file_path}' does not exist."
            )
        
        original = self._get_file_content(file_path, limit=-1)

        match original.count(search):
            case 0:
                logger.error(f"search string not found in file {file_path}. You need to share the exact code you want to replace.")
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
                logger.error(f"search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change.")
                raise EnhancedToolManager.Error(
                    EnhancedToolManager.Error.ErrorType.MULTIPLE_SEARCH_RESULTS_FOUND.name,
                    f"Error: search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change."
                )
    
    @EnhancedToolManager.tool
    def run_validation_test(self, file_path: str) -> str:
        """Run a validation test file."""
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
    def finish(self, investigation_summary: str):
        """Signals completion of the current workflow execution."""
        qa_response = {"is_patch_correct": "yes"}
        if qa_response.get("is_patch_correct", "no").lower() == "yes":
            return "finish"
        else: 
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.BUG_REPORT_REQUIRED.name,
                qa_response.get("analysis", "")
            )

    def _get_function_end_line(self, node, lines: List[str]) -> int:
        """Get the end line number of a function."""
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
    def finish_for_fixing(self, investigation_summary: str):
        """Signals completion of the current workflow execution for fixing tasks."""
        qa_response = {"is_patch_correct": "yes"}
        if qa_response.get("is_patch_correct", "no").lower() == "yes":
            return self.submit_solution(investigation_summary)
        else: 
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.BUG_REPORT_REQUIRED.name,
                qa_response.get("analysis", "")
            )
