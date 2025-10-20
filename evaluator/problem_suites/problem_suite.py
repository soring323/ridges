"""Base class for problem suites."""

import os
import requests
import traceback
import utils.logger as logger
import validator.config as config

from uuid import UUID
from models.problem import Problem
from abc import ABC, abstractmethod
from typing import Any, List, Tuple
from utils.temp import create_temp_dir
from models.problem import ProblemTestResult
from evaluator.models import EvaluationRunException
from models.evaluation_run import EvaluationRunErrorCode
from evaluator.sandbox.sandbox_manager import Sandbox, SandboxManager



class ProblemSuite(ABC):
    """
    Abstract base class for a problem suite.
    Classes like PolyglotProblemSuite and SWEBenchProblemSuite inherit from this class.
    """

    @classmethod
    def find_problem_in_suites(cls, problem_name: str):
        """
        Search for a problem across all available problem suites.
        Returns a tuple of (suite_name, suite_instance) if found, None otherwise.
        """
        from evaluator.problem_suites.polyglot.polyglot_suite import PolyglotSuite
        from evaluator.problem_suites.swebench_verified.swebench_verified_suite import SWEBenchVerifiedSuite
        
        # Try Polyglot suite
        try:
            # Dataset is in evaluator/datasets/polyglot, not evaluator/problem_suites/polyglot
            datasets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
            polyglot_suite = PolyglotSuite(os.path.join(datasets_dir, "polyglot"))
            if polyglot_suite.has_problem_name(problem_name):
                return ("polyglot", polyglot_suite)
        except Exception:
            pass
        
        # Try SWE-bench Verified suite
        try:
            swebench_suite = SWEBenchVerifiedSuite(os.path.join(datasets_dir, "swebench_verified"))
            if swebench_suite.has_problem_name(problem_name):
                return ("swebench_verified", swebench_suite)
        except Exception:
            pass
        
        return None

    def __init__(self, dataset_path: str):
        self.problems = {}
        
        self.dataset_path = dataset_path
        self._load_problems(dataset_path)



    def _add_problem(self, problem: Problem) -> None:
        """
        Adds a problem to the problem suite.
        Should only be called from within the load_problems() implementation of a derived class.
        """
        
        if problem.name in self.problems:
            logger.fatal(f"Problem {problem.name} already exists in the suite")
        
        self.problems[problem.name] = problem

    @abstractmethod
    def _load_problems(self, dataset_path: str):
        """
        Loads all the problems from the given dataset path.
        Each inherited class must implement this method according to how their problem suite is structured.
        The implementation should call the _add_problem() method as many times as needed.
        """

        pass

    def has_problem_name(self, problem_name: str) -> bool:
        """
        Returns True if the problem suite has a problem with the given name.
        """
        
        return problem_name in self.problems

    def get_problem(self, problem_name: str) -> Problem:
        """
        Returns the problem with the given name.
        """
        
        return self.problems.get(problem_name)

    def get_problem_test_count(self, problem_name: str) -> int:
        """
        Returns the number of tests for the given problem.
        """
        problem = self.get_problem(problem_name)
        if problem and hasattr(problem, 'test_count'):
            return problem.test_count
        # Default estimate if test_count not available
        return 10



    @abstractmethod
    def copy_problem_files_to_directory(
        self,
        problem: Problem,
        dir: str,
        *,
        include_tests: bool = False
    ) -> None:
        """
        Copies the problem files to the given directory.
        Each inherited class must implement this method according to how their problem suite is structured.
        """
        
        pass



    def initialize_agent_sandbox(
        self,
        sandbox_manager: SandboxManager,
        problem: Problem,
        evaluation_run_id: UUID,
        agent_code: str,
        *,
        include_solution: bool = False
    ) -> Sandbox:
        # TODO ADAM: Docs
    
        try:
            def _on_mount(temp_dir: str):
                # Create /sandbox/agent.py
                with open(os.path.join(temp_dir, "agent.py"), "w") as f:
                    f.write(agent_code)
                
                # Create /sandbox/repo directory
                sandbox_repo_dir = os.path.join(temp_dir, "repo")
                os.mkdir(sandbox_repo_dir)

                # Copy problem files to /sandbox/repo
                self.copy_problem_files_to_directory(problem, sandbox_repo_dir)

                if include_solution:
                    # Create /sandbox/solution.diff
                    with open(os.path.join(temp_dir, "solution.diff"), "w") as f:
                        f.write(problem.solution_diff)



            return sandbox_manager.initialize_sandbox(
                name=f"agent-sandbox-{problem.name}",
                python_script_path=os.path.join(os.path.dirname(__file__), "AGENT_RUNNER.py"),
                input_data={
                    "problem_statement": problem.problem_statement
                },
                env_vars={
                    "RUN_ID": evaluation_run_id
                },
                on_mount=_on_mount,
            )
        except Exception as e:
            raise EvaluationRunException(
                EvaluationRunErrorCode.VALIDATOR_FAILED_INIT_AGENT,
                f"{EvaluationRunErrorCode.VALIDATOR_FAILED_INIT_AGENT.get_error_message()}: {e}\n\nTraceback:\n{traceback.format_exc()}"
            )



    def run_agent_sandbox(
        self,
        sandbox_manager: SandboxManager,
        agent_sandbox: Sandbox,
        timeout_seconds: int
    ) -> Tuple[str, str]:
        # TODO ADAM: Docs

        try:
            try:
                sandbox_result_with_logs = sandbox_manager.run_sandbox(agent_sandbox, timeout_seconds=timeout_seconds)
                timed_out = False
            # TODO ADAM: Docker bug
            # except TimeoutError:
            except requests.exceptions.ConnectionError:
                timed_out = True

            if timed_out:
                raise EvaluationRunException(
                    EvaluationRunErrorCode.AGENT_TIMEOUT_RUNNING_AGENT,
                    f"{EvaluationRunErrorCode.AGENT_TIMEOUT_RUNNING_AGENT.get_error_message()}: The agent exceeded the timeout of {timeout_seconds} seconds."
                )

            if not sandbox_result_with_logs.success:
                raise EvaluationRunException(
                    EvaluationRunErrorCode.AGENT_EXCEPTION_RUNNING_AGENT,
                    f"{EvaluationRunErrorCode.AGENT_EXCEPTION_RUNNING_AGENT.get_error_message()}: {sandbox_result_with_logs.error}\n\nTraceback:\n{sandbox_result_with_logs.traceback}"
                )
            
            return sandbox_result_with_logs.output, sandbox_result_with_logs.logs

        except EvaluationRunException:
            raise

        except Exception as e:
            raise EvaluationRunException(
                EvaluationRunErrorCode.VALIDATOR_FAILED_RUNNING_AGENT,
                f"{EvaluationRunErrorCode.VALIDATOR_FAILED_RUNNING_AGENT.get_error_message()}: {e}\n\nTraceback:\n{traceback.format_exc()}"
            )


    
    @abstractmethod
    def initialize_eval_sandbox(
        self,
        sandbox_manager: SandboxManager,
        problem: Problem,
        evaluation_run_id: UUID,
        patch: str
    ) -> Any:
        # TODO ADAM: Docs
        pass



    @abstractmethod
    def run_eval_sandbox(
        self,
        sandbox_manager: SandboxManager,
        eval_sandbox: Any
    ) -> Tuple[List[ProblemTestResult], str]:
        # TODO ADAM: Docs
        pass