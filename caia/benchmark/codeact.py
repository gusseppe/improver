"""
CodeAct Agent Implementation for Model Improvement (v2)

Based on the paper: "Executable Code Actions Elicit Better LLM Agents" (Wang et al., ICML 2024)
and the LlamaIndex CodeAct implementation pattern.

Key features:
1. Uses executable Python code as the unified action space
2. Maintains state between executions (variables persist)
3. Multi-turn interaction with full error context for self-debugging
4. Simpler, more robust code generation
"""

from typing import TypedDict, Dict, Any, List, Optional, Tuple
import yaml
import textwrap
import re
import io
import ast
import time
import json
import contextlib
import traceback
from datetime import datetime
from langgraph.graph import StateGraph, END
from rich import print
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class CodeActState(TypedDict):
    """State for the CodeAct-based improver agent."""
    model_code: str                         # Original model code
    improved_code: str                      # Current best code
    execution_output: str                   # Latest execution output
    metrics: Dict[str, Any]                 # Current metrics
    previous_metrics: Dict[str, Any]        # Previous metrics
    iteration_count: int                    # Current iteration
    improvement_history: List[Dict[str, Any]]
    
    # CodeAct-specific: chat memory for multi-turn
    chat_history: List[Dict[str, str]]      # Full conversation history
    current_code: str                       # Code being worked on this turn
    
    # Tracking
    dataset_description: Optional[Dict[str, Any]]
    token_usage: Dict[str, int]
    iteration_times: List[Dict[str, Any]]
    start_time: float
    iteration_start_time: Optional[float]
    
    # Error handling
    consecutive_failures: int
    turn_count: int                         # Turns within current iteration
    max_turns_per_iteration: int


class SimpleCodeExecutor:
    """
    A code executor that maintains state between executions.
    Based on LlamaIndex's SimpleCodeExecutor pattern.
    """
    
    def __init__(self):
        """Initialize with common ML imports pre-loaded."""
        self.globals = {
            "__builtins__": __builtins__,
        }
        self.locals = {}
        
    def reset(self):
        """Reset execution state."""
        self.globals = {
            "__builtins__": __builtins__,
        }
        self.locals = {}
    
    def execute(self, code: str) -> Tuple[bool, str]:
        """
        Execute Python code and capture output.
        
        Returns:
            Tuple of (success: bool, output: str)
        """
        stdout = io.StringIO()
        stderr = io.StringIO()
        
        output = ""
        success = True
        return_value = None
        
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    tree = ast.parse(code)
                    last_node = tree.body[-1] if tree.body else None
                    
                    if isinstance(last_node, ast.Expr):
                        last_line = code.rstrip().split("\n")[-1]
                        exec_code = code[:-len(last_line)] + "\n__result__ = " + last_line
                        exec(exec_code, self.globals, self.locals)
                        return_value = self.locals.get("__result__")
                    else:
                        exec(code, self.globals, self.locals)
                except SyntaxError:
                    exec(code, self.globals, self.locals)
            
            output = stdout.getvalue()
            if stderr.getvalue():
                output += "\nStderr: " + stderr.getvalue()
            
            if return_value is not None:
                output += f"\nReturn value: {return_value}"
                
        except Exception as e:
            success = False
            output = f"Error: {type(e).__name__}: {str(e)}\n"
            output += traceback.format_exc()
        
        return success, output


# System prompt for CodeAct
CODEACT_SYSTEM_PROMPT = """You are an expert ML engineer that improves machine learning models by writing and executing Python code.

Your task is to improve a model's performance on a new data distribution while maintaining performance on the old distribution.

IMPORTANT RULES:
1. Write Python code between <execute>...</execute> tags
2. The code will be executed and you'll see the output
3. Variables persist between executions within this session
4. Keep your code SIMPLE - avoid complex operations like GridSearchCV initially
5. Focus on ONE change at a time

Available libraries: pandas, numpy, sklearn, yaml

REQUIRED OUTPUT FORMAT:
Your code MUST save metrics to 'metrics_codeact.yaml' with this exact format:
```python
import yaml
model_new_score = {{'on_new_data': float(score_new), 'on_old_data': float(score_old)}}
with open('metrics_codeact.yaml', 'w') as f:
    yaml.dump({{'model_new_score': model_new_score}}, f)
```

Dataset folder structure:
- {{dataset_folder}}/X_train_old.csv, X_test_old.csv, y_train_old.csv, y_test_old.csv
- {{dataset_folder}}/X_train_new.csv, X_test_new.csv, y_train_new.csv, y_test_new.csv

Current context:
{context}

When you're done improving the model and metrics are saved, respond with "IMPROVEMENT COMPLETE" (no code tags)."""


def create_codeact_prompt(context: str) -> str:
    """Create the system prompt with context."""
    return CODEACT_SYSTEM_PROMPT.format(context=context)


class CodeActImprover:
    """
    CodeAct-based model improver using executable Python code as actions.
    
    This implementation follows the multi-turn conversation pattern where:
    1. LLM generates code
    2. Code is executed
    3. Output is fed back to LLM
    4. LLM can debug or continue
    """
    
    def __init__(self, llm, max_iterations=3, max_turns_per_iteration=5, debug=False):
        """
        Initialize the CodeAct improver.
        
        Args:
            llm: Language model to use
            max_iterations: Maximum improvement iterations
            max_turns_per_iteration: Max code execution turns per iteration
            debug: Enable debug output
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.max_turns_per_iteration = max_turns_per_iteration
        self.debug = debug
        self.code_executor = SimpleCodeExecutor()
        
        self.graph = StateGraph(CodeActState)
        self.build_plan()
        self.decision_procedure = self.graph.compile(debug=debug)
        
        self.start_time = None
        self.token_counts = {"prompt": 0, "completion": 0, "total": 0}
    
    def build_plan(self):
        """Build the CodeAct graph."""
        self.graph.add_node("generate_or_debug", self.generate_or_debug_step)
        self.graph.add_node("execute_code", self.execute_code_step)
        self.graph.add_node("check_completion", self.check_completion_step)
        
        self.graph.set_entry_point("generate_or_debug")
        
        self.graph.add_edge("generate_or_debug", "execute_code")
        
        self.graph.add_conditional_edges(
            "execute_code",
            self.route_after_execution,
            {
                "continue": "generate_or_debug",
                "check": "check_completion",
                "end": END
            }
        )
        
        self.graph.add_conditional_edges(
            "check_completion",
            self.should_continue_iterations,
            {
                "next_iteration": "generate_or_debug",
                "end": END
            }
        )
    
    def _parse_code(self, response: str) -> Optional[str]:
        """Extract code from <execute>...</execute> tags."""
        matches = re.findall(r"<execute>(.*?)</execute>", response, re.DOTALL)
        if matches:
            return "\n\n".join(match.strip() for match in matches)
        return None
    
    def _record_token_usage(self, state: CodeActState, prompt_tokens: int, completion_tokens: int):
        """Record token usage."""
        if "token_usage" not in state:
            state["token_usage"] = {"prompt": 0, "completion": 0, "total": 0}
        
        state["token_usage"]["prompt"] += prompt_tokens
        state["token_usage"]["completion"] += completion_tokens
        state["token_usage"]["total"] += prompt_tokens + completion_tokens
        
        self.token_counts["prompt"] += prompt_tokens
        self.token_counts["completion"] += completion_tokens
        self.token_counts["total"] += prompt_tokens + completion_tokens
        
        return state
    
    def _build_context(self, state: CodeActState) -> str:
        """Build context string for the prompt."""
        context_parts = []
        
        # Dataset info
        ds = state.get("dataset_description", {})
        if ds:
            context_parts.append(f"Dataset: {ds.get('DATASET_TITLE', 'Unknown')}")
            if ds.get('NUMERICAL_FEATURES'):
                context_parts.append(f"Numerical features: {ds.get('NUMERICAL_FEATURES')}")
            if ds.get('CATEGORICAL_FEATURES'):
                context_parts.append(f"Categorical features: {ds.get('CATEGORICAL_FEATURES')}")
        
        # Current metrics
        metrics = state.get("metrics", {})
        if metrics.get("model_old_score"):
            old = metrics["model_old_score"]
            context_parts.append(f"\nBaseline performance (model trained on old data only):")
            context_parts.append(f"  - On old distribution: {old.get('on_old_data', 'N/A')}")
            context_parts.append(f"  - On new distribution: {old.get('on_new_data', 'N/A')}")
        
        if metrics.get("model_new_score"):
            new = metrics["model_new_score"]
            context_parts.append(f"\nCurrent improved model performance:")
            context_parts.append(f"  - On old distribution: {new.get('on_old_data', 'N/A')}")
            context_parts.append(f"  - On new distribution: {new.get('on_new_data', 'N/A')}")
        
        # Iteration info
        context_parts.append(f"\nIteration: {state.get('iteration_count', 0) + 1}/{self.max_iterations}")
        context_parts.append(f"Turn within iteration: {state.get('turn_count', 0) + 1}/{self.max_turns_per_iteration}")
        
        # Original code reference
        context_parts.append(f"\nOriginal model code for reference:\n```python\n{state.get('model_code', '')}\n```")
        
        return "\n".join(context_parts)
    
    def generate_or_debug_step(self, state: CodeActState) -> CodeActState:
        """Generate new code or debug based on conversation history."""
        print(f"\n💻 CODEACT: TURN {state.get('turn_count', 0) + 1}")
        
        # Start timing if new iteration
        if state.get("turn_count", 0) == 0:
            state["iteration_start_time"] = time.time()
        
        # Build messages
        context = self._build_context(state)
        system_prompt = create_codeact_prompt(context)
        
        messages = [SystemMessage(content=system_prompt)]
        
        # Add chat history
        for entry in state.get("chat_history", []):
            if entry["role"] == "user":
                messages.append(HumanMessage(content=entry["content"]))
            else:
                messages.append(AIMessage(content=entry["content"]))
        
        # If this is the first turn, add initial instruction
        if not state.get("chat_history"):
            initial_msg = f"""Please improve the ML model. The goal is to improve performance on the new distribution while maintaining performance on the old distribution.

Start by:
1. Loading the data from the dataset folder
2. Training a model on combined old + new training data
3. Evaluating on both test sets
4. Saving metrics to 'metrics_codeact.yaml'

Dataset folder: Look at the original code to find the dataset_folder path.

Write your code between <execute>...</execute> tags."""
            messages.append(HumanMessage(content=initial_msg))
            state["chat_history"] = [{"role": "user", "content": initial_msg}]
        
        # Estimate tokens
        prompt_text = "\n".join(m.content for m in messages)
        prompt_tokens = len(prompt_text) // 4
        
        # Generate response
        response = self.llm.invoke(messages)
        response_text = response.content
        
        completion_tokens = len(response_text) // 4
        state = self._record_token_usage(state, prompt_tokens, completion_tokens)
        
        # Parse code
        code = self._parse_code(response_text)
        
        if code:
            state["current_code"] = code
            print(f"\nGenerated {len(code)} characters of code")
            if self.debug:
                print(f"Code preview: {code[:200]}...")
        else:
            state["current_code"] = ""
            if "IMPROVEMENT COMPLETE" in response_text.upper():
                print("\nAgent signaled completion")
            else:
                print("\nNo code generated")
        
        # Add response to chat history
        state["chat_history"].append({"role": "assistant", "content": response_text})
        
        return state
    
    def execute_code_step(self, state: CodeActState) -> CodeActState:
        """Execute the generated code."""
        print("\n🔄 CODEACT: EXECUTING")
        
        code = state.get("current_code", "")
        
        if not code:
            state["execution_output"] = "No code to execute"
            state["execution_success"] = True  # Not a failure, just no code
            return state
        
        # Execute
        success, output = self.code_executor.execute(code)
        
        state["execution_output"] = output
        state["execution_success"] = success
        state["turn_count"] = state.get("turn_count", 0) + 1
        
        print(f"\nExecution {'succeeded' if success else 'failed'}")
        print(f"Output: {output[:300]}..." if len(output) > 300 else f"Output: {output}")
        
        # Add execution result to chat history for context
        if success:
            exec_msg = f"Code executed successfully. Output:\n{output}"
        else:
            exec_msg = f"Code execution failed with error:\n{output}\n\nPlease fix the error and try again."
        
        state["chat_history"].append({"role": "user", "content": exec_msg})
        
        # If successful, try to read metrics
        if success:
            try:
                with open('metrics_codeact.yaml', 'r') as f:
                    metrics = yaml.safe_load(f)
                    if metrics and isinstance(metrics, dict) and "model_new_score" in metrics:
                        state["previous_metrics"] = state.get("metrics", {}).copy()
                        
                        # Preserve baseline metrics
                        if "model_old_score" in state.get("metrics", {}):
                            metrics["model_old_score"] = state["metrics"]["model_old_score"]
                        
                        state["metrics"] = metrics
                        state["improved_code"] = code
                        state["consecutive_failures"] = 0
                        
                        print(f"\n✅ Metrics updated: {metrics.get('model_new_score')}")
            except FileNotFoundError:
                pass  # Metrics not saved yet
            except Exception as e:
                print(f"Error reading metrics: {e}")
        else:
            state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
        
        return state
    
    def route_after_execution(self, state: CodeActState) -> str:
        """Decide next step after execution."""
        # Check if max turns reached
        if state.get("turn_count", 0) >= self.max_turns_per_iteration:
            print(f"\nMax turns ({self.max_turns_per_iteration}) reached for this iteration")
            return "check"
        
        # Check if improvement complete signal
        last_response = ""
        if state.get("chat_history"):
            for entry in reversed(state["chat_history"]):
                if entry["role"] == "assistant":
                    last_response = entry["content"]
                    break
        
        if "IMPROVEMENT COMPLETE" in last_response.upper() and not state.get("current_code"):
            return "check"
        
        # Check if we have valid metrics
        if state.get("execution_success") and state.get("metrics", {}).get("model_new_score"):
            # We have metrics, check if LLM wants to continue or is done
            if not state.get("current_code"):
                return "check"
        
        # Continue the conversation
        return "continue"
    
    def check_completion_step(self, state: CodeActState) -> CodeActState:
        """Check if iteration is complete and record results."""
        print("\n📊 CODEACT: CHECKING COMPLETION")
        
        # Record iteration time
        if state.get("iteration_start_time"):
            iteration_time = time.time() - state["iteration_start_time"]
            if "iteration_times" not in state:
                state["iteration_times"] = []
            state["iteration_times"].append({
                "iteration": state.get("iteration_count", 0) + 1,
                "time": iteration_time,
                "turns": state.get("turn_count", 0)
            })
            print(f"Iteration time: {iteration_time:.2f}s, Turns: {state.get('turn_count', 0)}")
        
        # Record to improvement history
        if "improvement_history" not in state:
            state["improvement_history"] = []
        
        metrics = state.get("metrics", {}).get("model_new_score", {})
        state["improvement_history"].append({
            "iteration": state.get("iteration_count", 0) + 1,
            "metrics": state.get("metrics", {}),
            "code": state.get("improved_code", ""),
            "turns": state.get("turn_count", 0),
            "execution_time": state["iteration_times"][-1]["time"] if state.get("iteration_times") else 0,
            "token_usage": state.get("token_usage", {}).copy()
        })
        
        # Increment iteration and reset turn count
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        state["turn_count"] = 0
        
        # Reset code executor for next iteration (fresh state)
        self.code_executor.reset()
        
        # Clear chat history for next iteration but keep a summary
        if state.get("metrics", {}).get("model_new_score"):
            summary = f"Previous iteration achieved: {state['metrics']['model_new_score']}"
            state["chat_history"] = []  # Will be repopulated with fresh context
        
        return state
    
    def should_continue_iterations(self, state: CodeActState) -> str:
        """Decide if we should do another iteration."""
        if state.get("iteration_count", 0) >= self.max_iterations:
            print(f"\nMax iterations ({self.max_iterations}) reached")
            return "end"
        
        # Check if we're making progress
        current = state.get("metrics", {}).get("model_new_score", {})
        previous = state.get("previous_metrics", {}).get("model_new_score", {})
        
        if current and previous:
            improvement = current.get("on_new_data", 0) - previous.get("on_new_data", 0)
            if improvement <= -0.05:  # Significant regression
                print(f"\nRegression detected ({improvement:.4f}), stopping")
                return "end"
        
        return "next_iteration"
    
    def _export_results_to_yaml(self, state: CodeActState, runtime_seconds: float) -> Dict:
        """Export results to standardized format."""
        initial_metrics = {}
        if state.get("metrics") and "model_old_score" in state.get("metrics", {}):
            old_metrics = state["metrics"]["model_old_score"]
            initial_metrics = {
                "old_distribution": old_metrics.get("on_old_data", 0),
                "new_distribution": old_metrics.get("on_new_data", 0)
            }
        
        final_metrics = {}
        if state.get("metrics") and "model_new_score" in state.get("metrics", {}):
            metrics = state["metrics"]["model_new_score"]
            final_metrics = {
                "old_distribution": metrics.get("on_old_data", 0),
                "new_distribution": metrics.get("on_new_data", 0)
            }
        
        improvement_path = []
        for entry in state.get("improvement_history", []):
            metrics = entry.get("metrics", {}).get("model_new_score", {})
            improvement_path.append({
                "iteration": entry.get("iteration", 0),
                "code": entry.get("code", ""),
                "metrics": {
                    "old_distribution": metrics.get("on_old_data", 0),
                    "new_distribution": metrics.get("on_new_data", 0)
                },
                "turns": entry.get("turns", 0),
                "execution_time": entry.get("execution_time", 0)
            })
        
        token_usage = state.get("token_usage", self.token_counts)
        
        return {
            "agent_name": "codeact",
            "initial_code": state.get("model_code", ""),
            "initial_metrics": initial_metrics,
            "improvement_path": improvement_path,
            "final_code": state.get("improved_code", ""),
            "final_metrics": final_metrics,
            "runtime_statistics": {
                "total_time_seconds": runtime_seconds,
                "iterations": state.get("iteration_count", 0),
                "tokens_used": token_usage.get("total", 0),
                "prompt_tokens": token_usage.get("prompt", 0),
                "completion_tokens": token_usage.get("completion", 0),
                "iteration_times": state.get("iteration_times", []),
                "evaluation_timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
    
    def run(self, initial_state: Dict) -> Dict:
        """
        Run the CodeAct improvement process.
        
        Args:
            initial_state: Dictionary with:
                - model_code: Original model code
                - metrics: Initial metrics (model_old_score)
                - max_iterations: (Optional) Override
                - dataset_description: (Optional) Dataset info
        
        Returns:
            Final state with improved code and metrics
        """
        self.start_time = time.time()
        
        # Override settings
        if "max_iterations" in initial_state:
            self.max_iterations = initial_state.pop("max_iterations")
        
        # Reset code executor
        self.code_executor.reset()
        
        initial_metrics = initial_state.get("metrics", {})
        dataset_description = initial_state.get("dataset_description", {})
        
        # Create typed state
        typed_state = CodeActState(
            model_code=initial_state.get("model_code", ""),
            improved_code="",
            execution_output="",
            metrics=initial_metrics.copy() if initial_metrics else {},
            previous_metrics={},
            iteration_count=0,
            improvement_history=[],
            chat_history=[],
            current_code="",
            dataset_description=dataset_description,
            token_usage={"prompt": 0, "completion": 0, "total": 0},
            iteration_times=[],
            start_time=self.start_time,
            iteration_start_time=None,
            consecutive_failures=0,
            turn_count=0,
            max_turns_per_iteration=self.max_turns_per_iteration
        )
        
        print("\n🚀 Starting CodeAct Model Improvement")
        print(f"Dataset: {dataset_description.get('DATASET_TITLE', 'Unknown')}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Max turns per iteration: {self.max_turns_per_iteration}")
        
        try:
            final_output = None
            for output in self.decision_procedure.stream(typed_state):
                final_output = output
            
            final_state = final_output[list(final_output.keys())[-1]]
            runtime_seconds = time.time() - self.start_time
            
            print("\n" + "="*50)
            print("📊 CodeAct Process Complete")
            print(f"Total runtime: {runtime_seconds:.2f}s")
            print(f"Iterations completed: {final_state.get('iteration_count', 0)}")
            print(f"Total tokens: {final_state.get('token_usage', {}).get('total', 0)}")
            
            if final_state.get("metrics", {}).get("model_new_score"):
                m = final_state["metrics"]["model_new_score"]
                print(f"Final metrics - Old: {m.get('on_old_data', 'N/A')}, New: {m.get('on_new_data', 'N/A')}")
            
            final_state["yaml_output"] = self._export_results_to_yaml(final_state, runtime_seconds)
            
            return final_state
            
        except Exception as e:
            print(f"Error in CodeAct execution: {e}")
            import traceback
            traceback.print_exc()
            raise
