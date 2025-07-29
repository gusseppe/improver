from typing import TypedDict, Dict, Any, List, Optional
import yaml
import textwrap
import os
import time
from datetime import datetime
from langgraph.graph import StateGraph, END
from rich import print
from rich.panel import Panel
from rich.text import Text
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor
import json


class StandardState(TypedDict):
    model_code: str
    improved_code: str
    execution_output: str
    metrics: Dict[str, Any]
    iteration_count: int
    changes: List[str]
    previous_metrics: Dict[str, Any]
    improvement_history: List[Dict[str, Any]]
    dataset_description: Optional[Dict[str, Any]]
    start_time: float
    iteration_times: List[Dict[str, Any]]
    token_usage: Dict[str, int]


def prompt_improve_code() -> ChatPromptTemplate:
    """Enhanced prompt to improve ML training code with dataset description awareness."""
    examples = [
        {
            "input": """
            current_code: |
                import pandas as pd
                from sklearn.ensemble import RandomForestClassifier
                # load the old data
                dataset_folder = "datasets/financial" # it can any other like "datasets/healthcare"
                X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                model_old = RandomForestClassifier(random_state=42)
                model_old.fit(X_train_old, y_train_old)
                # Test the model on the old test set
                old_accuracy = model_old.score(X_test_old, y_test_old)
                print(f'Model trained and evaluated on the old distribution: {old_accuracy}')
            
            execution_output: |
                Model trained and evaluated on the old distribution: 0.913
            
            iteration: 1
            current_metrics:
                model_old_score:
                    on_old_data: 0.913
                    on_new_data: 0.717
                    
            dataset_description: {
                "NUM_SAMPLES": 2000,
                "NUMERICAL_FEATURES": ["Age", "Income", "Credit Score", "Loan Amount", "Loan Term", "Interest Rate", "Employment Length"],
                "CATEGORICAL_FEATURES": ["Home Ownership", "Marital Status", "Dependents"],
                "COLUMN_VALUES": {
                    "Home Ownership": {"0": "Rent", "1": "Own", "2": "Mortgage"},
                    "Marital Status": {"0": "Single", "1": "Married", "2": "Divorced", "3": "Widowed"},
                    "Loan Default": {"0": "No default", "1": "Default"}
                },
                "DATASET_TITLE": "Loan Default Prediction Data"
            }
            
            history: []
            """,
            "output": """
            improved_code: |
                import yaml
                import pandas as pd
                from sklearn.ensemble import GradientBoostingClassifier
                from sklearn.metrics import accuracy_score
                
                # Initialize metrics dictionary
                model_new_score = {
                    'on_new_data': 0.0,
                    'on_old_data': 0.0
                }
                
                # Load data from specified folder
                dataset_folder = "datasets/financial"
                X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                
                # Load new data
                X_train_new = pd.read_csv(f"{dataset_folder}/X_train_new.csv")
                y_train_new = pd.read_csv(f"{dataset_folder}/y_train_new.csv").squeeze("columns")
                X_test_new = pd.read_csv(f"{dataset_folder}/X_test_new.csv")
                y_test_new = pd.read_csv(f"{dataset_folder}/y_test_new.csv").squeeze("columns")
                
                # Train new model on combined data
                X_train = pd.concat([X_train_old, X_train_new])
                y_train = pd.concat([y_train_old, y_train_new])
                
                # Try GradientBoosting for better handling of loan default prediction
                # Good for both numerical features (Income, Loan Amount) and categorical (Home Ownership)
                model_new = GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    random_state=42
                )
                
                model_new.fit(X_train, y_train)
                
                # Evaluate new model on old test set
                new_score_old = accuracy_score(y_test_old, model_new.predict(X_test_old))
                print(f'New model trained and evaluated on old distribution: {new_score_old}')
                model_new_score['on_old_data'] = float(new_score_old)
                
                # Evaluate new model on new test set
                new_score_new = accuracy_score(y_test_new, model_new.predict(X_test_new))
                print(f'New model evaluated on new distribution: {new_score_new}')
                model_new_score['on_new_data'] = float(new_score_new)
                
                # Save metrics
                with open('metrics_baseline.yaml', 'w') as f:
                    yaml.dump({'model_new_score': model_new_score}, f)
            
            changes_made:
              - "Switched to GradientBoostingClassifier for better handling of loan default prediction with mixed feature types"
              - "Increased n_estimators to 200 for better model capacity"
              - "Added subsample=0.8 to reduce overfitting on the financial features"
              - "Implemented combined training on old and new data"
            """
        }
    ]
    
    for example in examples:
        example["input"] = textwrap.dedent(example["input"]).strip()
        example["output"] = textwrap.dedent(example["output"]).strip()

    system_prompt = """
    You are an expert ML engineer. Your task is to improve the given training code to make it more robust and perform better.
    
    Context: You have:
    1. Current model code
    2. Execution output with performance metrics
    3. Current iteration number
    4. Current performance metrics
    5. Dataset description with details about features and data distributions
    6. Improvement history from previous iterations (if available)
    
    Each iteration should try ONE of these improvements:
    1. Try a different sklearn model (RandomForest, GradientBoosting, XGBoost, etc.)
    2. Modify model hyperparameters significantly (e.g., n_estimators, max_depth)
    3. Implement data preprocessing (scaling, sampling, etc.)
    
    Focus on model changes that could improve performance on both distributions.
    
    Requirements:
    1. Save metrics to 'metrics_baseline.yaml'
    2. Use metrics format:
       model_new_score:          # IMPORTANT: Use this exact key name
           on_new_data: [score]
           on_old_data: [score]
    4. Use sklearn-compatible models
    5. Always train on combined old and new data
    6. Evaluate on both test sets
    
    CRITICAL REQUIREMENTS:
    1. YOU MUST TRAIN THE MODEL ON COMBINED DATA (X_train = pd.concat([X_train_old, X_train_new]))
    2. Save metrics with 'model_new_score' key
    3. Use the dataset description to inform your model choice and parameter settings
    4. If previous iterations exist in history, build upon the most successful approach
    
    IMPORTANT: You MUST save metrics with the key 'model_new_score', not 'model_old_score'.
    
    Response Format: Format your response as YAML with:
    improved_code: |
      [COMPLETE IMPLEMENTATION]
    changes_made:
      - [List significant model/parameter changes]
    
    Only provide the YAML-formatted output. No additional commentary.
    """
    
    system_prompt = textwrap.dedent(system_prompt).strip()
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        FewShotChatMessagePromptTemplate(
            example_prompt=ChatPromptTemplate.from_messages([
                ("human", "{input}"),
                ("ai", "{output}")
            ]),
            examples=examples
        ),
        ("human", "{input}"),
    ])
    
    return final_prompt


class StandardGraph:
    def __init__(self, llm, max_iterations=3, debug=False):
        """Initialize the standard improvement graph.
        
        Args:
            llm: The language model to use for generation
            max_iterations: Maximum number of improvement iterations to run
            debug: Whether to run in debug mode
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.graph = StateGraph(StandardState)
        self.build_plan()
        self.decision_procedure = self.graph.compile(debug=debug)
        self.start_time = None
        self.iteration_start_time = None
        self.token_counts = {"prompt": 0, "completion": 0, "total": 0}
        
    def build_plan(self):
        """Build the simple graph structure."""
        # Add nodes
        self.graph.add_node("improve_code", self.improve_code)
        self.graph.add_node("execute_code", self.execute_code)
        
        # Set entry point
        self.graph.set_entry_point("improve_code")
        
        # Add edges
        self.graph.add_edge("improve_code", "execute_code")
        
        # Add conditional edge
        self.graph.add_conditional_edges(
            "execute_code",
            self.should_end,
            {
                "continue": "improve_code",
                "end": END
            }
        )

    def _record_token_usage(self, state: StandardState, prompt_tokens: int, completion_tokens: int):
        """Record token usage in the state."""
        if "token_usage" not in state:
            state["token_usage"] = {"prompt": 0, "completion": 0, "total": 0}
            
        state["token_usage"]["prompt"] += prompt_tokens
        state["token_usage"]["completion"] += completion_tokens
        state["token_usage"]["total"] += prompt_tokens + completion_tokens
        
        # Also update the class-level counters
        self.token_counts["prompt"] += prompt_tokens
        self.token_counts["completion"] += completion_tokens
        self.token_counts["total"] += prompt_tokens + completion_tokens
        
        return state
        
    def _log_run_status(self, state: StandardState):
        """Log current run status."""
        if state.get('metrics'):
            metrics = state['metrics'].get('model_new_score', {})
            old_metrics = state.get('previous_metrics', {}).get('model_new_score', {})
            
            print("\nCurrent Performance:")
            print(f"Old Distribution: {metrics.get('on_old_data', 0):.4f}")
            print(f"New Distribution: {metrics.get('on_new_data', 0):.4f}")
            
            if old_metrics:
                print("\nImprovements:")
                old_diff = metrics.get('on_old_data', 0) - old_metrics.get('on_old_data', 0)
                new_diff = metrics.get('on_new_data', 0) - old_metrics.get('on_new_data', 0)
                print(f"Old Distribution: {old_diff:+.4f}")
                print(f"New Distribution: {new_diff:+.4f}")
                
                # Calculate distribution gap
                current_gap = metrics.get('on_old_data', 0) - metrics.get('on_new_data', 0)
                previous_gap = old_metrics.get('on_old_data', 0) - old_metrics.get('on_new_data', 0)
                gap_change = previous_gap - current_gap
                
                print(f"Distribution Gap: {current_gap:.4f} (changed by {gap_change:+.4f})")
            
            # Log token usage if available
            if "token_usage" in state:
                token_usage = state["token_usage"]
                print(f"\nToken Usage (Cumulative):")
                print(f"Prompt: {token_usage.get('prompt', 0)}")
                print(f"Completion: {token_usage.get('completion', 0)}")
                print(f"Total: {token_usage.get('total', 0)}")
    
    def _has_execution_errors(self, output: str) -> bool:
        """Check if execution output contains errors."""
        error_indicators = [
            'error', 'exception', 'failed', 'failure',
            'traceback', 'exitcode: 1'
        ]
        return any(indicator in output.lower() for indicator in error_indicators)
    
    def improve_code(self, state: StandardState) -> StandardState:
        """Improve the training code using standard prompt."""
        # Start timing this iteration
        self.iteration_start_time = time.time()
        
        # Get the best code from history if available
        best_code = state.get("model_code", "")
        best_metrics = {}
        
        # Attempt to find the best performing code from history
        if state.get("improvement_history"):
            # Sort by performance on new distribution (prioritize adapting to the new data)
            sorted_history = sorted(
                state.get("improvement_history", []),
                key=lambda x: x.get("metrics", {}).get("model_new_score", {}).get("on_new_data", 0),
                reverse=True
            )
            
            # Use the best performing code as the base
            if sorted_history:
                best_entry = sorted_history[0]
                best_code = best_entry.get("code", best_code)
                best_metrics = best_entry.get("metrics", {})
                print(f"\nUsing best performing code from history (iteration {best_entry.get('iteration', 0)})")
        
        # Prepare YAML input with enhanced context
        input_yaml = {
            "current_code": best_code,
            "execution_output": state.get("execution_output", ""),
            "iteration": state.get("iteration_count", 0) + 1,  # Add 1 because we're about to run this iteration
            "current_metrics": best_metrics if best_metrics else state.get("metrics", {}),
            "dataset_description": state.get("dataset_description", {}),
            "history": [
                {
                    "iteration": entry.get("iteration", 0),
                    "changes": entry.get("changes", []),
                    "metrics": entry.get("metrics", {})
                }
                for entry in state.get("improvement_history", [])
            ]
        }
        
        # Generate improved code
        prompt = prompt_improve_code()
        chain = prompt | self.llm
        
        # Estimate prompt tokens (rough approximation)
        prompt_text = yaml.dump(input_yaml)
        prompt_tokens = len(prompt_text) // 4
        
        # Invoke the model
        output = chain.invoke({"input": yaml.dump(input_yaml)}).content
        
        # Estimate completion tokens 
        completion_tokens = len(output) // 4
        
        # Record token usage in state
        state = self._record_token_usage(state, prompt_tokens, completion_tokens)
        
        # Parse YAML output
        try:
            result = yaml.safe_load(output)
            state["improved_code"] = result["improved_code"]
            state["changes"] = result.get("changes_made", [])
        except yaml.YAMLError:
            state["improved_code"] = output
            state["changes"] = []
        
        # Update iteration count
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        
        return state
    
    def execute_code(self, state: StandardState) -> StandardState:
        """Execute the improved code and evaluate results."""
        code = state["improved_code"]
        wrapped_code = f"```python\n{code}\n```"
        
        # Set up executor
        executor = LocalCommandLineCodeExecutor(timeout=30)  # Increased timeout
        code_executor_agent = ConversableAgent(
            "executor",
            llm_config=False,
            code_execution_config={"executor": executor}
        )
        
        # Execute code
        execution_output = code_executor_agent.generate_reply(
            messages=[{"role": "user", "content": wrapped_code}]
        )
        
        # Store execution output
        state["execution_output"] = execution_output
        
        # Check for execution errors
        if self._has_execution_errors(execution_output):
            print("\nExecution failed. Keeping previous metrics.")
            if state.get("previous_metrics"):
                state["metrics"] = state["previous_metrics"]
            return state
        
        # Parse metrics - accept both model_old_score and model_new_score
        try:
            with open('metrics_baseline.yaml', 'r') as f:
                metrics = yaml.safe_load(f)
                if not metrics or not isinstance(metrics, dict):
                    raise ValueError("Invalid metrics format")
                    
                # Store previous metrics before updating
                state["previous_metrics"] = state.get("metrics", {})
                
                # Ensure model_old_score is preserved if it exists in original metrics
                if "model_old_score" in state.get("metrics", {}):
                    # If original metrics only has model_old_score, make sure it's not lost
                    if "model_old_score" not in metrics:
                        metrics["model_old_score"] = state["metrics"]["model_old_score"]
                
                # Handle case where metrics has model_old_score but no model_new_score
                # This happens when the model saves metrics with the wrong key
                if "model_old_score" in metrics and "model_new_score" not in metrics:
                    print("\nConverting model_old_score to model_new_score in metrics")
                    metrics["model_new_score"] = metrics["model_old_score"]
                
                state["metrics"] = metrics
                
            # Update improvement history with more robust handling of metric keys
            # Convert execution metrics from output to standard format
            execution_metrics = {}
            if "model_new_score" in state["metrics"]:
                execution_metrics = state["metrics"]["model_new_score"]
            elif "model_old_score" in state["metrics"] and state["metrics"]["model_old_score"] != state.get("previous_metrics", {}).get("model_old_score", {}):
                # If only model_old_score was updated, use that
                execution_metrics = state["metrics"]["model_old_score"]
            
            # If we have metrics from execution, add to improvement history
            if execution_metrics:
                # Calculate iteration time
                iteration_end_time = time.time()
                iteration_time = iteration_end_time - (self.iteration_start_time or iteration_end_time)
                
                history_entry = {
                    "iteration": state.get("iteration_count", 0),
                    "metrics": {"model_new_score": execution_metrics},
                    "changes": state.get("changes", []),
                    "code": state.get("improved_code", ""),
                    "execution_time": iteration_time,
                    "token_usage": {
                        "prompt": state.get("token_usage", {}).get("prompt", 0),
                        "completion": state.get("token_usage", {}).get("completion", 0),
                        "total": state.get("token_usage", {}).get("total", 0)
                    }
                }
                
                if "improvement_history" not in state:
                    state["improvement_history"] = []
                    
                state["improvement_history"].append(history_entry)
                
                # Also track iteration time in a separate list
                if "iteration_times" not in state:
                    state["iteration_times"] = []
                    
                state["iteration_times"].append({
                    "iteration": state.get("iteration_count", 0),
                    "time": iteration_time
                })
                
                print(f"\nAdded entry to improvement history with metrics: {execution_metrics}")
                print(f"Iteration {state.get('iteration_count', 0)} time: {iteration_time:.2f} seconds")
                
        except Exception as e:
            print(f"\nError reading metrics: {str(e)}")
            if state.get("previous_metrics"):
                state["metrics"] = state["previous_metrics"]
        
        return state
    
    def should_end(self, state: StandardState) -> str:
        """Determine if improvement process should end."""
        # End after max_iterations
        if state.get("iteration_count", 0) >= self.max_iterations:
            print(f"\nReached maximum iterations ({state['iteration_count']}/{self.max_iterations})")
            return "end"
            
        # Get metrics
        current = state.get("metrics", {}).get('model_new_score', {})
        previous = state.get("previous_metrics", {}).get('model_new_score', {})
        
        # First iteration
        if not previous:
            state["previous_metrics"] = state.get("metrics", {})
            return "continue"
        
        # Calculate improvements
        improvement_new = current.get('on_new_data', 0) - previous.get('on_new_data', 0)
        improvement_old = current.get('on_old_data', 0) - previous.get('on_old_data', 0)
        
        print(f"\nImprovements this iteration:")
        print(f"New Distribution: {improvement_new:+.4f}")
        print(f"Old Distribution: {improvement_old:+.4f}")
        
        # Stop if no improvement or degradation on both distributions
        if improvement_new <= 0 and improvement_old <= 0:
            print("\nNo improvement detected on either distribution")
            return "end"
        
        state["previous_metrics"] = state.get("metrics", {})
        return "continue"
    
    def _export_results_to_yaml(self, state: StandardState, runtime_seconds: float) -> Dict:
        """
        Export the results to a standardized YAML-compatible dictionary format.
        
        Args:
            state: The final state after improvement
            runtime_seconds: Total runtime in seconds
            
        Returns:
            Dictionary with standardized result format
        """
        # Extract initial metrics
        initial_metrics = {}
        if state.get("metrics") and "model_old_score" in state.get("metrics", {}):
            old_metrics = state["metrics"]["model_old_score"]
            initial_metrics = {
                "old_distribution": old_metrics.get("on_old_data", 0),
                "new_distribution": old_metrics.get("on_new_data", 0)
            }
        
        # Extract final metrics - handle cases where metrics might be in different formats
        final_metrics = {}
        
        # If we have improvement history, get metrics from the last entry
        if state.get("improvement_history"):
            last_entry = state["improvement_history"][-1]
            if "metrics" in last_entry and "model_new_score" in last_entry["metrics"]:
                metrics = last_entry["metrics"]["model_new_score"]
                final_metrics = {
                    "old_distribution": metrics.get("on_old_data", 0),
                    "new_distribution": metrics.get("on_new_data", 0)
                }
        # Otherwise try to get from the state metrics
        elif state.get("metrics") and "model_new_score" in state.get("metrics", {}):
            metrics = state["metrics"]["model_new_score"]
            final_metrics = {
                "old_distribution": metrics.get("on_old_data", 0),
                "new_distribution": metrics.get("on_new_data", 0)
            }
        
        # Build improvement path with more robust handling
        improvement_path = []
        for i, entry in enumerate(state.get("improvement_history", [])):
            entry_metrics = {}
            # Get metrics from the entry if available
            if "metrics" in entry and "model_new_score" in entry["metrics"]:
                metrics_dict = entry["metrics"]["model_new_score"]
                entry_metrics = {
                    "old_distribution": metrics_dict.get("on_old_data", 0),
                    "new_distribution": metrics_dict.get("on_new_data", 0)
                }
            
            # Add reflection data based on changes
            reflection = f"Iteration {i+1} changes: "
            if entry.get("changes"):
                reflection += "; ".join(entry.get("changes", []))
            else:
                reflection += "No specific changes recorded"
            
            path_entry = {
                "iteration": i + 1,
                "code": entry.get("code", ""),
                "metrics": entry_metrics,
                "changes": entry.get("changes", []),
                "reflection": reflection,
                "execution_time": entry.get("execution_time", 0)
            }
            improvement_path.append(path_entry)
        
        # Get token usage from state
        token_usage = state.get("token_usage", {"total": self._estimate_token_usage(state)})
        
        if not final_metrics:
            # If no final metrics were computed, use initial metrics as fallback
            final_metrics = initial_metrics.copy()

        # Create the standardized output
        result = {
            "agent_name": "baseline",
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
                "evaluation_timestamp": self._get_current_timestamp()
            }
        }
        
        # Print debug information
        print(f"\nExporting results:")
        print(f"  Initial metrics: {initial_metrics}")
        print(f"  Final metrics: {final_metrics}")
        print(f"  Improvement path: {len(improvement_path)} entries")
        print(f"  Total tokens used: {token_usage.get('total', 0)}")
        
        return result
    
    def _estimate_token_usage(self, state: StandardState) -> int:
        """
        Estimate token usage based on the text length in the state.
        This is a rough approximation; for more accurate counting you would
        need to use a tokenizer specific to your model.
        
        Args:
            state: The final state
            
        Returns:
            Estimated token count
        """
        # Use the tracked token usage if available
        if "token_usage" in state and "total" in state["token_usage"]:
            return state["token_usage"]["total"]
        
        # Otherwise make a rough estimate
        # Simple approximation: ~1 token per 4 characters for English text
        char_count = 0
        
        # Count characters in model code
        char_count += len(state.get("model_code", ""))
        
        # Count characters in improved code
        char_count += len(state.get("improved_code", ""))
        
        # Count characters in execution outputs
        char_count += len(state.get("execution_output", ""))
        
        # Count characters in dataset description
        if state.get("dataset_description"):
            char_count += len(json.dumps(state.get("dataset_description", {})))
        
        # Approximate token count
        token_estimate = char_count // 4
        
        return token_estimate
    
    def _get_current_timestamp(self) -> str:
        """Get the current timestamp in ISO format."""
        return datetime.utcnow().isoformat() + "Z"
    
    def run(self, initial_state: Dict):
        """Run the improvement process.
        
        Args:
            initial_state: Dictionary containing the initial state, should include:
                - model_code: The original model code to improve
                - metrics: Optional initial metrics (model_old_score)
                - max_iterations: (Optional) Override the default max iterations
                - dataset_description: (Optional) Dataset description in JSON format
                
        Returns:
            The final state after improvement, including the improved code and metrics.
        """
        self.start_time = time.time()
        
        # Override max_iterations if provided in initial_state
        if "max_iterations" in initial_state:
            self.max_iterations = initial_state.pop("max_iterations")
            print(f"Max iterations set to: {self.max_iterations}")
        
        # Extract initial metrics if provided
        initial_metrics = initial_state.get("metrics", {})
        
        # Extract dataset description if provided
        dataset_description = initial_state.get("dataset_description", {})
        
        # Create properly typed initial state
        typed_state = StandardState(
            model_code=initial_state.get("model_code", ""),
            improved_code="",
            execution_output="",
            metrics=initial_metrics.copy() if initial_metrics else {},
            iteration_count=0,
            changes=[],
            previous_metrics={},
            improvement_history=[],
            dataset_description=dataset_description,
            start_time=self.start_time,
            iteration_times=[],
            token_usage={"prompt": 0, "completion": 0, "total": 0}
        )
        
        print("\n🚀 Starting Improved Baseline Model Improvement Process")
        print(f"Dataset: {dataset_description.get('DATASET_TITLE', 'Unknown')}")
        print(f"Features: {len(dataset_description.get('FEATURES', []))} total, {len(dataset_description.get('NUMERICAL_FEATURES', []))} numerical, {len(dataset_description.get('CATEGORICAL_FEATURES', []))} categorical")
        
        try:
            # Run the decision procedure
            final_output = None
            for output in self.decision_procedure.stream(
                typed_state,
                debug=False
            ):
                final_output = output
                for node_name, state in output.items():
                    # Log execution progress
                    print(f"\nExecuting Node: {node_name}")
                    print(f"Iteration: {state.get('iteration_count', 0)}")
                    
                    # Log changes
                    if state.get('changes'):
                        print("\nChanges made:")
                        for change in state['changes']:
                            print(f"- {change}")
                    
                    # Log performance metrics
                    self._log_run_status(state)
            
            # Get the final state
            final_state = final_output[list(final_output.keys())[-1]]
            
            # Calculate total runtime
            end_time = time.time()
            runtime_seconds = end_time - self.start_time
            
            # Generate final report
            print("\n📊 Improved Baseline Process Complete")
            print(f"\nTotal runtime: {runtime_seconds:.2f} seconds")
            
            # Summary of iterations
            if final_state.get("iteration_times"):
                print("\nIteration Times:")
                for iter_time in final_state["iteration_times"]:
                    print(f"  Iteration {iter_time['iteration']}: {iter_time['time']:.2f} seconds")
            
            # Final token usage
            if final_state.get("token_usage"):
                token_usage = final_state["token_usage"]
                print(f"\nFinal Token Usage:")
                print(f"  Prompt: {token_usage.get('prompt', 0)}")
                print(f"  Completion: {token_usage.get('completion', 0)}")
                print(f"  Total: {token_usage.get('total', 0)}")
            
            # Create standardized YAML output
            final_state["yaml_output"] = self._export_results_to_yaml(final_state, runtime_seconds)
            
            return final_state
            
        except Exception as e:
            print(f"Error in graph execution: {str(e)}")
            import traceback
            traceback.print_exc()
            raise