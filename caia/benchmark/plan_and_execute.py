from typing import TypedDict, Dict, Any, List, Tuple, Literal, Annotated, Optional
import yaml
import textwrap
import os
import time
import json
from datetime import datetime
from langgraph.graph import StateGraph, END
from rich import print
from rich.panel import Panel
from rich.text import Text
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from operator import add
from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor


class PlanExecuteState(TypedDict):
    """State for the Plan-and-Execute ML code improvement agent."""
    model_code: str                         # Original model code
    improved_code: str                      # Current improved code
    execution_output: str                   # Output from executing the code
    metrics: Dict[str, Any]                 # Current metrics
    previous_metrics: Dict[str, Any]        # Metrics from previous iteration
    iteration_count: int                    # Current iteration number
    improvement_history: List[Dict[str, Any]]  # History of all improvements
    
    # Plan-and-Execute specific state components
    plan: List[str]                         # List of steps in the plan
    current_step_index: int                 # Index of the current step
    past_steps: Annotated[List[Tuple[str, str]], add]  # History of executed steps (step, result)
    changes: List[str]                      # Current changes being made
    
    # Additional components for standardized tracking
    dataset_description: Optional[Dict[str, Any]]  # Dataset description in JSON format
    token_usage: Dict[str, int]             # Token usage tracking
    iteration_times: List[Dict[str, Any]]   # Time tracking per iteration
    start_time: float                       # Start time of the process
    iteration_start_time: Optional[float]   # Start time of current iteration
    
    # Error handling components
    consecutive_failures: int               # Count of consecutive execution failures
    last_successful_state: Dict[str, Any]   # Last successful execution state
    max_failures_allowed: int               # Maximum allowed consecutive failures


def prompt_create_plan() -> ChatPromptTemplate:
    """Create a prompt for the planning step."""
    examples = [
        {
            "input": """
            model_code: |
                import pandas as pd
                from sklearn.ensemble import RandomForestClassifier
                
                # load the old data
                dataset_folder = "datasets/financial" #it can be any folder
                X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                
                model_old = RandomForestClassifier(random_state=42)
                model_old.fit(X_train_old, y_train_old)
                
                # Test the model on the old test set
                old_accuracy = model_old.score(X_test_old, y_test_old)
                print(f'Model trained and evaluated on the old distribution: {old_accuracy}')
                
            metrics: 
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
            """,
            "output": """
            plan:
              - "Load both old and new data from datasets/financial"
              - "Implement StandardScaler for preprocessing numerical features"
              - "Apply one-hot encoding for categorical features"
              - "Train a GradientBoostingClassifier on combined old and new data"
              - "Evaluate model on both old and new test sets"
              - "Save metrics using the model_new_score key format"
            rationale: |
              The initial model is using only the old data, which explains the performance gap between distributions (0.913 vs 0.717). Based on the dataset description, we have both numerical and categorical features that need proper preprocessing. My plan addresses this by:
              1. Including new data in training
              2. Adding appropriate preprocessing for both numerical and categorical features
              3. Using GradientBoosting which often handles shifts between distributions better than RandomForest
              4. Ensuring proper evaluation on both distributions
              5. Saving metrics in the correct format for comparison
            """
        }
    ]
    for example in examples:
        example["input"] = textwrap.dedent(example["input"]).strip()
        example["output"] = textwrap.dedent(example["output"]).strip()
    
    system_prompt = """
    You are an expert ML engineer tasked with creating a step-by-step plan to improve ML code.
    
    Given:
    1. Current model code
    2. Current performance metrics
    3. Dataset description with feature information
    
    Create a 3-5 step plan to improve the model's performance, especially on new data distributions.
    
    Your plan should:
    - Be specific and actionable
    - Focus on proven techniques to improve model robustness
    - Include data preprocessing, model selection, or hyperparameter tuning
    - Ensure the model is trained on combined old and new datasets
    - Include proper metrics saving with the key 'model_new_score'
    - Consider the specific features in the dataset description
    
    Your plan must ensure the model:
    - Maintains good performance on old data
    - Improves performance on new data
    - Reduces the gap between distributions
    - Appropriately handles both numerical and categorical features
    
    Response Format: Format your response as YAML with:
    
    plan:
      - "Step 1 description"
      - "Step 2 description"
      - "Step 3 description"
      ...
    rationale: |
      [Explanation of your plan and why it should work]
    
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


def prompt_execute_step() -> ChatPromptTemplate:
    """Create a prompt for executing a specific step in the plan."""
    examples = [
        {
            "input": """
            model_code: |
                import pandas as pd
                from sklearn.ensemble import RandomForestClassifier
                
                # load the old data
                dataset_folder = "datasets/financial" #it can be any folder like datasets/eligibility
                X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                
                model_old = RandomForestClassifier(random_state=42)
                model_old.fit(X_train_old, y_train_old)
                
                # Test the model on the old test set
                old_accuracy = model_old.score(X_test_old, y_test_old)
                print(f'Model trained and evaluated on the old distribution: {old_accuracy}')
            
            current_step: "Train a GradientBoostingClassifier on combined old and new data"
            
            full_plan:
              - "Load both old and new data from datasets/financial"
              - "Implement StandardScaler for preprocessing both datasets"
              - "Train a GradientBoostingClassifier on combined old and new data"
              - "Evaluate model on both old and new test sets"
              - "Save metrics using the model_new_score key format"
              
            past_steps_results:
              - step: "Load both old and new data from datasets/financial"
                result: "Successfully loaded old and new datasets from 'datasets/financial'"
              - step: "Implement StandardScaler for preprocessing both datasets"
                result: "Added StandardScaler preprocessing for both old and new data"
                
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
            """,
            "output": """
            improved_code: |
                import yaml
                import pandas as pd
                from sklearn.preprocessing import StandardScaler
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
                
                # Scale data using StandardScaler
                scaler = StandardScaler()
                X_train_old_scaled = scaler.fit_transform(X_train_old)
                X_test_old_scaled = scaler.transform(X_test_old)
                X_train_new_scaled = scaler.fit_transform(X_train_new)
                X_test_new_scaled = scaler.transform(X_test_new)
                
                # Combine old and new data for training
                X_train_combined = pd.concat([pd.DataFrame(X_train_old_scaled), pd.DataFrame(X_train_new_scaled)])
                y_train_combined = pd.concat([y_train_old, y_train_new])
                
                # Train GradientBoostingClassifier on combined data
                model_new = GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    random_state=42
                )
                
                model_new.fit(X_train_combined, y_train_combined)
                
                # Evaluate model on old test set
                old_score = accuracy_score(y_test_old, model_new.predict(X_test_old_scaled))
                print(f'New model evaluated on old distribution: {old_score}')
                model_new_score['on_old_data'] = float(old_score)
                
                # Evaluate model on new test set
                new_score = accuracy_score(y_test_new, model_new.predict(X_test_new_scaled))
                print(f'New model evaluated on new distribution: {new_score}')
                model_new_score['on_new_data'] = float(new_score)
                
                # Save metrics
                with open('metrics_plan_execute.yaml', 'w') as f:
                    yaml.dump({'model_new_score': model_new_score}, f)
            
            changes_made:
              - "Trained GradientBoostingClassifier with n_estimators=150"
              - "Used learning_rate=0.1 for moderate boosting speed"
              - "Set max_depth=5 to prevent overfitting"
              - "Added subsample=0.8 for stochastic gradient boosting"
              - "Combined old and new data for training"
            
            step_result: "Successfully trained GradientBoostingClassifier on combined data with optimized hyperparameters for loan default prediction"
            """
        }
    ]
    for example in examples:
        example["input"] = textwrap.dedent(example["input"]).strip()
        example["output"] = textwrap.dedent(example["output"]).strip()
    
    system_prompt = """
    You are an expert ML engineer implementing a specific step in an improvement plan.
    
    Context: You have:
    1. Current model code
    2. Current step to implement
    3. Full plan for context
    4. Results of past steps
    5. Dataset description with feature information
    
    Your task is to implement ONLY the current step, building on the results of past steps.
    
    Requirements:
    1. ONLY implement the current step
    2. Build on the code from past steps
    3. Save metrics to 'metrics_plan_execute.yaml'
    4. Use metrics format:
       model_new_score:         # IMPORTANT: Use this exact key name
           on_new_data: [score]
           on_old_data: [score]
    6. Train on combined old and new data
    7. Evaluate on both test sets
    8. Use the dataset description to inform your implementation choices
    
    IMPORTANT CONSTRAINTS:
    1. DO NOT use GridSearchCV or any hyperparameter search that takes a long time to run
    2. Keep the model simple and efficient - avoid complex operations that might timeout. Only use sklearn models.
    3. Make sure your Python code syntax is correct without YAML markers inside it
    4. YAML markers like 'changes_made:' should NOT appear in your Python code
    
    Response Format: Format your response as YAML with:
    
    improved_code: |
      [COMPLETE IMPLEMENTATION]
    changes_made:
      - [List significant changes made for this step]
    step_result: [Brief summary of what was accomplished]
    
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


def prompt_replan() -> ChatPromptTemplate:
    """Create a prompt for replanning after executing a step."""
    system_prompt = """
    You are an expert ML engineer reviewing a step in an improvement plan.
    
    Context: You have:
    1. Original improvement plan
    2. Results of executed steps so far
    3. Current performance metrics
    4. Initial performance metrics
    5. Dataset description (if provided)
    
    Your task is to decide:
    1. If the plan should be continued as is
    2. If the plan should be modified
    3. If the process should end
    
    Guidelines:
    - Continue the plan if steps are proceeding successfully
    - Modify the plan if the results suggest a better approach
    - End the process if all steps have been completed successfully or if the performance has plateaued
    
    Response Format: Format your response as YAML with:
    
    decision: [continue, modify, end]
    updated_plan:
      - [Updated step 1]
      - [Updated step 2]
      ...
    rationale: |
      [Explanation of your decision]
    
    Only provide the YAML-formatted output. No additional commentary.
    """
    
    system_prompt = textwrap.dedent(system_prompt).strip()
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
        Original plan:
        {original_plan}
        
        Steps executed:
        {past_steps}
        
        Current metrics:
        {current_metrics}
        
        Initial metrics:
        {initial_metrics}
        
        Remaining steps:
        {remaining_steps}
        
        Dataset description:
        {dataset_description}
        
        Based on the progress so far, should we continue with the original plan, modify it, or end the process?
        """),
    ])
    
    return final_prompt


class PlanAndExecuteGraph:
    def __init__(self, llm, max_iterations=3, max_failures=3, debug=False):
        """Initialize the Plan-and-Execute improvement graph.
        
        Args:
            llm: The language model to use for generation
            max_iterations: Maximum number of improvement iterations to run
            max_failures: Maximum number of consecutive execution failures allowed
            debug: Whether to run in debug mode
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.max_failures = max_failures
        self.graph = StateGraph(PlanExecuteState)
        self.debug = debug
        self.build_plan()
        self.decision_procedure = self.graph.compile(debug=debug)
        self.start_time = None
        self.iteration_start_time = None
        self.token_counts = {"prompt": 0, "completion": 0, "total": 0}
        
    def build_plan(self):
        """Build the Plan-and-Execute graph structure."""
        # Add nodes for each step in the process
        self.graph.add_node("planner", self.plan_step)
        self.graph.add_node("executor", self.execute_step)
        self.graph.add_node("evaluate", self.evaluate_step)
        self.graph.add_node("replanner", self.replan_step)
        
        # Set entry point
        self.graph.set_entry_point("planner")
        
        # Add edges between nodes
        self.graph.add_edge("planner", "executor")
        self.graph.add_edge("executor", "evaluate")
        self.graph.add_edge("evaluate", "replanner")
        
        # Add conditional edges from replanner
        self.graph.add_conditional_edges(
            "replanner",
            self.should_continue,
            {
                "continue": "executor",
                "end": END
            }
        )
    
    def _record_token_usage(self, state: PlanExecuteState, prompt_tokens: int, completion_tokens: int):
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
    
    def plan_step(self, state: PlanExecuteState) -> PlanExecuteState:
        """Create a plan for improving the ML code."""
        print("\n🧠 PLANNING STEP")
        
        # Record the start time of the overall process
        if self.start_time is None:
            self.start_time = time.time()
            state["start_time"] = self.start_time
        
        # Prepare input for planning prompt
        input_yaml = {
            "model_code": state.get("model_code", ""),
            "metrics": state.get("metrics", {}),
            "dataset_description": state.get("dataset_description", {})
        }
        
        # Estimate prompt tokens (rough approximation)
        prompt_text = yaml.dump(input_yaml)
        prompt_tokens = len(prompt_text) // 4
        
        # Generate plan
        prompt = prompt_create_plan()
        chain = prompt | self.llm
        output = chain.invoke({"input": yaml.dump(input_yaml)}).content
        
        # Estimate completion tokens 
        completion_tokens = len(output) // 4
        
        # Record token usage in state
        state = self._record_token_usage(state, prompt_tokens, completion_tokens)
        
        # Parse YAML output
        try:
            result = yaml.safe_load(output)
            state["plan"] = result.get("plan", [])
            # Store rationale in state for debugging
            state["plan_rationale"] = result.get("rationale", "")
            state["current_step_index"] = 0
            state["past_steps"] = []
            
            # Log the plan
            print("\nImprovement Plan:")
            for i, step in enumerate(state["plan"]):
                print(f"{i+1}. {step}")
            print(f"\nRationale: {state['plan_rationale']}")
            
            # Print token usage
            print(f"\nPlanning token usage:")
            print(f"  Prompt: {prompt_tokens}")
            print(f"  Completion: {completion_tokens}")
            print(f"  Total: {prompt_tokens + completion_tokens}")
            
        except yaml.YAMLError as e:
            print(f"Error parsing planning output: {e}")
            # Fallback planning
            state["plan"] = [
                "Load both old and new data", 
                "Add preprocessing steps",
                "Try a different model architecture",
                "Evaluate on both data distributions",
                "Save metrics properly"
            ]
            state["current_step_index"] = 0
            state["past_steps"] = []
            
        return state
    
    def execute_step(self, state: PlanExecuteState) -> PlanExecuteState:
        """Execute the current step of the plan."""
        current_index = state.get("current_step_index", 0)
        
        # Start timing this step
        self.iteration_start_time = time.time()
        state["iteration_start_time"] = self.iteration_start_time
        
        if current_index >= len(state.get("plan", [])):
            # No more steps to execute
            return state
        
        current_step = state["plan"][current_index]
        print(f"\n⚙️ EXECUTING STEP {current_index + 1}: {current_step}")
        
        # Format past steps for the prompt
        past_steps_results = []
        for step, result in state.get("past_steps", []):
            past_steps_results.append({
                "step": step,
                "result": result
            })
        
        # Check if this is the first step, and there's no improved code yet
        last_code = state.get("improved_code", "")
        if not last_code:
            last_code = state.get("model_code", "")
        
        # Prepare input for execution prompt
        input_yaml = {
            "model_code": last_code,  # Use last improved code or original
            "current_step": current_step,
            "full_plan": state.get("plan", []),
            "past_steps_results": past_steps_results,
            "dataset_description": state.get("dataset_description", {})
        }
        
        # Estimate prompt tokens
        prompt_text = yaml.dump(input_yaml)
        prompt_tokens = len(prompt_text) // 4
        
        # Generate code for this step
        prompt = prompt_execute_step()
        chain = prompt | self.llm
        output = chain.invoke({"input": yaml.dump(input_yaml)}).content
        
        # Estimate completion tokens 
        completion_tokens = len(output) // 4
        
        # Record token usage in state
        state = self._record_token_usage(state, prompt_tokens, completion_tokens)
        
        # Parse YAML output
        try:
            result = yaml.safe_load(output)
            
            # Extract only the code, not the full YAML
            if result and isinstance(result, dict):
                code = result.get("improved_code", "")
                changes = result.get("changes_made", [])
                step_result = result.get("step_result", "Step completed")
                
                # Validate the code doesn't contain YAML markers
                if "changes_made:" in code or "step_result:" in code:
                    # The YAML is embedded in the code, need to extract just the Python code
                    code_lines = code.split('\n')
                    filtered_lines = []
                    for line in code_lines:
                        if not (line.strip().startswith('changes_made:') or 
                                line.strip().startswith('step_result:') or
                                line.strip().startswith('-')):
                            filtered_lines.append(line)
                    code = '\n'.join(filtered_lines)
                
                state["improved_code"] = code
                state["changes"] = changes
            else:
                # Fallback if parsing failed but didn't throw exception
                state["improved_code"] = output
                state["changes"] = []
                step_result = "Parsing issue with LLM output"
            
            # Store this step's result
            state["past_steps"] = state.get("past_steps", []) + [(current_step, step_result)]
            
            # Increment step counter
            state["current_step_index"] = current_index + 1
            
            # Log changes
            print("\nChanges made in this step:")
            for change in state.get("changes", []):
                print(f"- {change}")
                
            # Print token usage
            print(f"\nExecution token usage:")
            print(f"  Prompt: {prompt_tokens}")
            print(f"  Completion: {completion_tokens}")
            print(f"  Total: {prompt_tokens + completion_tokens}")
            
        except yaml.YAMLError as e:
            print(f"Error parsing execution output: {e}")
            # Try to salvage code even if YAML parsing fails
            try:
                # Look for code between triple backticks if YAML parsing failed
                import re
                code_match = re.search(r'```python\n(.*?)\n```', output, re.DOTALL)
                if code_match:
                    state["improved_code"] = code_match.group(1)
                else:
                    # Look for something that looks like Python code
                    if "import " in output and "def " in output or "class " in output:
                        state["improved_code"] = output
                    else:
                        # Keep previous code if we can't extract anything useful
                        state["improved_code"] = last_code
            except Exception:
                state["improved_code"] = last_code
            
            state["changes"] = []
            state["past_steps"] = state.get("past_steps", []) + [(current_step, "Error in execution")]
            state["current_step_index"] = current_index + 1
            
        return state
    
    def evaluate_step(self, state: PlanExecuteState) -> PlanExecuteState:
        """Evaluate the executed step by running the code with error handling."""
        print("\n📊 EVALUATING STEP")
        
        code = state.get("improved_code", "")
        
        if not code:
            return state
            
        # Check for common issues in the code that would cause syntax errors
        if "changes_made:" in code or "step_result:" in code:
            print("\nDetected YAML markers in code. Attempting to clean...")
            code_lines = code.split('\n')
            filtered_lines = []
            in_yaml_section = False
            for line in code_lines:
                if line.strip().startswith('changes_made:') or line.strip().startswith('step_result:'):
                    in_yaml_section = True
                    continue
                elif in_yaml_section and line.strip().startswith('-'):
                    continue
                else:
                    in_yaml_section = False
                    filtered_lines.append(line)
            code = '\n'.join(filtered_lines)
            state["improved_code"] = code
        
        wrapped_code = f"```python\n{code}\n```"
        
        # Set up executor with increased timeout for GridSearchCV operations
        executor = LocalCommandLineCodeExecutor(timeout=60)  # Increased timeout
        code_executor_agent = ConversableAgent(
            "executor",
            llm_config=False,
            code_execution_config={"executor": executor}
        )
        
        # Execute code
        try:
            execution_output = code_executor_agent.generate_reply(
                messages=[{"role": "user", "content": wrapped_code}]
            )
            
            # Store execution output
            state["execution_output"] = execution_output
            print(f"\nExecution output:\n{execution_output}")
            
            # Check for execution errors
            if self._has_execution_errors(execution_output):
                print("\nExecution failed.")
                
                # Increment consecutive failures count
                state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
                
                # Log the failure count
                print(f"⚠️ Consecutive failures: {state['consecutive_failures']}/{self.max_failures}")
                
                # Check if we've reached the failure limit
                if state["consecutive_failures"] >= self.max_failures:
                    print(f"❌ Reached maximum consecutive failures ({self.max_failures}). Stopping execution attempts.")
                    
                    # Use the last successful state if available
                    if state.get("last_successful_state"):
                        print("📥 Restoring last successful state...")
                        for key, value in state["last_successful_state"].items():
                            if key not in ["consecutive_failures", "last_successful_state"]:
                                state[key] = value
                                
                        # Keep the consecutive_failures count
                        state["execution_output"] += f"\n\nReached maximum failures ({self.max_failures}). Restored last successful state."
                    else:
                        print("⚠️ No successful state found. Using input metrics.")
                        # Keep the initial metrics
                        if state.get("previous_metrics"):
                            state["metrics"] = state["previous_metrics"]
                else:
                    print("🔄 Using previous metrics for this attempt.")
                    # Keep using previous metrics for now
                    if state.get("previous_metrics"):
                        state["metrics"] = state["previous_metrics"]
                        
                return state
            
            # If execution succeeded, reset the failure count
            state["consecutive_failures"] = 0
            
        except Exception as e:
            print(f"\nException during code execution: {str(e)}")
            state["execution_output"] = f"Execution error: {str(e)}"
            
            # Increment consecutive failures count
            state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
            
            # Check if we've reached the failure limit
            if state["consecutive_failures"] >= self.max_failures:
                print(f"❌ Reached maximum consecutive failures ({self.max_failures}). Stopping execution attempts.")
                
                # Use the last successful state if available
                if state.get("last_successful_state"):
                    print("📥 Restoring last successful state...")
                    for key, value in state["last_successful_state"].items():
                        if key not in ["consecutive_failures", "last_successful_state"]:
                            state[key] = value
                            
                    # Keep the consecutive_failures count
                    state["execution_output"] += f"\n\nReached maximum failures ({self.max_failures}). Restored last successful state."
                
            if state.get("previous_metrics"):
                state["metrics"] = state["previous_metrics"]
                
            return state
        
        # Parse metrics
        try:
            with open('metrics_plan_execute.yaml', 'r') as f:
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
                if "model_old_score" in metrics and "model_new_score" not in metrics:
                    print("\nConverting model_old_score to model_new_score in metrics")
                    metrics["model_new_score"] = metrics["model_old_score"]
                
                state["metrics"] = metrics
                
                # Store this as the last successful state (create a deep copy)
                state["last_successful_state"] = json.loads(json.dumps({
                    "improved_code": state["improved_code"],
                    "execution_output": state["execution_output"],
                    "metrics": state["metrics"],
                    "changes": state.get("changes", [])
                }))
                
            # Update improvement history
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
                iteration_time = iteration_end_time - (state.get("iteration_start_time") or iteration_end_time)
                
                history_entry = {
                    "iteration": state.get("iteration_count", 0) + 1,
                    "metrics": {"model_new_score": execution_metrics},
                    "changes": state.get("changes", []),
                    "code": state.get("improved_code", ""),
                    "execution_time": iteration_time
                }
                
                if "improvement_history" not in state:
                    state["improvement_history"] = []
                    
                state["improvement_history"].append(history_entry)
                
                # Also track iteration time in a separate list
                if "iteration_times" not in state:
                    state["iteration_times"] = []
                    
                state["iteration_times"].append({
                    "iteration": state.get("iteration_count", 0) + 1,
                    "time": iteration_time
                })
                
                print(f"\nAdded entry to improvement history with metrics: {execution_metrics}")
                print(f"Step execution time: {iteration_time:.2f} seconds")
                
                # Increment iteration count
                state["iteration_count"] = state.get("iteration_count", 0) + 1
                
        except Exception as e:
            print(f"\nError reading metrics: {str(e)}")
            if state.get("previous_metrics"):
                state["metrics"] = state["previous_metrics"]
        
        # Log metrics
        self._log_metrics(state.get("metrics", {}), state.get("previous_metrics", {}))
        
        return state
    
    def replan_step(self, state: PlanExecuteState) -> PlanExecuteState:
        """Review progress and decide whether to continue, modify plan, or end."""
        print("\n🔄 REPLANNING STEP")
        
        # Check if we've hit the max consecutive failures
        if state.get("consecutive_failures", 0) >= self.max_failures:
            state["replan_decision"] = "end"
            print(f"\nReached maximum consecutive failures ({self.max_failures}). Ending process.")
            return state
        
        # If we've reached the end of the plan, consider if we should end
        if state.get("current_step_index", 0) >= len(state.get("plan", [])):
            state["replan_decision"] = "end"
            print("\nAll steps in plan completed. Ending process.")
            return state
            
        # If we've reached the max iterations, end
        if state.get("iteration_count", 0) >= self.max_iterations:
            state["replan_decision"] = "end"
            print(f"\nReached maximum iterations ({self.max_iterations}). Ending process.")
            return state
        
        # Format past steps for the prompt
        past_steps_formatted = ""
        for i, (step, result) in enumerate(state.get("past_steps", [])):
            past_steps_formatted += f"{i+1}. {step} -> {result}\n"
        
        # Format remaining steps
        remaining_steps = []
        for i in range(state.get("current_step_index", 0), len(state.get("plan", []))):
            remaining_steps.append(state["plan"][i])
        
        remaining_steps_formatted = "\n".join([f"- {step}" for step in remaining_steps])
        
        # Format plan for the prompt
        original_plan_formatted = "\n".join([f"- {step}" for step in state.get("plan", [])])
        
        # Prepare input for replan prompt
        input_data = {
            "original_plan": original_plan_formatted,
            "past_steps": past_steps_formatted,
            "current_metrics": yaml.dump(state.get("metrics", {})),
            "initial_metrics": yaml.dump(state.get("previous_metrics", {})),
            "remaining_steps": remaining_steps_formatted,
            "dataset_description": yaml.dump(state.get("dataset_description", {}))
        }
        
        # Estimate prompt tokens
        prompt_text = yaml.dump(input_data)
        prompt_tokens = len(prompt_text) // 4
        
        try:
            # Generate replanning decision
            prompt = prompt_replan()
            chain = prompt | self.llm
            
            # Add error handling for LLM call
            try:
                output = chain.invoke(input_data)
                if output is None:
                    raise ValueError("Received None response from LLM")
                    
                output_content = output.content
                if not output_content:
                    raise ValueError("Empty response content from LLM")
            except Exception as llm_error:
                print(f"Error in LLM call: {str(llm_error)}")
                # Default to continue
                state["replan_decision"] = "continue"
                return state
            
            # Estimate completion tokens 
            completion_tokens = len(output_content) // 4
            
            # Record token usage in state
            state = self._record_token_usage(state, prompt_tokens, completion_tokens)
                
            # Clean the output if it starts with markdown code block markers
            if output_content.startswith("```"):
                lines = output_content.split("\n")
                # Remove code block markers
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                output_content = "\n".join(lines)
            
            # Parse YAML output with error handling
            try:
                result = yaml.safe_load(output_content)
                if not result or not isinstance(result, dict):
                    raise yaml.YAMLError("Invalid YAML structure")
                
                decision = result.get("decision", "continue")
                
                if decision == "modify":
                    # Update the plan
                    updated_plan = result.get("updated_plan")
                    if updated_plan and isinstance(updated_plan, list):
                        state["plan"] = updated_plan
                        state["current_step_index"] = 0  # Start from the beginning of the new plan
                        print("\nPlan modified:")
                        for i, step in enumerate(state["plan"]):
                            print(f"{i+1}. {step}")
                
                state["replan_decision"] = decision
                print(f"\nReplanning decision: {decision}")
                print(f"Rationale: {result.get('rationale', 'No rationale provided')}")
                
                # Print token usage
                print(f"\nReplanning token usage:")
                print(f"  Prompt: {prompt_tokens}")
                print(f"  Completion: {completion_tokens}")
                print(f"  Total: {prompt_tokens + completion_tokens}")
                
            except yaml.YAMLError as e:
                print(f"Error parsing replanning output: {e}")
                # Default to continue if parsing fails
                state["replan_decision"] = "continue"
                
        except Exception as e:
            print(f"Unexpected error in replanning: {str(e)}")
            # Default to continue
            state["replan_decision"] = "continue"
            
        return state
    
    def should_continue(self, state: PlanExecuteState) -> str:
        """Determine if the process should continue or end."""
        # Check if we've hit the max consecutive failures
        if state.get("consecutive_failures", 0) >= self.max_failures:
            print(f"\nReached maximum consecutive failures ({self.max_failures}). Ending process.")
            return "end"
            
        decision = state.get("replan_decision", "continue")
        
        if decision == "end":
            return "end"
        else:
            return "continue"
    
    def _has_execution_errors(self, output: str) -> bool:
        """Check if execution output contains errors."""
        error_indicators = [
            'error', 'exception', 'failed', 'failure',
            'traceback', 'exitcode: 1'
        ]
        return any(indicator in output.lower() for indicator in error_indicators)
    
    def _log_metrics(self, metrics: Dict, previous_metrics: Dict = None):
        """Log current metrics with optional comparison to previous."""
        if not metrics or not (metrics.get('model_new_score') or metrics.get('model_old_score')):
            print("No metrics available")
            return
        
        # Try to get metrics from model_new_score, fall back to model_old_score
        current = metrics.get('model_new_score', metrics.get('model_old_score', {}))
        
        print("\nCurrent Performance:")
        print(f"Old Distribution: {current.get('on_old_data', 0):.4f}")
        print(f"New Distribution: {current.get('on_new_data', 0):.4f}")
        
        if previous_metrics:
            previous = previous_metrics.get('model_new_score', previous_metrics.get('model_old_score', {}))
            
            if previous:
                print("\nImprovements:")
                old_diff = current.get('on_old_data', 0) - previous.get('on_old_data', 0)
                new_diff = current.get('on_new_data', 0) - previous.get('on_new_data', 0)
                print(f"Old Distribution: {old_diff:+.4f}")
                print(f"New Distribution: {new_diff:+.4f}")
                
                # Calculate distribution gap
                current_gap = current.get('on_old_data', 0) - current.get('on_new_data', 0)
                previous_gap = previous.get('on_old_data', 0) - previous.get('on_new_data', 0)
                gap_change = previous_gap - current_gap
                
                print(f"Distribution Gap: {current_gap:.4f} (changed by {gap_change:+.4f})")
                
        # Log token usage if available
        if "token_usage" in metrics:
            token_usage = metrics["token_usage"]
            print(f"\nToken Usage (Cumulative):")
            print(f"Prompt: {token_usage.get('prompt', 0)}")
            print(f"Completion: {token_usage.get('completion', 0)}")
            print(f"Total: {token_usage.get('total', 0)}")
    
    def _export_results_to_yaml(self, state: PlanExecuteState, runtime_seconds: float) -> Dict:
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
        
        # Extract final metrics
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
        
        # Build improvement path with more comprehensive metadata
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
            
            # Extract step information for reflection
            reflection = f"Plan and Execute iteration {i+1}"
            if state.get("past_steps") and i < len(state.get("past_steps")):
                step_info = state["past_steps"][i]
                reflection = f"Step: {step_info[0]}\nResult: {step_info[1]}"
            
            path_entry = {
                "iteration": i + 1,
                "code": entry.get("code", ""),
                "metrics": entry_metrics,
                "changes": entry.get("changes", []),
                "reflection": reflection,
                "execution_time": entry.get("execution_time", 0)
            }
            improvement_path.append(path_entry)
        
        # Get token usage from state or class counters
        token_usage = state.get("token_usage", {
            "prompt": self.token_counts["prompt"],
            "completion": self.token_counts["completion"],
            "total": self.token_counts["total"]
        })
        
        # Create the standardized output
        if not final_metrics:
            final_metrics = initial_metrics.copy()
            
        result = {
            "agent_name": "plan_execute",
            "initial_code": state.get("model_code", ""),
            "initial_metrics": initial_metrics,
            "improvement_plan": state.get("plan", []),
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
                "evaluation_timestamp": self._get_current_timestamp(),
                "execution_attempts": state.get("iteration_count", 0) + state.get("consecutive_failures", 0),
                "successful_executions": state.get("iteration_count", 0),
                "failed_executions": state.get("consecutive_failures", 0)
            }
        }
        
        # Print debug information
        print(f"\nExporting results:")
        print(f"  Initial metrics: {initial_metrics}")
        print(f"  Final metrics: {final_metrics}")
        print(f"  Improvement path: {len(improvement_path)} entries")
        print(f"  Total tokens used: {token_usage.get('total', 0)}")
        
        return result
    
    def _estimate_token_usage(self, state: PlanExecuteState) -> int:
        """Estimate token usage for reporting purposes."""
        # If token_usage is tracked, use that
        if "token_usage" in state and "total" in state["token_usage"]:
            return state["token_usage"]["total"]
            
        # Simple approximation: ~1 token per 4 characters for English text
        char_count = 0
        
        # Count characters in model code
        char_count += len(state.get("model_code", ""))
        
        # Count characters in improved code
        char_count += len(state.get("improved_code", ""))
        
        # Count characters in execution outputs
        char_count += len(state.get("execution_output", ""))
        
        # Count characters in plan and past steps
        for step in state.get("plan", []):
            char_count += len(step)
        
        for step, result in state.get("past_steps", []):
            char_count += len(step) + len(result)
        
        # Count characters in dataset description if present
        if state.get("dataset_description"):
            char_count += len(json.dumps(state.get("dataset_description", {})))
        
        # Approximate token count
        token_estimate = char_count // 4
        
        return token_estimate
    
    def _get_current_timestamp(self) -> str:
        """Get the current timestamp in ISO format."""
        return datetime.utcnow().isoformat() + "Z"
        
    def run(self, initial_state: Dict) -> Dict:
        """Run the Plan-and-Execute improvement process with enhanced error handling.
        
        Args:
            initial_state: Dictionary containing the initial state, should include:
                - model_code: The original model code to improve
                - metrics: Optional initial metrics (model_old_score)
                - max_iterations: (Optional) Override the default max iterations
                - max_failures: (Optional) Override the default max failures allowed
                - dataset_description: (Optional) Dataset description in JSON format
                
        Returns:
            The final state after improvement, including the improved code and metrics.
        """
        self.start_time = time.time()
        
        # Override max_iterations if provided in initial_state
        if "max_iterations" in initial_state:
            self.max_iterations = initial_state.pop("max_iterations")
            print(f"Max iterations set to: {self.max_iterations}")
            
        # Override max_failures if provided in initial_state
        if "max_failures" in initial_state:
            self.max_failures = initial_state.pop("max_failures")
            print(f"Max consecutive failures set to: {self.max_failures}")
        
        # Extract initial metrics if provided
        initial_metrics = initial_state.get("metrics", {})
        
        # Extract dataset description if provided
        dataset_description = initial_state.get("dataset_description", {})
        
        # Create properly typed initial state
        typed_state = PlanExecuteState(
            model_code=initial_state.get("model_code", ""),
            improved_code="",
            execution_output="",
            metrics=initial_metrics.copy() if initial_metrics else {},
            previous_metrics={},
            iteration_count=0,
            improvement_history=[],
            plan=[],
            current_step_index=0,
            past_steps=[],
            changes=[],
            dataset_description=dataset_description,
            token_usage={"prompt": 0, "completion": 0, "total": 0},
            iteration_times=[],
            start_time=self.start_time,
            iteration_start_time=None,
            consecutive_failures=0,
            last_successful_state={},
            max_failures_allowed=self.max_failures
        )
        
        print("\n🚀 Starting Plan-and-Execute Model Improvement Process")
        print(f"Dataset: {dataset_description.get('DATASET_TITLE', 'Unknown')}")
        print(f"Error handling: stopping after {self.max_failures} consecutive failures")
        
        if dataset_description.get('NUMERICAL_FEATURES') and dataset_description.get('CATEGORICAL_FEATURES'):
            print(f"Features: {len(dataset_description.get('NUMERICAL_FEATURES', [])) + len(dataset_description.get('CATEGORICAL_FEATURES', []))} total, {len(dataset_description.get('NUMERICAL_FEATURES', []))} numerical, {len(dataset_description.get('CATEGORICAL_FEATURES', []))} categorical")
        
        try:
            # Run the decision procedure
            final_output = None
            for output in self.decision_procedure.stream(typed_state):
                final_output = output
                
                # Log execution progress for each node
                for node_name, state in output.items():
                    # Log token usage periodically
                    if "token_usage" in state:
                        token_usage = state["token_usage"]
                        print(f"\nCurrent Token Usage:")
                        print(f"Prompt: {token_usage.get('prompt', 0)}")
                        print(f"Completion: {token_usage.get('completion', 0)}")
                        print(f"Total: {token_usage.get('total', 0)}")
            
            # Get the final state from the last node executed
            final_state = final_output[list(final_output.keys())[-1]]
            
            # Calculate total runtime
            end_time = time.time()
            runtime_seconds = end_time - self.start_time
            
            # Generate final report
            print("\n📊 Plan-and-Execute Improvement Process Complete")
            print(f"\nTotal runtime: {runtime_seconds:.2f} seconds")
            print(f"Execution attempts: successful={final_state.get('iteration_count', 0)}, " + 
                  f"failed={final_state.get('consecutive_failures', 0)}")
            
            print("\nFinal Metrics:")
            self._log_metrics(
                final_state.get("metrics", {}), 
                final_state.get("improvement_history", [{}])[0].get("metrics", {}) if final_state.get("improvement_history") else {}
            )
            
            # Summary of iterations
            if final_state.get("iteration_times"):
                print("\nIteration Times:")
                for iter_time in final_state["iteration_times"]:
                    print(f"  Iteration {iter_time['iteration']}: {iter_time['time']:.2f} seconds")
            
            # Create standardized YAML output
            final_state["yaml_output"] = self._export_results_to_yaml(final_state, runtime_seconds)
            
            return final_state
            
        except Exception as e:
            print(f"Error in graph execution: {str(e)}")
            import traceback
            traceback.print_exc()
            raise