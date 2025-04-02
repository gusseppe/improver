from typing import TypedDict, Dict, Any, List, Union, Optional
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

class ReactState(TypedDict):
    """State for the React-based improver agent."""
    model_code: str                         # Original model code
    improved_code: str                      # Current improved code
    execution_output: str                   # Output from executing the code
    metrics: Dict[str, Any]                 # Current metrics
    previous_metrics: Dict[str, Any]        # Metrics from previous iteration
    iteration_count: int                    # Current iteration number
    improvement_history: List[Dict[str, Any]]  # History of all improvements
    
    # React-specific state components
    thought: str                            # Current reasoning step
    action: str                             # Current action to take
    action_input: Any                       # Input for the current action
    observation: str                        # Result of the last action
    scratchpad: List[Dict[str, str]]        # History of thoughts, actions, and observations
    
    # Additional components for standardized tracking
    dataset_description: Optional[Dict[str, Any]]  # Dataset description in JSON format
    token_usage: Dict[str, int]             # Token usage tracking
    iteration_times: List[Dict[str, Any]]   # Time tracking per iteration
    start_time: float                       # Start time of the process
    iteration_start_time: Optional[float]   # Start time of current iteration


def prompt_react_reasoning() -> ChatPromptTemplate:
    """Create a prompt for the React reasoning step."""
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
            
            iteration_count: 1
            
            metrics:
                model_old_score:
                    on_old_data: 0.913
                    on_new_data: 0.717
                    
            scratchpad: []
            
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
            thought: |
                I'm looking at a RandomForestClassifier model that's trained and evaluated only on the old data distribution. The metrics show performance of 0.913 on old data but only 0.717 on new data, indicating a significant drift problem.
                
                The dataset contains both numerical features (Age, Income, Credit Score, Loan Amount, Loan Term, Interest Rate, Employment Length) and categorical features (Home Ownership, Marital Status, Dependents). This is a loan default prediction problem.
                
                For the first iteration, I should focus on the most fundamental improvements:
                1. Reading and incorporating the new data
                2. Training on both old and new data combined
                3. Evaluating on both test sets
                4. Considering a model that handles distribution shifts better than RandomForest
                
                GradientBoostingClassifier would be a good alternative since it builds trees sequentially and can adapt better to shifts in data distributions. I'll also include proper metrics tracking.
                
            action: ImproveCode
            
            action_input: |
                Switch to GradientBoostingClassifier with parameters optimized for loan default prediction data, train on combined old and new data, and evaluate on both distributions.
            """
        }
    ]
    
    for example in examples:
        example["input"] = textwrap.dedent(example["input"]).strip()
        example["output"] = textwrap.dedent(example["output"]).strip()

    system_prompt = """
    You are an expert ML engineer using the React (Reasoning, Action) approach to improve ML code. In this reasoning step, you'll analyze the current code and metrics to decide what improvement to make.
    
    Remember that you're in a multi-step process:
    1. Reason about the current code and metrics (this step)
    2. Choose an action to take
    3. Provide action input (what improvement to make)
    4. Later you'll observe the results and continue the process
    
    Available actions:
    - ImproveCode: Generate improved code based on your reasoning
    
    Focus on ONE improvement strategy per iteration:
    1. Try a different sklearn model (RandomForest, GradientBoosting, XGBoost, etc.)
    2. Modify model hyperparameters significantly
    3. Implement data preprocessing techniques
    
    Analyze:
    - Current model performance on both distributions
    - The gap between old and new distribution performance
    - Previous attempts and their results (if available)
    - What approach would address the specific issues observed
    - Dataset description and characteristics (data types, feature counts, etc.)
    
    Your response must be in this format:
    
    thought: |
      [Your detailed analysis of the current state]
      
    action: ImproveCode
    
    action_input: |
      [Brief description of the improvement approach]
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


def prompt_improve_code() -> ChatPromptTemplate:
    """Prompt to generate improved code based on reasoning."""
    examples = [
        {
            "input": """
            model_code: |
                import pandas as pd
                from sklearn.ensemble import RandomForestClassifier
                # load the old data
                dataset_folder = "datasets/financial"
                X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
                X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
                y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
                y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
                model_old = RandomForestClassifier(random_state=42)
                model_old.fit(X_train_old, y_train_old)
                # Test the model on the old test set
                old_accuracy = model_old.score(X_test_old, y_test_old)
                print(f'Model trained and evaluated on the old distribution: {old_accuracy}')
            
            thought: |
                I'm looking at a RandomForestClassifier model that's trained and evaluated only on the old data distribution. The metrics show performance of 0.913 on old data but only 0.717 on new data, indicating a significant drift problem.
                
                For the first iteration, I should focus on the most fundamental improvements:
                1. Reading and incorporating the new data
                2. Training on both old and new data combined
                3. Evaluating on both test sets
                4. Considering a model that handles distribution shifts better than RandomForest
                
                GradientBoostingClassifier would be a good alternative since it builds trees sequentially and can adapt better to shifts in data distributions. I'll also include proper metrics tracking.
            
            action_input: |
                Switch to GradientBoostingClassifier with parameters optimized for loan default prediction data, train on combined old and new data, and evaluate on both distributions.
            
            iteration_count: 1
            
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
                
                # Load old data
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
                
                # Combine training data
                X_train = pd.concat([X_train_old, X_train_new])
                y_train = pd.concat([y_train_old, y_train_new])
                
                # Create and train model with parameters optimized for loan default prediction
                model_new = GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42
                )
                model_new.fit(X_train, y_train)
                
                # Evaluate on old distribution
                new_score_old = accuracy_score(y_test_old, model_new.predict(X_test_old))
                print(f'New model trained and evaluated on old distribution: {new_score_old}')
                model_new_score['on_old_data'] = float(new_score_old)
                
                # Evaluate on new distribution
                new_score_new = accuracy_score(y_test_new, model_new.predict(X_test_new))
                print(f'New model evaluated on new distribution: {new_score_new}')
                model_new_score['on_new_data'] = float(new_score_new)
                
                # Save metrics
                with open('metrics_react.yaml', 'w') as f:
                    yaml.dump({'model_new_score': model_new_score}, f)
            
            changes_made:
              - "Switched to GradientBoostingClassifier for better distribution shift handling"
              - "Increased n_estimators to 200 for better model capacity"
              - "Added subsample=0.8 to reduce overfitting"
              - "Set min_samples_split=20 and min_samples_leaf=10 for more stable trees"
              - "Implemented combined training on old and new data"
              - "Added proper evaluation on both distributions"
            """
        }
    ]
    
    for example in examples:
        example["input"] = textwrap.dedent(example["input"]).strip()
        example["output"] = textwrap.dedent(example["output"]).strip()

    system_prompt = """
    You are an expert ML engineer implementing improvements to ML code. Your task is to generate improved code based on the given reasoning and action input.
    
    Context: You have:
    1. Current model code
    2. A reasoning analysis of issues
    3. An action input describing what improvement to make
    4. Current iteration number
    5. Current metrics
    6. Dataset description with details about features and value distributions
    
    Requirements:
    1. Save metrics to 'metrics_react.yaml'
    2. Use metrics format:
       model_new_score:
           on_new_data: [score]
           on_old_data: [score]
    4. Use sklearn-compatible models
    5. Always train on combined old and new data
    6. Evaluate on both test sets
    7. Use the dataset description to inform your implementation choices
    
    Only implement ONE improvement strategy per iteration, focusing on the strategy described in the action_input.
    
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


def prompt_analyze_results() -> ChatPromptTemplate:
    """Prompt to analyze execution results."""
    examples = [
        {
            "input": """
            execution_output: |
                New model trained and evaluated on old distribution: 0.909
                New model evaluated on new distribution: 0.762
            
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
                
                # Load old data
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
                
                # Combine training data
                X_train = pd.concat([X_train_old, X_train_new])
                y_train = pd.concat([y_train_old, y_train_new])
                
                # Create and train model with improved parameters
                model_new = GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    random_state=42
                )
                model_new.fit(X_train, y_train)
                
                # Evaluate on old distribution
                new_score_old = accuracy_score(y_test_old, model_new.predict(X_test_old))
                print(f'New model trained and evaluated on old distribution: {new_score_old}')
                model_new_score['on_old_data'] = float(new_score_old)
                
                # Evaluate on new distribution
                new_score_new = accuracy_score(y_test_new, model_new.predict(X_test_new))
                print(f'New model evaluated on new distribution: {new_score_new}')
                model_new_score['on_new_data'] = float(new_score_new)
                
                # Save metrics
                with open('metrics_react.yaml', 'w') as f:
                    yaml.dump({'model_new_score': model_new_score}, f)
            
            changes_made:
              - "Switched to GradientBoostingClassifier for better distribution shift handling"
              - "Increased n_estimators to 200 for better model capacity"
              - "Added subsample=0.8 to reduce overfitting"
              - "Implemented combined training on old and new data"
              - "Added proper evaluation on both distributions"
            
            iteration_count: 1
            
            metrics:
                model_new_score:
                    on_old_data: 0.909
                    on_new_data: 0.762
                    
            previous_metrics:
                model_old_score:
                    on_old_data: 0.913
                    on_new_data: 0.717
            """,
            "output": """
            thought: |
                The execution has completed successfully, and I can now analyze the results of switching to GradientBoostingClassifier and the other improvements:
                
                Previous metrics:
                - Old distribution: 0.913
                - New distribution: 0.717
                - Gap: 0.196 (19.6%)
                
                New metrics:
                - Old distribution: 0.909
                - New distribution: 0.762
                - Gap: 0.147 (14.7%)
                
                Changes in performance:
                - Old distribution: -0.004 (-0.4%) slight decrease
                - New distribution: +0.045 (+4.5%) significant improvement
                - Gap reduction: 4.9 percentage points
                
                The strategy was successful overall. While we lost a minimal amount of performance on the old distribution (-0.4%), we gained a substantial improvement on the new distribution (+4.5%), which was the primary goal. The gap between distributions decreased from 19.6% to 14.7%.
                
                For the next iteration, I should try to further improve the model's ability to handle the new distribution while minimizing any performance loss on the old distribution. I could explore:
                
                1. Fine-tuning the GradientBoostingClassifier parameters
                2. Adding preprocessing steps like feature scaling or selection
                3. Implementing a different model like XGBoost
                
                The most promising approach would be to focus on hyperparameter tuning since we've already seen good results with GradientBoostingClassifier.
                
            observation: |
                The model changes resulted in a significant improvement on the new distribution (+4.5%) with only a minimal decrease on the old distribution (-0.4%). The performance gap between distributions decreased from 19.6% to 14.7%.
            """
        }
    ]
    
    for example in examples:
        example["input"] = textwrap.dedent(example["input"]).strip()
        example["output"] = textwrap.dedent(example["output"]).strip()

    system_prompt = """
    You are an expert ML engineer analyzing the results of code execution. Your task is to thoroughly analyze the results of the implemented changes.
    
    Context: You have:
    1. Execution output showing model performance
    2. The improved code that was executed
    3. List of changes made in this iteration
    4. Current iteration number
    5. Current and previous metrics
    
    Analysis requirement:
    1. Compare current metrics with previous metrics
    2. Calculate exact improvements or regressions
    3. Analyze the gap between old and new distribution performance
    4. Evaluate if the changes were successful
    5. Identify what worked and what didn't
    
    Response Format: Format your response as YAML with:
    
    thought: |
      [Your detailed analysis of the results]
      
    observation: |
      [Concise summary of key findings]
    
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


class ReactImprover:
    def __init__(self, llm, max_iterations=3, max_failures=3, debug=False):
        """Initialize the React-based improvement graph.
        
        Args:
            llm: The language model to use for generation
            max_iterations: Maximum number of improvement iterations to run
            max_failures: Maximum number of consecutive execution failures allowed
            debug: Whether to run in debug mode
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.max_failures = max_failures
        self.graph = StateGraph(ReactState)
        self.debug = debug
        self.build_plan()
        self.decision_procedure = self.graph.compile(debug=debug)
        self.start_time = None
        self.iteration_start_time = None
        self.token_counts = {"prompt": 0, "completion": 0, "total": 0}
    
    def build_plan(self):
        """Build the React graph structure."""
        # Add nodes for each step in the React process
        self.graph.add_node("reasoning", self.reasoning_step)
        self.graph.add_node("take_action", self.action_step)  # Renamed from "action" to "take_action"
        self.graph.add_node("execute", self.execute_step)
        self.graph.add_node("observe", self.observe_step)
        
        # Set entry point
        self.graph.set_entry_point("reasoning")
        
        # Add edges
        self.graph.add_edge("reasoning", "take_action")  # Updated edge
        self.graph.add_edge("take_action", "execute")    # Updated edge
        self.graph.add_edge("execute", "observe")
        
        # Add conditional edge
        self.graph.add_conditional_edges(
            "observe",
            self.should_continue,
            {
                "continue": "reasoning",
                "end": END
            }
        )
    
    def _record_token_usage(self, state: ReactState, prompt_tokens: int, completion_tokens: int):
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
    
    def reasoning_step(self, state: ReactState) -> ReactState:
        """Reason about the current state and decide on an action."""
        print("\n🧠 REASONING STEP")
        
        # Start timing this iteration if it's a new iteration
        if state.get("iteration_count", 0) == len(state.get("improvement_history", [])):
            self.iteration_start_time = time.time()
            state["iteration_start_time"] = self.iteration_start_time
        
        # Prepare input for reasoning prompt
        input_data = {
            "model_code": state.get("model_code", ""),
            "iteration_count": state.get("iteration_count", 0),
            "metrics": state.get("metrics", {}),
            "scratchpad": state.get("scratchpad", []),
            "dataset_description": state.get("dataset_description", {})
        }
        
        if self.debug:
            print("\nInput to reasoning step:")
            print(yaml.dump(input_data))
        
        # Estimate prompt tokens (rough approximation)
        prompt_text = yaml.dump(input_data)
        prompt_tokens = len(prompt_text) // 4
        
        # Generate reasoning and action
        prompt = prompt_react_reasoning()
        chain = prompt | self.llm
        output = chain.invoke({"input": yaml.dump(input_data)}).content
        
        # Estimate completion tokens 
        completion_tokens = len(output) // 4
        
        # Record token usage in state
        state = self._record_token_usage(state, prompt_tokens, completion_tokens)
        
        # Parse YAML output
        try:
            result = yaml.safe_load(output)
            state["thought"] = result.get("thought", "")
            state["action"] = result.get("action", "")
            state["action_input"] = result.get("action_input", "")
            
            # Add to scratchpad
            if "scratchpad" not in state:
                state["scratchpad"] = []
            
            state["scratchpad"].append({
                "thought": state["thought"],
                "action": state["action"],
                "action_input": state["action_input"]
            })
            
            print(f"\nThought: {state['thought'].strip()}")
            print(f"\nAction: {state['action']}")
            print(f"\nAction Input: {state['action_input'].strip()}")
            
        except yaml.YAMLError as e:
            print(f"Error parsing reasoning output: {e}")
            # Fallback handling
            state["thought"] = "Error parsing reasoning output"
            state["action"] = "ImproveCode"
            state["action_input"] = "Fix errors in the current implementation"
            
            if "scratchpad" not in state:
                state["scratchpad"] = []
            
            state["scratchpad"].append({
                "thought": state["thought"],
                "action": state["action"],
                "action_input": state["action_input"]
            })
        
        return state
    
    def action_step(self, state: ReactState) -> ReactState:
        """Take action based on reasoning."""
        print("\n⚙️ ACTION STEP")
        
        # Currently only supporting ImproveCode action
        if state["action"] == "ImproveCode":
            # Prepare input for code improvement prompt
            input_data = {
                "model_code": state.get("model_code", ""),
                "thought": state.get("thought", ""),
                "action_input": state.get("action_input", ""),
                "iteration_count": state.get("iteration_count", 0),
                "metrics": state.get("metrics", {}),
                "dataset_description": state.get("dataset_description", {})
            }
            
            # Estimate prompt tokens
            prompt_text = yaml.dump(input_data)
            prompt_tokens = len(prompt_text) // 4
            
            # Generate improved code
            prompt = prompt_improve_code()
            chain = prompt | self.llm
            output = chain.invoke({"input": yaml.dump(input_data)}).content
            
            # Estimate completion tokens 
            completion_tokens = len(output) // 4
            
            # Record token usage in state
            state = self._record_token_usage(state, prompt_tokens, completion_tokens)
            
            # Parse YAML output
            try:
                result = yaml.safe_load(output)
                state["improved_code"] = result.get("improved_code", "")
                state["changes"] = result.get("changes_made", [])
                
                print("\nGenerated improved code for implementation")
                print("\nChanges made:")
                for change in state["changes"]:
                    print(f"- {change}")
                
            except yaml.YAMLError as e:
                print(f"Error parsing code improvement output: {e}")
                state["improved_code"] = output
                state["changes"] = []
        else:
            print(f"Unsupported action: {state['action']}")
            state["improved_code"] = state.get("model_code", "")
            state["changes"] = []
        
        return state
    
    def execute_step(self, state: ReactState) -> ReactState:
        """Execute the improved code with error handling for repeated failures."""
        print("\n🔄 EXECUTION STEP")
        
        code = state["improved_code"]
        wrapped_code = f"```python\n{code}\n```"
        
        # Set up executor
        executor = LocalCommandLineCodeExecutor(timeout=30)
        code_executor_agent = ConversableAgent(
            "executor",
            llm_config=False,
            code_execution_config={"executor": executor}
        )
        
        # Execute code
        print("\nExecuting generated code...")
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
        
        # If execution succeeded, reset the failure count and save this as the last successful state
        state["consecutive_failures"] = 0
        
        # Parse metrics
        try:
            with open('metrics_react.yaml', 'r') as f:
                metrics = yaml.safe_load(f)
                if not metrics or not isinstance(metrics, dict):
                    raise ValueError("Invalid metrics format")
                
                # Store previous metrics before updating
                state["previous_metrics"] = state.get("metrics", {})
                
                # Ensure initial_metrics is preserved if it exists
                if "model_old_score" in state.get("metrics", {}):
                    metrics["model_old_score"] = state["metrics"]["model_old_score"]
                    
                state["metrics"] = metrics
                
                # Store this as the last successful state (create a deep copy)
                state["last_successful_state"] = json.loads(json.dumps({
                    "improved_code": state["improved_code"],
                    "execution_output": state["execution_output"],
                    "metrics": state["metrics"],
                    "changes": state.get("changes", [])
                }))
                
        except Exception as e:
            print(f"\nError reading metrics: {str(e)}")
            if state.get("previous_metrics"):
                state["metrics"] = state["previous_metrics"]
        
        # Increment iteration count
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        
        # Add to scratchpad
        if "scratchpad" in state:
            state["scratchpad"][-1]["execution_output"] = state["execution_output"]
        
        # Calculate iteration time
        if "iteration_start_time" in state:
            iteration_end_time = time.time()
            iteration_time = iteration_end_time - state["iteration_start_time"]
            
            # Track iteration times
            if "iteration_times" not in state:
                state["iteration_times"] = []
                
            state["iteration_times"].append({
                "iteration": state.get("iteration_count", 0),
                "time": iteration_time
            })
            
            print(f"\nIteration {state.get('iteration_count', 0)} time: {iteration_time:.2f} seconds")
        
        # Update improvement history
        if not self._has_execution_errors(execution_output):
            if "improvement_history" not in state:
                state["improvement_history"] = []
            
            # Get current token usage for this iteration
            token_usage = state.get("token_usage", {})
            
            history_entry = {
                "iteration": state.get("iteration_count", 0),
                "metrics": state.get("metrics", {}),
                "changes": state.get("changes", []),
                "code": state.get("improved_code", ""),
                "execution_time": state["iteration_times"][-1]["time"] if state.get("iteration_times") else 0,
                "token_usage": {
                    "prompt": token_usage.get("prompt", 0),
                    "completion": token_usage.get("completion", 0),
                    "total": token_usage.get("total", 0)
                }
            }
            
            state["improvement_history"].append(history_entry)
        
        return state
    
    def observe_step(self, state: ReactState) -> ReactState:
        """Observe and analyze execution results."""
        print("\n👁️ OBSERVATION STEP")
        
        # Prepare input for analysis prompt
        input_data = {
            "execution_output": state.get("execution_output", ""),
            "improved_code": state.get("improved_code", ""),
            "changes_made": state.get("changes", []),
            "iteration_count": state.get("iteration_count", 0),
            "metrics": state.get("metrics", {}),
            "previous_metrics": state.get("previous_metrics", {})
        }
        
        # Estimate prompt tokens
        prompt_text = yaml.dump(input_data)
        prompt_tokens = len(prompt_text) // 4
        
        # Generate analysis
        prompt = prompt_analyze_results()
        chain = prompt | self.llm
        output = chain.invoke({"input": yaml.dump(input_data)}).content
        
        # Estimate completion tokens 
        completion_tokens = len(output) // 4
        
        # Record token usage in state
        state = self._record_token_usage(state, prompt_tokens, completion_tokens)
        
        # Parse YAML output
        try:
            result = yaml.safe_load(output)
            state["thought"] = result.get("thought", "")
            state["observation"] = result.get("observation", "")
            
            # Add to scratchpad
            if "scratchpad" in state and state["scratchpad"]:
                state["scratchpad"][-1]["observation"] = state["observation"]
            
            print(f"\nObservation: {state['observation'].strip()}")
            
        except yaml.YAMLError as e:
            print(f"Error parsing observation output: {e}")
            # Fallback handling
            state["observation"] = "Error analyzing results"
            
            if "scratchpad" in state and state["scratchpad"]:
                state["scratchpad"][-1]["observation"] = state["observation"]
        
        return state
    
    def should_continue(self, state: ReactState) -> str:
        """Determine if improvement process should continue."""
        # Check if we've hit the max consecutive failures
        if state.get("consecutive_failures", 0) >= self.max_failures:
            print(f"\nReached maximum consecutive failures ({self.max_failures}). Ending process.")
            return "end"
            
        # End after max_iterations
        if state.get("iteration_count", 0) >= self.max_iterations:
            print(f"\nReached maximum iterations ({state['iteration_count']}/{self.max_iterations})")
            return "end"
        
        # Get metrics
        current = state.get("metrics", {}).get('model_new_score', {})
        previous = state.get("previous_metrics", {}).get('model_new_score', {})
        
        # First iteration always continues
        if not previous:
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
        if not metrics or not metrics.get('model_new_score'):
            print("No metrics available")
            return
        
        current = metrics.get('model_new_score', {})
        
        print("\nCurrent Performance:")
        print(f"Old Distribution: {current.get('on_old_data', 0):.4f}")
        print(f"New Distribution: {current.get('on_new_data', 0):.4f}")
        
        if previous_metrics and previous_metrics.get('model_new_score'):
            previous = previous_metrics.get('model_new_score', {})
            
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
    
    def _export_results_to_yaml(self, state: ReactState, runtime_seconds: float) -> Dict:
        """Export the results to a standardized YAML-compatible dictionary format."""
        
        # Extract initial metrics from the provided initial metrics
        initial_metrics = {}
        if state.get("metrics") and "model_old_score" in state.get("metrics", {}):
            old_metrics = state["metrics"]["model_old_score"]
            initial_metrics = {
                "old_distribution": old_metrics.get("on_old_data", 0),
                "new_distribution": old_metrics.get("on_new_data", 0)
            }
        
        # Extract final metrics
        final_metrics = {}
        if state.get("metrics") and "model_new_score" in state.get("metrics", {}):
            metrics = state["metrics"]["model_new_score"]
            final_metrics = {
                "old_distribution": metrics.get("on_old_data", 0),
                "new_distribution": metrics.get("on_new_data", 0)
            }
        
        # Build improvement path
        improvement_path = []
        max_iterations = min(state.get("iteration_count", 0), len(state.get("improvement_history", [])))
        for i in range(max_iterations):
            # Extract metrics correctly based on the entry structure
            entry = state.get("improvement_history", [])[i]
            metrics = entry.get("metrics", {}).get("model_new_score", {})
            
            # Add reflection based on scratchpad entries
            reflection = ""
            if state.get("scratchpad") and i < len(state.get("scratchpad")):
                scratchpad_entry = state["scratchpad"][i]
                if "observation" in scratchpad_entry:
                    reflection = scratchpad_entry["observation"]
                elif "thought" in scratchpad_entry:
                    # Extract first 200 chars as a fallback
                    thought_text = scratchpad_entry["thought"]
                    reflection = thought_text[:min(200, len(thought_text))] + "..."
            
            path_entry = {
                "iteration": i + 1,
                "code": entry.get("code", ""),
                "metrics": {
                    "old_distribution": metrics.get("on_old_data", 0),
                    "new_distribution": metrics.get("on_new_data", 0)
                },
                "changes": entry.get("changes", []),
                "reflection": reflection,
                "execution_time": entry.get("execution_time", 0)
            }
            improvement_path.append(path_entry)
        
        # Get token usage from state or use class counters as fallback
        token_usage = state.get("token_usage", {
            "prompt": self.token_counts["prompt"],
            "completion": self.token_counts["completion"],
            "total": self.token_counts["total"]
        })
        
        # Create standardized output
        result = {
            "agent_name": "react",
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
        
        return result
    
    def _estimate_token_usage(self, state: ReactState) -> int:
        """
        Estimate token usage based on the text length in the state.
        This is a rough approximation; for more accurate counting you would
        need to use a tokenizer specific to your model.
        
        Args:
            state: The final state
            
        Returns:
            Estimated token count
        """
        # If token_usage is tracked in state, use that
        if "token_usage" in state and "total" in state["token_usage"]:
            return state["token_usage"]["total"]
            
        # Otherwise estimate based on character counts
        # Simple approximation: ~1 token per 4 characters for English text
        char_count = 0
        
        # Count characters in model code
        char_count += len(state.get("model_code", ""))
        
        # Count characters in improved code
        char_count += len(state.get("improved_code", ""))
        
        # Count characters in scratchpad (thoughts, observations)
        for entry in state.get("scratchpad", []):
            char_count += len(entry.get("thought", ""))
            char_count += len(entry.get("action_input", ""))
            char_count += len(entry.get("observation", ""))
        
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
        """Run the React improvement process with improved error handling.
        
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
        typed_state = ReactState(
            model_code=initial_state.get("model_code", ""),
            improved_code="",
            execution_output="",
            metrics=initial_metrics.copy() if initial_metrics else {},
            previous_metrics={},
            iteration_count=0,
            improvement_history=[],
            thought="",
            action="",
            action_input="",
            observation="",
            scratchpad=[],
            dataset_description=dataset_description,
            token_usage={"prompt": 0, "completion": 0, "total": 0},
            iteration_times=[],
            start_time=self.start_time,
            iteration_start_time=None,
            # Error handling state variables
            consecutive_failures=0,
            last_successful_state={},
            max_failures_allowed=self.max_failures
        )
        
        print("\n🚀 Starting React-based Model Improvement Process")
        print(f"Dataset: {dataset_description.get('DATASET_TITLE', 'Unknown')}")
        print(f"Error handling: stopping after {self.max_failures} consecutive failures")
        
        if dataset_description.get('NUMERICAL_FEATURES') and dataset_description.get('CATEGORICAL_FEATURES'):
            print(f"Features: {len(dataset_description.get('NUMERICAL_FEATURES', [])) + len(dataset_description.get('CATEGORICAL_FEATURES', []))} total, {len(dataset_description.get('NUMERICAL_FEATURES', []))} numerical, {len(dataset_description.get('CATEGORICAL_FEATURES', []))} categorical")
        
        try:
            # Run the decision procedure
            final_output = None
            for output in self.decision_procedure.stream(typed_state):
                # Store the latest output
                final_output = output
                
                # Log current node
                for node_name, state in output.items():
                    # Log token usage after each step
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
            print("\n📊 React Model Improvement Process Complete")
            print(f"\nTotal runtime: {runtime_seconds:.2f} seconds")
            print(f"Execution attempts: successful={len(final_state.get('improvement_history', []))}, " + 
                  f"failed={final_state.get('consecutive_failures', 0)}")
            print("\nFinal Metrics:")
            self._log_metrics(
                final_state.get("metrics", {}), 
                final_state.get("improvement_history", [{}])[0].get("metrics", {})
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