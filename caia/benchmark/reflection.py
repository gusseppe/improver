from typing import TypedDict, Dict, Any, List, Literal, Annotated, Optional
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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from operator import add
from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor

class ReflectionState(TypedDict):
    """State for the Reflection-based ML code improvement agent."""
    model_code: str                         # Original model code
    improved_code: str                      # Current improved code
    execution_output: str                   # Output from executing the code
    metrics: Dict[str, Any]                 # Current metrics
    previous_metrics: Dict[str, Any]        # Metrics from previous iteration
    iteration_count: int                    # Current iteration number
    improvement_history: List[Dict[str, Any]]  # History of all improvements
    changes: List[str]                      # List of changes made
    
    # Reflection-specific state components
    reflections: List[str]                  # History of reflections
    current_reflection: str                 # Current reflection
    messages: List[Dict[str, Any]]          # Message history for the conversation
    
    # Additional components for standardized tracking
    dataset_description: Optional[Dict[str, Any]]  # Dataset description in JSON format
    token_usage: Dict[str, int]             # Token usage tracking
    iteration_times: List[Dict[str, Any]]   # Time tracking per iteration
    start_time: float                       # Start time of the process
    iteration_start_time: Optional[float]   # Start time of current iteration
    
    # Error handling components
    consecutive_failures: int               # Count of consecutive execution failures
    last_successful_state: Dict             # Last state with successful execution
    max_failures_allowed: int               # Maximum allowed consecutive failures


def prompt_generate_improvement() -> ChatPromptTemplate:
    """Create prompt for generating code improvements."""
    examples = [
        {
            "input": """
            You are improving an ML model to handle both old and new data distributions.
            
            Current code:
            ```python
            import pandas as pd
            from sklearn.ensemble import RandomForestClassifier
            
            # load the old data
            dataset_folder = "datasets/financial" # it can be any folder like datasets/eligibility
            X_train_old = pd.read_csv(f"{dataset_folder}/X_train_old.csv")
            X_test_old = pd.read_csv(f"{dataset_folder}/X_test_old.csv")
            y_train_old = pd.read_csv(f"{dataset_folder}/y_train_old.csv").squeeze("columns")
            y_test_old = pd.read_csv(f"{dataset_folder}/y_test_old.csv").squeeze("columns")
            
            model_old = RandomForestClassifier(random_state=42)
            model_old.fit(X_train_old, y_train_old)
            
            # Test the model on the old test set
            old_accuracy = model_old.score(X_test_old, y_test_old)
            print(f'Model trained and evaluated on the old distribution: {old_accuracy}')
            ```
            
            Current metrics:
            - Old distribution: 0.91
            - New distribution: 0.72
            
            Dataset description:
            - Dataset: Loan Default Prediction Data
            - Numerical features: Age, Income, Credit Score, Loan Amount, Loan Term, Interest Rate, Employment Length
            - Categorical features: Home Ownership, Marital Status, Dependents
            
            Please improve this code.
            """,
            "output": """
            I'll improve the model to handle both distributions by:
            1. Loading the new data
            2. Combining old and new data for training
            3. Using a GradientBoostingClassifier which handles distribution shifts better
            4. Adding preprocessing with StandardScaler for numerical features
            5. Properly handling categorical features with OneHotEncoder
            6. Saving metrics for both distributions
            
            Here's the improved code:
            
            ```python
            import yaml
            import pandas as pd
            import numpy as np
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
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
            
            # Define feature types
            numerical_features = ['Age', 'Income', 'Credit Score', 'Loan Amount', 'Loan Term', 'Interest Rate', 'Employment Length']
            categorical_features = ['Home Ownership', 'Marital Status', 'Dependents']
            
            # Create preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ],
                remainder='passthrough'
            )
            
            # Create pipeline with preprocessing and model
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    subsample=0.8,
                    random_state=42
                ))
            ])
            
            # Combine old and new training data
            X_train_combined = pd.concat([X_train_old, X_train_new])
            y_train_combined = pd.concat([y_train_old, y_train_new])
            
            # Ensure all column names are strings to avoid type errors
            X_train_combined.columns = X_train_combined.columns.astype(str)
            X_test_old.columns = X_test_old.columns.astype(str)
            X_test_new.columns = X_test_new.columns.astype(str)
            
            # Train the pipeline
            pipeline.fit(X_train_combined, y_train_combined)
            
            # Evaluate on old distribution
            old_predictions = pipeline.predict(X_test_old)
            old_score = accuracy_score(y_test_old, old_predictions)
            print(f'New model evaluated on old distribution: {old_score}')
            model_new_score['on_old_data'] = float(old_score)
            
            # Evaluate on new distribution
            new_predictions = pipeline.predict(X_test_new)
            new_score = accuracy_score(y_test_new, new_predictions)
            print(f'New model evaluated on new distribution: {new_score}')
            model_new_score['on_new_data'] = float(new_score)
            
            # Save metrics
            with open('metrics_reflection.yaml', 'w') as f:
                yaml.dump({'model_new_score': model_new_score}, f)
            ```
            
            The changes I've made:
            - Added loading of new training and test data
            - Added proper preprocessing with StandardScaler for numerical features
            - Added OneHotEncoder for categorical features
            - Used ColumnTransformer to handle mixed feature types
            - Used GradientBoostingClassifier which handles distribution shifts better
            - Added Pipeline to combine preprocessing and model
            - Converted column names to strings to avoid type errors
            - Combined old and new data for training
            - Evaluated on both old and new distributions
            - Saved metrics in the required format
            """
        }
    ]
    
    system_template = """
    You are an expert ML engineer tasked with improving an ML model to better handle data distribution shifts.
    
    Your goal is to:
    1. Improve performance on the new data distribution
    2. Maintain good performance on the old data distribution
    3. Reduce the gap between performances on both distributions
    
    Focus on these techniques:
    - Data preprocessing (scaling, normalization)
    - Model selection (try different classifiers)
    - Hyperparameter tuning
    - Training on combined datasets
    - Proper handling of numerical and categorical features
    
    DATASET INFORMATION:
    I'll provide details about the dataset structure in my prompt, including feature types and distributions.
    Use this information to guide your preprocessing and model selection decisions.
    
    IMPORTANT REQUIREMENTS:
    - Save metrics to 'metrics_reflection.yaml'
    - Use metrics format:
      model_new_score:
          on_new_data: [score]
          on_old_data: [score]
    - Use sklearn-compatible models
    - Always train on combined old and new data
    - Evaluate on both test sets
    - Convert column names to strings to avoid type errors
    
    Respond with:
    1. Your reasoning for the proposed improvements
    2. The complete improved code (implementation)
    3. A list of specific changes you've made
    
    IMPORTANT CONSTRAINTS:
    - Avoid using GridSearchCV or complex search algorithms that may time out
    - Ensure your code is efficient and will run in under 30 seconds
    - Never truncate or skip parts of your implementation
    - Always handle mixed feature types correctly
    """
    
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        FewShotChatMessagePromptTemplate(
            example_prompt=ChatPromptTemplate.from_messages([
                ("human", "{input}"),
                ("ai", "{output}")
            ]),
            examples=examples
        ),
        MessagesPlaceholder(variable_name="messages")
    ])


def prompt_reflect_on_code() -> ChatPromptTemplate:
    """Create prompt for reflecting on the generated code."""
    system_template = """
    You are an expert ML code reviewer specifically focused on models that handle distribution shifts.
    
    Your task is to critically analyze the proposed code improvements and suggest refinements.
    
    Focus your analysis on:
    1. Correctness - Will the code run without errors?
    2. Effectiveness - Will the changes improve performance on both distributions?
    3. Efficiency - Are there any optimizations that could be made?
    4. Robustness - Will the model generalize well to both distributions?
    
    CRITICAL EVALUATION AREAS:
    - Data preprocessing approach
    - Model selection and parameters
    - Training methodology
    - Evaluation methodology
    - Metrics handling
    - Feature handling (both numerical and categorical)
    
    Your reflection should:
    - Identify specific strengths in the current approach
    - Highlight any potential issues or risks
    - Check for common errors (like feature type handling, column type issues)
    - Suggest 2-3 concrete improvements with justification
    - Be specific about what to change and why
    
    DO NOT rewrite the entire code - focus on targeted improvements.
    
    If the approach is already strong, acknowledge this but still suggest refinements.
    """
    
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="messages")
    ])


class ReflectionGraph:
    def __init__(self, llm, max_iterations=3, max_failures=3, debug=False):
        """Initialize the Reflection-based improvement graph.
        
        Args:
            llm: The language model to use for generation
            max_iterations: Maximum number of improvement iterations to run
            max_failures: Maximum number of consecutive execution failures allowed
            debug: Whether to run in debug mode
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.max_failures = max_failures
        self.debug = debug
        self.graph = self.build_graph()
        self.start_time = None
        self.iteration_start_time = None
        self.token_counts = {"prompt": 0, "completion": 0, "total": 0}
        
    def build_graph(self) -> StateGraph:
        """Build the reflection graph structure."""
        workflow = StateGraph(ReflectionState)
        
        # Add nodes for each step
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("reflect", self.reflect_node)
        workflow.add_node("execute", self.execute_node)
        
        # Set entry point
        workflow.set_entry_point("generate")
        
        # Define the edges
        workflow.add_conditional_edges(
            "generate",
            self.should_reflect_or_execute,
            {
                "reflect": "reflect",
                "execute": "execute"
            }
        )
        
        workflow.add_edge("reflect", "generate")
        
        workflow.add_conditional_edges(
            "execute",
            self.should_continue,
            {
                "continue": "generate",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _record_token_usage(self, state: ReflectionState, prompt_tokens: int, completion_tokens: int):
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
    
    def generate_node(self, state: ReflectionState) -> ReflectionState:
        """Generate improved code based on current state."""
        print("\n🔍 GENERATING IMPROVED CODE")
        
        # Start timing this iteration if it's a new iteration
        if state.get("iteration_count", 0) == len(state.get("improvement_history", [])):
            self.iteration_start_time = time.time()
            state["iteration_start_time"] = self.iteration_start_time
        
        # Initialize message history if needed
        if "messages" not in state or not state["messages"]:
            # First run - create initial message
            current_code = state["model_code"]
            metrics = state["metrics"]
            dataset_desc = state.get("dataset_description", {})
            
            metrics_text = ""
            if metrics and "model_old_score" in metrics:
                metrics_text = f"""
                Current metrics:
                - Old distribution: {metrics['model_old_score'].get('on_old_data', 0):.4f}
                - New distribution: {metrics['model_old_score'].get('on_new_data', 0):.4f}
                """
            
            # Include dataset description if available
            dataset_text = ""
            if dataset_desc:
                dataset_text = f"""
                Dataset description:
                - Dataset: {dataset_desc.get('DATASET_TITLE', 'Unknown')}
                - Numerical features: {', '.join(dataset_desc.get('NUMERICAL_FEATURES', []))}
                - Categorical features: {', '.join(dataset_desc.get('CATEGORICAL_FEATURES', []))}
                """
                
            initial_message = f"""
            Current code:
            ```python
            {current_code}
            ```
            
            {metrics_text}
            
            {dataset_text}
            
            Please improve this code to handle both distributions better.
            """
            
            state["messages"] = [
                {"role": "human", "content": initial_message.strip()}
            ]
            state["reflections"] = []
        
        # Get generation prompt
        prompt = prompt_generate_improvement()
        
        # Prepare messages in the proper format for the chain
        messages = []
        for msg in state["messages"]:
            if msg["role"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        # Estimate prompt tokens (rough approximation)
        prompt_text = "\n".join([msg["content"] for msg in state["messages"]])
        prompt_tokens = len(prompt_text) // 4
        
        # Generate improvement
        chain = prompt | self.llm  # Chain the prompt with the LLM
        response = chain.invoke({"messages": messages})
        
        # Get the content safely
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # Estimate completion tokens 
        completion_tokens = len(response_content) // 4
        
        # Record token usage in state
        state = self._record_token_usage(state, prompt_tokens, completion_tokens)
        
        # Extract the code from the response
        code = self._extract_code(response_content)
        changes = self._extract_changes(response_content)
        
        # Update state
        state["improved_code"] = code
        state["changes"] = changes
        
        # Update message history
        state["messages"].append({
            "role": "ai",
            "content": response_content
        })
        
        print("\nProposed changes:")
        for change in changes:
            print(f"- {change}")
            
        # Log token usage
        token_usage = state["token_usage"]
        print(f"\nCurrent Token Usage:")
        print(f"Prompt: {token_usage.get('prompt', 0)}")
        print(f"Completion: {token_usage.get('completion', 0)}")
        print(f"Total: {token_usage.get('total', 0)}")
        
        return state
    
    def reflect_node(self, state: ReflectionState) -> ReflectionState:
        """Reflect on the proposed improvements."""
        print("\n🤔 REFLECTING ON PROPOSED IMPROVEMENTS")
        
        # Prepare messages for reflection
        reflection_messages = []
        
        # Add the last exchange
        for msg in state["messages"][-2:]:  # Just the last human-AI exchange
            if msg["role"] == "human":
                reflection_messages.append(HumanMessage(content=msg["content"]))
            else:
                reflection_messages.append(AIMessage(content=msg["content"]))
        
        # Get reflection prompt
        prompt = prompt_reflect_on_code()
        
        # Estimate prompt tokens
        prompt_text = "\n".join([msg.content for msg in reflection_messages])
        prompt_tokens = len(prompt_text) // 4
        
        # Generate reflection
        chain = prompt | self.llm  # Chain the prompt with the LLM
        
        try:
            response = chain.invoke({"messages": reflection_messages})
            
            # Get the content safely
            if hasattr(response, 'content'):
                reflection = response.content
            elif isinstance(response, str):
                reflection = response
            else:
                reflection = str(response)
                
            # Estimate completion tokens 
            completion_tokens = len(reflection) // 4
            
            # Record token usage in state
            state = self._record_token_usage(state, prompt_tokens, completion_tokens)
                
        except Exception as e:
            print(f"Error generating reflection: {e}")
            reflection = "I couldn't generate a detailed reflection. Let's try a different approach."
            
            # Still record token usage for the prompt
            state = self._record_token_usage(state, prompt_tokens, 0)
        
        # Store reflection
        state["current_reflection"] = reflection
        state["reflections"] = state.get("reflections", []) + [reflection]
        
        # Add reflection to message history
        state["messages"].append({
            "role": "human",
            "content": f"""
            Here are my thoughts on your proposed solution:
            
            {reflection}
            
            Based on this feedback, please revise your solution.
            """
        })
        
        print(f"\nReflection:\n{reflection}")
        
        # Log token usage
        token_usage = state["token_usage"]
        print(f"\nCurrent Token Usage:")
        print(f"Prompt: {token_usage.get('prompt', 0)}")
        print(f"Completion: {token_usage.get('completion', 0)}")
        print(f"Total: {token_usage.get('total', 0)}")
        
        return state
    
    def execute_node(self, state: ReflectionState) -> ReflectionState:
        """Execute the improved code and evaluate results with error handling."""
        print("\n⚙️ EXECUTING IMPROVED CODE")
        
        code = state["improved_code"]
        
        if not code:
            return state
            
        wrapped_code = f"```python\n{code}\n```"
        
        # Set up executor
        executor = LocalCommandLineCodeExecutor(timeout=60)  # Extended timeout
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
                    
                    # Calculate iteration time even for failed attempts
                    if "iteration_start_time" in state:
                        iteration_end_time = time.time()
                        iteration_time = iteration_end_time - state["iteration_start_time"]
                        
                        # Track iteration times
                        if "iteration_times" not in state:
                            state["iteration_times"] = []
                            
                        state["iteration_times"].append({
                            "iteration": state.get("iteration_count", 0) + 1,
                            "time": iteration_time,
                            "status": "failed"
                        })
                        
                        print(f"\nIteration {state.get('iteration_count', 0) + 1} time: {iteration_time:.2f} seconds (failed)")
                else:
                    # Add execution error feedback to messages
                    state["messages"].append({
                        "role": "human",
                        "content": f"""
                        I tried to run your code but encountered the following errors:
                        
                        {execution_output}
                        
                        Please fix these issues and provide an updated solution.
                        """
                    })
                    
                    # Don't update metrics if execution failed
                    if state.get("previous_metrics"):
                        state["metrics"] = state["previous_metrics"]
                
                # Don't increment iteration count for failed executions
                return state
        except Exception as e:
            print(f"\nException during code execution: {str(e)}")
            state["execution_output"] = f"Execution error: {str(e)}"
            
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
                    
                # Calculate iteration time even for failed attempts
                if "iteration_start_time" in state:
                    iteration_end_time = time.time()
                    iteration_time = iteration_end_time - state["iteration_start_time"]
                    
                    # Track iteration times
                    if "iteration_times" not in state:
                        state["iteration_times"] = []
                        
                    state["iteration_times"].append({
                        "iteration": state.get("iteration_count", 0) + 1,
                        "time": iteration_time,
                        "status": "failed"
                    })
                    
                    print(f"\nIteration {state.get('iteration_count', 0) + 1} time: {iteration_time:.2f} seconds (failed)")
            else:
                # Add execution error feedback to messages
                state["messages"].append({
                    "role": "human",
                    "content": f"""
                    I tried to run your code but encountered an exception:
                    
                    {str(e)}
                    
                    Please fix these issues and provide an updated solution.
                    """
                })
                
            if state.get("previous_metrics"):
                state["metrics"] = state["previous_metrics"]
            return state
        
        # If execution succeeded, reset the failure count and save as last successful state
        state["consecutive_failures"] = 0
        
        # Parse metrics
        try:
            with open('metrics_reflection.yaml', 'r') as f:
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
                    "changes": state.get("changes", []),
                    "current_reflection": state.get("current_reflection", "")
                }))
                
            # Calculate iteration time
            if "iteration_start_time" in state:
                iteration_end_time = time.time()
                iteration_time = iteration_end_time - state["iteration_start_time"]
                
                # Track iteration times
                if "iteration_times" not in state:
                    state["iteration_times"] = []
                    
                state["iteration_times"].append({
                    "iteration": state.get("iteration_count", 0) + 1,
                    "time": iteration_time,
                    "status": "success"
                })
                
                print(f"\nIteration {state.get('iteration_count', 0) + 1} time: {iteration_time:.2f} seconds (success)")
            
            # Update improvement history
            execution_metrics = {}
            if "model_new_score" in state["metrics"]:
                execution_metrics = state["metrics"]["model_new_score"]
            elif "model_old_score" in state["metrics"] and state["metrics"]["model_old_score"] != state.get("previous_metrics", {}).get("model_old_score", {}):
                # If only model_old_score was updated, use that
                execution_metrics = state["metrics"]["model_old_score"]
            
            # If we have metrics from execution, add to improvement history
            if execution_metrics:
                # Get current token usage for this iteration
                token_usage = state.get("token_usage", {})
                
                history_entry = {
                    "iteration": state.get("iteration_count", 0) + 1,
                    "metrics": {"model_new_score": execution_metrics},
                    "changes": state.get("changes", []),
                    "code": state.get("improved_code", ""),
                    "reflection": state.get("current_reflection", ""),
                    "execution_time": iteration_time if "iteration_start_time" in state else 0,
                    "token_usage": {
                        "prompt": token_usage.get("prompt", 0),
                        "completion": token_usage.get("completion", 0),
                        "total": token_usage.get("total", 0)
                    }
                }
                
                if "improvement_history" not in state:
                    state["improvement_history"] = []
                    
                state["improvement_history"].append(history_entry)
                print(f"\nAdded entry to improvement history with metrics: {execution_metrics}")
                
                # Add execution success feedback to messages
                state["messages"].append({
                    "role": "human",
                    "content": f"""
                    Your code executed successfully with the following results:
                    
                    - Old distribution performance: {execution_metrics.get('on_old_data', 0):.4f}
                    - New distribution performance: {execution_metrics.get('on_new_data', 0):.4f}
                    
                    Please continue to improve on these results.
                    """
                })
                
                # Increment iteration count
                state["iteration_count"] = state.get("iteration_count", 0) + 1
                
        except Exception as e:
            print(f"\nError reading metrics: {str(e)}")
            
            # Add metrics error feedback to messages
            state["messages"].append({
                "role": "human",
                "content": f"""
                Your code ran but I had issues reading the metrics:
                
                {str(e)}
                
                Please make sure you're saving metrics correctly to 'metrics_reflection.yaml'.
                """
            })
            
            if state.get("previous_metrics"):
                state["metrics"] = state["previous_metrics"]
        
        # Log metrics
        self._log_metrics(state.get("metrics", {}), state.get("previous_metrics", {}))
        
        # Log token usage
        token_usage = state["token_usage"]
        print(f"\nCurrent Token Usage:")
        print(f"Prompt: {token_usage.get('prompt', 0)}")
        print(f"Completion: {token_usage.get('completion', 0)}")
        print(f"Total: {token_usage.get('total', 0)}")
        
        return state
    
    def should_reflect_or_execute(self, state: ReflectionState) -> str:
        """Determine whether to reflect on or execute the generated code."""
        # Always reflect on the first iteration for each step
        if not state.get("reflections") or len(state.get("reflections", [])) < 1 + state.get("iteration_count", 0):
            return "reflect"
        else:
            return "execute"
    
    def should_continue(self, state: ReflectionState) -> str:
        """Determine if improvement process should continue or end."""
        # Check if we've hit the max consecutive failures
        if state.get("consecutive_failures", 0) >= self.max_failures:
            print(f"\nReached maximum consecutive failures ({self.max_failures}). Ending process.")
            return "end"
            
        # End if we've reached max iterations
        if state.get("iteration_count", 0) >= self.max_iterations:
            print(f"\nReached maximum iterations ({self.max_iterations}). Ending process.")
            return "end"
        
        # Get metrics
        current = state.get("metrics", {}).get('model_new_score', {})
        previous = state.get("previous_metrics", {}).get('model_new_score', {})
        
        # If this is the first iteration with successful execution
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
    
    def _extract_code(self, text: str) -> str:
        """Extract code from response text."""
        import re
        code_blocks = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)
        
        if code_blocks:
            return code_blocks[0]
        else:
            # Fallback if no code block is found
            return text
    
    def _extract_changes(self, text: str) -> List[str]:
        """Extract list of changes from response text."""
        import re
        
        # Try to find a section with changes
        changes_section = re.search(r'changes (I\'ve|made).*?:(.*?)(?:$|```)', text, re.DOTALL | re.IGNORECASE)
        
        if changes_section:
            changes_text = changes_section.group(2)
            # Extract bullet points
            changes = re.findall(r'[-*]\s*(.*?)(?:\n|$)', changes_text)
            return changes
        
        # Fallback - look for any bullet points in the text
        changes = re.findall(r'[-*]\s*(.*?)(?:\n|$)', text)
        
        # Filter to only include likely changes (avoid table of contents, etc.)
        changes = [c for c in changes if len(c) > 10 and any(keyword in c.lower() for keyword in 
                                                           ['add', 'improv', 'chang', 'updat', 'modif', 'implement', 'us'])]
        
        return changes[:5]  # Limit to 5 changes to avoid capturing unrelated bullet points
    
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
    
    def _export_results_to_yaml(self, state: ReflectionState, runtime_seconds: float) -> Dict:
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
        
        # Build improvement path
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
            
            path_entry = {
                "iteration": i + 1,
                "code": entry.get("code", ""),
                "metrics": entry_metrics,
                "changes": entry.get("changes", []),
                "reflection": entry.get("reflection", ""),
                "execution_time": entry.get("execution_time", 0)
            }
            improvement_path.append(path_entry)
        
        # Get token usage from state
        token_usage = state.get("token_usage", {
            "prompt": self.token_counts["prompt"],
            "completion": self.token_counts["completion"],
            "total": self.token_counts["total"]
        })
        
        # Create the standardized output
        result = {
            "agent_name": "reflection",
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
        print(f"  Reflections: {len(state.get('reflections', []))}")
        print(f"  Total tokens used: {token_usage.get('total', 0)}")
        
        return result
    
    def _get_current_timestamp(self) -> str:
        """Get the current timestamp in ISO format."""
        return datetime.utcnow().isoformat() + "Z"
    
    def run(self, initial_state: Dict) -> Dict:
        """Run the reflection-based improvement process.
        
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
        typed_state = ReflectionState(
            model_code=initial_state.get("model_code", ""),
            improved_code="",
            execution_output="",
            metrics=initial_metrics.copy() if initial_metrics else {},
            previous_metrics={},
            iteration_count=0,
            improvement_history=[],
            changes=[],
            reflections=[],
            current_reflection="",
            messages=[],
            dataset_description=dataset_description,
            token_usage={"prompt": 0, "completion": 0, "total": 0},
            iteration_times=[],
            start_time=self.start_time,
            iteration_start_time=None,
            consecutive_failures=0,
            last_successful_state={},
            max_failures_allowed=self.max_failures
        )
        
        print("\n🚀 Starting Reflection-Based Model Improvement Process")
        print(f"Dataset: {dataset_description.get('DATASET_TITLE', 'Unknown')}")
        print(f"Error handling: stopping after {self.max_failures} consecutive failures")
        
        if dataset_description.get('NUMERICAL_FEATURES') and dataset_description.get('CATEGORICAL_FEATURES'):
            print(f"Features: {len(dataset_description.get('NUMERICAL_FEATURES', [])) + len(dataset_description.get('CATEGORICAL_FEATURES', []))} total, {len(dataset_description.get('NUMERICAL_FEATURES', []))} numerical, {len(dataset_description.get('CATEGORICAL_FEATURES', []))} categorical")
        
        try:
            # Run the decision procedure
            final_output = None
            for output in self.graph.stream(typed_state):
                final_output = output
                
                # Log token usage after each step
                for node_name, state in output.items():
                    if "token_usage" in state:
                        token_usage = state["token_usage"]
                        print(f"\nCurrent Token Usage:")
                        print(f"Prompt: {token_usage.get('prompt', 0)}")
                        print(f"Completion: {token_usage.get('completion', 0)}")
                        print(f"Total: {token_usage.get('total', 0)}")
            
            # Get the final state
            final_state = final_output[list(final_output.keys())[-1]]
            
            # Calculate total runtime
            end_time = time.time()
            runtime_seconds = end_time - self.start_time
            
            # Generate final report
            print("\n📊 Reflection-Based Improvement Process Complete")
            print(f"\nTotal runtime: {runtime_seconds:.2f} seconds")
            print(f"Execution attempts: successful={len(final_state.get('improvement_history', []))}, " + 
                  f"failed={final_state.get('consecutive_failures', 0)}")
            
            # Summary of iterations
            if final_state.get("iteration_times"):
                print("\nIteration Times:")
                for iter_time in final_state["iteration_times"]:
                    status = iter_time.get("status", "unknown")
                    print(f"  Iteration {iter_time['iteration']}: {iter_time['time']:.2f} seconds ({status})")
            
            # Create standardized YAML output
            final_state["yaml_output"] = self._export_results_to_yaml(final_state, runtime_seconds)
            
            return final_state
            
        except Exception as e:
            print(f"Error in graph execution: {str(e)}")
            import traceback
            traceback.print_exc()
            raise