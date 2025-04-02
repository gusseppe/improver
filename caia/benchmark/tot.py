from typing import TypedDict, Dict, Any, List, Tuple, Literal, Union, Optional, NamedTuple
import yaml
import textwrap
import os
import time
import json
from datetime import datetime
import operator
from langgraph.graph import StateGraph, END
from rich import print
from rich.panel import Panel
from rich.text import Text
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from operator import add
from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor
from typing_extensions import Annotated


# Define update function for candidates
def update_candidates(
    existing: Optional[list] = None,
    updates: Optional[Union[list, Literal["clear"]]] = None,
) -> List:
    if existing is None:
        existing = []
    if updates is None:
        return existing
    if updates == "clear":
        return []
    # Concatenate the lists
    return existing + updates


class Candidate(NamedTuple):
    """Single candidate ML code solution."""
    code: str
    changes: List[str]
    score: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
    feedback: Optional[str] = None
    
    def __str__(self):
        score_str = f"Score: {self.score}" if self.score is not None else "Unscored"
        changes_str = "\n".join([f"- {change}" for change in self.changes])
        return f"{score_str}\nChanges:\n{changes_str}"


class ScoredCandidate(Candidate):
    """Candidate with evaluation score and feedback."""
    code: str
    changes: List[str]
    score: float
    metrics: Dict[str, Any]
    feedback: str


class ToTState(TypedDict):
    """State for the Tree of Thoughts ML improvement agent."""
    model_code: str                          # Original model code
    initial_code: str                        # Preserved copy of original code 
    initial_metrics: Dict[str, Any]          # Preserved copy of initial metrics
    candidates: Annotated[List[Candidate], update_candidates]  # Current candidate solutions
    scored_candidates: Annotated[List[ScoredCandidate], update_candidates]  # Evaluated candidates
    improvement_history: List[Dict[str, Any]]  # History of all improvements
    metrics: Dict[str, Any]                  # Current metrics
    previous_metrics: Dict[str, Any]         # Metrics from previous iteration
    depth: Annotated[int, operator.add]      # Current search depth
    iteration_count: int                     # Current iteration number
    
    # Additional components for standardized tracking
    dataset_description: Optional[Dict[str, Any]]  # Dataset description in JSON format
    token_usage: Dict[str, int]             # Token usage tracking
    iteration_times: List[Dict[str, Any]]   # Time tracking per iteration
    start_time: float                       # Start time of the process
    iteration_start_time: Optional[float]   # Start time of current iteration
    
    # Error handling components
    consecutive_failures: int               # Track consecutive execution failures
    last_successful_candidates: List[ScoredCandidate]  # Last successful candidates
    max_failures_allowed: int               # Maximum allowed consecutive failures


def prompt_expand_solutions() -> ChatPromptTemplate:
    """Create a prompt for expanding candidate solutions."""
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
            
            metrics: {
                "model_old_score": {
                    "on_old_data": 0.913,
                    "on_new_data": 0.717
                }
            }
            
            num_candidates: 3
            
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
            candidates:
            - code: |
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
                
                # Train on combined data with GradientBoostingClassifier
                X_train = pd.concat([X_train_old, X_train_new])
                y_train = pd.concat([y_train_old, y_train_new])
                
                model_new = GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
                
                model_new.fit(X_train, y_train)
                
                # Evaluate on old distribution
                old_score = accuracy_score(y_test_old, model_new.predict(X_test_old))
                print(f'New model trained and evaluated on old distribution: {old_score}')
                model_new_score['on_old_data'] = float(old_score)
                
                # Evaluate on new distribution
                new_score = accuracy_score(y_test_new, model_new.predict(X_test_new))
                print(f'New model evaluated on new distribution: {new_score}')
                model_new_score['on_new_data'] = float(new_score)
                
                # Save metrics
                with open('metrics_tot.yaml', 'w') as f:
                    yaml.dump({'model_new_score': model_new_score}, f)
              changes:
                - "Switched to GradientBoostingClassifier for better handling of distribution shifts"
                - "Used higher n_estimators (200) for more robust model"
                - "Added max_depth=5 to prevent overfitting"
                - "Combined old and new data for training"
                - "Evaluated on both distributions"
                
            - code: |
                import yaml
                import pandas as pd
                from sklearn.preprocessing import StandardScaler
                from sklearn.ensemble import RandomForestClassifier
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
                
                # Scale features
                scaler = StandardScaler()
                X_train_old_scaled = scaler.fit_transform(X_train_old)
                X_test_old_scaled = scaler.transform(X_test_old)
                X_train_new_scaled = scaler.fit_transform(X_train_new)
                X_test_new_scaled = scaler.transform(X_test_new)
                
                # Train on combined data
                X_train = pd.concat([pd.DataFrame(X_train_old_scaled), pd.DataFrame(X_train_new_scaled)])
                y_train = pd.concat([y_train_old, y_train_new])
                
                model_new = RandomForestClassifier(
                    n_estimators=500,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42
                )
                
                model_new.fit(X_train, y_train)
                
                # Evaluate on old distribution
                old_score = accuracy_score(y_test_old, model_new.predict(X_test_old_scaled))
                print(f'New model trained and evaluated on old distribution: {old_score}')
                model_new_score['on_old_data'] = float(old_score)
                
                # Evaluate on new distribution
                new_score = accuracy_score(y_test_new, model_new.predict(X_test_new_scaled))
                print(f'New model evaluated on new distribution: {new_score}')
                model_new_score['on_new_data'] = float(new_score)
                
                # Save metrics
                with open('metrics_tot.yaml', 'w') as f:
                    yaml.dump({'model_new_score': model_new_score}, f)
              changes:
                - "Added StandardScaler for feature normalization"
                - "Increased n_estimators to 500 for better performance"
                - "Added max_depth=10 to balance model capacity"
                - "Added min_samples_split=5 to prevent overfitting"
                - "Combined old and new data for training"
                
            - code: |
                import yaml
                import pandas as pd
                from sklearn.preprocessing import OneHotEncoder
                from sklearn.ensemble import AdaBoostClassifier
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.metrics import accuracy_score
                from sklearn.compose import ColumnTransformer
                from sklearn.pipeline import Pipeline
                
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
                
                # Define feature columns
                numerical_features = ['Age', 'Income', 'Credit Score', 'Loan Amount', 'Loan Term', 'Interest Rate', 'Employment Length']
                categorical_features = ['Home Ownership', 'Marital Status', 'Dependents']
                
                # Create column transformer for preprocessing
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', 'passthrough', numerical_features),
                        ('cat', OneHotEncoder(drop='first'), categorical_features)
                    ],
                    remainder='drop'
                )
                
                # Combine old and new data
                X_train = pd.concat([X_train_old, X_train_new])
                y_train = pd.concat([y_train_old, y_train_new])
                
                # Create pipeline with preprocessing and AdaBoost
                base_estimator = DecisionTreeClassifier(max_depth=3)
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', AdaBoostClassifier(
                        base_estimator=base_estimator,
                        n_estimators=300,
                        learning_rate=0.05,
                        random_state=42
                    ))
                ])
                
                # Train the pipeline
                pipeline.fit(X_train, y_train)
                
                # Evaluate on old distribution
                old_score = accuracy_score(y_test_old, pipeline.predict(X_test_old))
                print(f'New model trained and evaluated on old distribution: {old_score}')
                model_new_score['on_old_data'] = float(old_score)
                
                # Evaluate on new distribution
                new_score = accuracy_score(y_test_new, pipeline.predict(X_test_new))
                print(f'New model evaluated on new distribution: {new_score}')
                model_new_score['on_new_data'] = float(new_score)
                
                # Save metrics
                with open('metrics_tot.yaml', 'w') as f:
                    yaml.dump({'model_new_score': model_new_score}, f)
              changes:
                - "Implemented proper feature preprocessing with ColumnTransformer"
                - "Used OneHotEncoder for categorical features"
                - "Switched to AdaBoostClassifier with DecisionTreeClassifier base"
                - "Used 300 estimators with reduced learning rate (0.05)"
                - "Created an end-to-end pipeline for reproducibility"
            """
        }
    ]
    for example in examples:
        example["input"] = textwrap.dedent(example["input"]).strip()
        example["output"] = textwrap.dedent(example["output"]).strip()
        
    system_prompt = """
    You are an expert ML engineer tasked with generating multiple candidate solutions for improving an ML model.
    
    Your task is to generate {num_candidates} different approaches to improve the given model code.
    Each approach should be significantly different (e.g., different model types, different preprocessing steps).
    
    For each candidate solution, provide:
    1. Complete, runnable Python code
    2. A list of key changes made to the original code
    
    IMPORTANT REQUIREMENTS:
    - Save metrics to 'metrics_tot.yaml'
    - Use metrics format with the exact key "model_new_score" (not model_old_score)
    - Keep dataset folder paths: "datasets/financial"
    - Ensure each candidate uses a different approach
    - Always train on combined old and new data
    - Evaluate on both test sets
    - Make sure all code is executable
    - Keep the code efficient and avoid operations that might timeout
    - Consider the dataset description to inform your implementation choices
      (feature types, value ranges, etc.)
    
    CRITICAL: You MUST save metrics with 'model_new_score' as the key, NOT 'model_old_score'.
    Example of correct metrics saving:
    
    ```python
    metrics = {{
        'on_new_data': new_score,
        'on_old_data': old_score
    }}
    with open('metrics_tot.yaml', 'w') as f:
        yaml.dump({{'model_new_score': metrics}}, f)
    ```
    
    Response Format: Format your response as YAML with:
    candidates:
    - code: |
        [COMPLETE CODE FOR CANDIDATE 1]
      changes:
        - [CHANGE 1]
        - [CHANGE 2]
        ...
    - code: |
        [COMPLETE CODE FOR CANDIDATE 2]
      changes:
        - [CHANGE 1]
        - [CHANGE 2]
        ...
    ...
    
    Only provide the YAML-formatted output. No additional commentary.
    """
    
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        FewShotChatMessagePromptTemplate(
            example_prompt=ChatPromptTemplate.from_messages([
                ("human", "{input}"),
                ("ai", "{output}")
            ]),
            examples=examples
        ),
        ("human", """
        model_code: |
            {model_code}
        
        metrics: {metrics}
        
        num_candidates: {num_candidates}
        
        dataset_description: {dataset_description}
        
        {previous_feedback}
        """),
    ])


class TreeOfThoughtsGraph:
    def __init__(self, llm, max_iterations=3, beam_width=3, num_candidates=3, threshold=0.9, max_depth=3, max_failures=3, debug=False):
        """Initialize the Tree of Thoughts improvement graph.
        
        Args:
            llm: The language model to use for generation
            max_iterations: Maximum number of improvement iterations to run
            beam_width: Number of candidates to keep after pruning
            num_candidates: Number of candidates to generate in each expansion
            threshold: Score threshold for accepting a solution
            max_depth: Maximum search depth
            max_failures: Maximum number of consecutive execution failures allowed
            debug: Whether to run in debug mode
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.beam_width = beam_width
        self.num_candidates = num_candidates
        self.threshold = threshold
        self.max_depth = max_depth
        self.max_failures = max_failures
        self.debug = debug
        self.graph = self.build_graph()
        self.start_time = None
        self.token_counts = {"prompt": 0, "completion": 0, "total": 0}
        
    def build_graph(self) -> StateGraph:
        """Build the ToT graph structure."""
        workflow = StateGraph(ToTState)
        
        # Add nodes
        workflow.add_node("expand", self.expand_solutions)
        workflow.add_node("score", self.score_candidates)
        workflow.add_node("prune", self.prune_candidates)
        
        # Add edges
        workflow.add_edge("expand", "score")
        workflow.add_edge("score", "prune")
        
        # Add conditional edge from prune
        workflow.add_conditional_edges(
            "prune",
            self.should_terminate,
            {
                "expand": "expand",
                "__end__": END
            }
        )
        
        # Set entry point
        workflow.set_entry_point("expand")
        
        # Compile the graph
        return workflow.compile()
        
    def _record_token_usage(self, state: ToTState, prompt_tokens: int, completion_tokens: int):
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
        
    def expand_solutions(self, state: ToTState, *, config: Dict) -> Dict[str, List[Candidate]]:
        """Generate candidate solutions for improving the ML code."""
        print("\n🌱 EXPANDING CANDIDATE SOLUTIONS")
        
        # Start timing this iteration if not already started
        if "start_time" not in state:
            self.start_time = time.time()
            state["start_time"] = self.start_time
            
        # Start timing this step
        iteration_start_time = time.time()
        state["iteration_start_time"] = iteration_start_time
        
        # Get configuration values
        num_candidates = config.get("configurable", {}).get("num_candidates", self.num_candidates)
        
        # Prepare feedback from previous candidates if available
        previous_feedback = ""
        if state.get("scored_candidates"):
            # Use the top scored candidate feedback if available
            sorted_candidates = sorted(state["scored_candidates"], key=lambda c: c.score, reverse=True)
            if sorted_candidates and sorted_candidates[0].feedback:
                previous_feedback = f"Previous feedback: {sorted_candidates[0].feedback}"
        
        # Prepare prompt input
        prompt_input = {
            "model_code": state["model_code"],
            "metrics": state.get("metrics", {}),
            "num_candidates": num_candidates,
            "dataset_description": state.get("dataset_description", {}),
            "previous_feedback": previous_feedback
        }
        
        # Estimate prompt tokens (rough approximation)
        prompt_text = yaml.dump(prompt_input)
        prompt_tokens = len(prompt_text) // 4
        
        # Get the prompt template
        prompt = prompt_expand_solutions()
        
        # Generate candidates
        chain = prompt | self.llm
        
        try:
            response = chain.invoke(prompt_input)
            
            # Estimate completion tokens 
            response_content = response.content if hasattr(response, 'content') else str(response)
            completion_tokens = len(response_content) // 4
            
            # Record token usage in state
            state = self._record_token_usage(state, prompt_tokens, completion_tokens)
            
            # Parse the response
            result = yaml.safe_load(response_content)
            
            candidates = []
            if "candidates" in result:
                for candidate_data in result["candidates"]:
                    candidate = Candidate(
                        code=candidate_data.get("code", ""),
                        changes=candidate_data.get("changes", []),
                        score=None,
                        metrics=None,
                        feedback=None
                    )
                    candidates.append(candidate)
                    
                print(f"\nGenerated {len(candidates)} candidate solutions")
                
                # Print token usage
                print(f"\nExpansion token usage:")
                print(f"  Prompt: {prompt_tokens}")
                print(f"  Completion: {completion_tokens}")
                print(f"  Total: {prompt_tokens + completion_tokens}")
            else:
                print("\nNo candidates found in the response")
                
            return {"candidates": candidates}
            
        except Exception as e:
            print(f"\nError generating candidates: {str(e)}")
            
            # If we already have previously successful candidates, use them
            if state.get("last_successful_candidates"):
                print(f"Using {len(state['last_successful_candidates'])} previously successful candidates")
                return {"candidates": state["last_successful_candidates"]}
                
            return {"candidates": []}
            
    def score_candidates(self, state: ToTState) -> Dict[str, Any]:
        """Evaluate the candidate solutions by executing the code."""
        print("\n⚖️ SCORING CANDIDATE SOLUTIONS")
        
        candidates = state["candidates"]
        scored_candidates = []
        execution_times = []
        success_count = 0
        
        # Get current consecutive failures count
        consecutive_failures = state.get("consecutive_failures", 0)
        
        for i, candidate in enumerate(candidates):
            candidate_start_time = time.time()
            print(f"\nExecuting candidate {i+1}/{len(candidates)}...")
            
            code = candidate.code
            if not code:
                # Skip empty candidates
                continue
                
            # Execute the code
            wrapped_code = f"```python\n{code}\n```"
            
            # Set up executor
            executor = LocalCommandLineCodeExecutor(timeout=60)  # Extended timeout
            code_executor_agent = ConversableAgent(
                "executor",
                llm_config=False,
                code_execution_config={"executor": executor}
            )
            
            try:
                # Execute the code
                execution_output = code_executor_agent.generate_reply(
                    messages=[{"role": "user", "content": wrapped_code}]
                )
                
                # Print debug information
                print(f"Execution output: {execution_output[:100]}...")
                
                # Check for execution errors
                if self._has_execution_errors(execution_output):
                    # Increment consecutive failures
                    consecutive_failures += 1
                    print(f"⚠️ Execution failed. Consecutive failures: {consecutive_failures}/{self.max_failures}")
                    
                    # Check if we've reached the max failures limit
                    if consecutive_failures >= self.max_failures:
                        # Return early with best candidates we have
                        if state.get("last_successful_candidates"):
                            print(f"❌ Maximum consecutive failures reached. Using previously successful candidates.")
                            return {
                                "scored_candidates": state["last_successful_candidates"], 
                                "candidates": "clear",
                                "consecutive_failures": consecutive_failures
                            }
                    
                    scored_candidate = ScoredCandidate(
                        code=candidate.code,
                        changes=candidate.changes,
                        score=0.0,
                        metrics={},
                        feedback=f"Execution failed: {execution_output}"
                    )
                    scored_candidates.append(scored_candidate)
                    continue
                
                # Execution successful, reset consecutive failures
                consecutive_failures = 0
                success_count += 1
                    
                # Parse metrics
                try:
                    with open('metrics_tot.yaml', 'r') as f:
                        metrics_yaml = f.read()
                        print(f"Metrics file content: {metrics_yaml}")
                        metrics = yaml.safe_load(metrics_yaml)
                        
                    if not metrics or not isinstance(metrics, dict):
                        raise ValueError("Invalid metrics format")
                    
                    print(f"Loaded metrics: {metrics}")
                    
                    # Try to extract model_new_score, but also handle model_old_score as fallback
                    model_score = None
                    if "model_new_score" in metrics:
                        model_score = metrics["model_new_score"]
                        print("Found model_new_score in metrics")
                    elif "model_old_score" in metrics:
                        # If model_old_score is found but not model_new_score, use that instead
                        model_score = metrics["model_old_score"]
                        print("Using model_old_score as fallback")
                    
                    # Extract and calculate scores
                    if model_score and isinstance(model_score, dict):
                        old_data_score = model_score.get("on_old_data", 0)
                        new_data_score = model_score.get("on_new_data", 0)
                        
                        # Calculate the weighted score - prioritize new distribution performance
                        # while maintaining old distribution performance
                        weighted_score = 0.7 * new_data_score + 0.3 * old_data_score
                        
                        # Calculate improvement over previous metrics
                        previous_old = state.get("metrics", {}).get("model_old_score", {}).get("on_old_data", 0)
                        previous_new = state.get("metrics", {}).get("model_old_score", {}).get("on_new_data", 0)
                        previous_weighted = 0.7 * previous_new + 0.3 * previous_old
                        
                        # Final score is the relative improvement
                        if previous_weighted > 0:
                            improvement = weighted_score / previous_weighted - 1
                            score = 0.5 + improvement  # Normalize around 0.5
                        else:
                            score = weighted_score
                            
                        # Ensure score is between 0 and 1
                        score = max(0, min(1, score))
                        
                        # Create feedback
                        feedback = (
                            f"Old distribution: {old_data_score:.4f} (was {previous_old:.4f})\n"
                            f"New distribution: {new_data_score:.4f} (was {previous_new:.4f})\n"
                            f"Weighted score: {weighted_score:.4f}\n"
                            f"Improvement: {improvement:.2%}"
                        )
                        
                        candidate_end_time = time.time()
                        execution_time = candidate_end_time - candidate_start_time
                        
                        # Record execution time
                        execution_times.append({
                            "candidate": i,
                            "time": execution_time
                        })
                        
                        scored_candidate = ScoredCandidate(
                            code=candidate.code,
                            changes=candidate.changes,
                            score=score,
                            metrics={"on_old_data": old_data_score, "on_new_data": new_data_score},
                            feedback=feedback
                        )
                        scored_candidates.append(scored_candidate)
                        print(f"Candidate {i+1} execution time: {execution_time:.2f} seconds")
                    else:
                        # No model_new_score or model_old_score found, or it's not a dict
                        scored_candidate = ScoredCandidate(
                            code=candidate.code,
                            changes=candidate.changes,
                            score=0.0,
                            metrics={},
                            feedback="No valid model score metrics found in metrics file"
                        )
                        scored_candidates.append(scored_candidate)
                        
                except Exception as e:
                    # Error parsing metrics
                    print(f"Error parsing metrics: {str(e)}")
                    scored_candidate = ScoredCandidate(
                        code=candidate.code,
                        changes=candidate.changes,
                        score=0.0,
                        metrics={},
                        feedback=f"Error parsing metrics: {str(e)}"
                    )
                    scored_candidates.append(scored_candidate)
                    
            except Exception as e:
                # Execution exception
                print(f"Execution exception: {str(e)}")
                # Increment consecutive failures
                consecutive_failures += 1
                print(f"⚠️ Execution failed. Consecutive failures: {consecutive_failures}/{self.max_failures}")
                
                scored_candidate = ScoredCandidate(
                    code=candidate.code,
                    changes=candidate.changes,
                    score=0.0,
                    metrics={},
                    feedback=f"Execution exception: {str(e)}"
                )
                scored_candidates.append(scored_candidate)
        
        print(f"\nScored {len(scored_candidates)} candidates")
        print(f"Successful executions: {success_count}/{len(candidates)}")
        
        # If we have successful candidates, store them for recovery
        if scored_candidates and any(c.score > 0 for c in scored_candidates):
            successful_candidates = [c for c in scored_candidates if c.score > 0]
            if successful_candidates:
                state["last_successful_candidates"] = successful_candidates
        
        # Calculate the average execution time if available
        if execution_times:
            avg_time = sum(entry["time"] for entry in execution_times) / len(execution_times)
            print(f"Average execution time: {avg_time:.2f} seconds")
            
            # Store the execution times
            if "iteration_times" not in state:
                state["iteration_times"] = []
                
            state["iteration_times"].extend(execution_times)
        
        return {
            "scored_candidates": scored_candidates, 
            "candidates": "clear",
            "consecutive_failures": consecutive_failures
        }
        
    def _has_execution_errors(self, output: str) -> bool:
        """Check if execution output contains errors."""
        error_indicators = [
            'error', 'exception', 'failed', 'failure',
            'traceback', 'exitcode: 1'
        ]
        return any(indicator in output.lower() for indicator in error_indicators)
        
    def prune_candidates(self, state: ToTState, *, config: Dict) -> Dict[str, Any]:
        """Select the best candidates for the next iteration."""
        print("\n✂️ PRUNING CANDIDATES")
        
        scored_candidates = state["scored_candidates"]
        
        # Get beam width from config
        beam_width = config.get("configurable", {}).get("beam_width", self.beam_width)
        
        # Sort candidates by score
        sorted_candidates = sorted(scored_candidates, key=lambda c: c.score, reverse=True)
        
        # Take top K candidates
        pruned_candidates = sorted_candidates[:beam_width]
        
        # Print scores of pruned candidates
        for i, candidate in enumerate(pruned_candidates):
            print(f"\nCandidate {i+1}: Score = {candidate.score:.4f}")
            print(f"Feedback: {candidate.feedback}")
        
        # Update improvement history with the best candidate
        if pruned_candidates and pruned_candidates[0].score > 0:
            best_candidate = pruned_candidates[0]
            
            # Calculate iteration time
            if "iteration_start_time" in state and state["iteration_start_time"] is not None:
                iteration_end_time = time.time()
                iteration_time = iteration_end_time - state["iteration_start_time"]
            else:
                iteration_time = 0
                
            # Get current token usage
            token_usage = state.get("token_usage", {
                "prompt": self.token_counts["prompt"],
                "completion": self.token_counts["completion"],
                "total": self.token_counts["total"]
            })
            
            # Add it to the improvement history
            improvement_entry = {
                "iteration": state.get("iteration_count", 0) + 1,
                "code": best_candidate.code,
                "changes": best_candidate.changes,
                "metrics": best_candidate.metrics,
                "score": best_candidate.score,
                "feedback": best_candidate.feedback,
                "execution_time": iteration_time,
                "token_usage": {
                    "prompt": token_usage.get("prompt", 0),
                    "completion": token_usage.get("completion", 0),
                    "total": token_usage.get("total", 0)
                }
            }
            
            # Get the current improvement history or initialize it
            improvement_history = state.get("improvement_history", [])
            improvement_history.append(improvement_entry)
            
            # Increment iteration count
            iteration_count = state.get("iteration_count", 0) + 1
            
            print(f"\nIteration {iteration_count} completed in {iteration_time:.2f} seconds")
            print(f"Best candidate score: {best_candidate.score:.4f}")
            print(f"Token usage: {token_usage.get('total', 0)} total tokens")
            
            return {
                "candidates": pruned_candidates,
                "scored_candidates": "clear",
                "depth": 1,
                "improvement_history": improvement_history,
                "iteration_count": iteration_count
            }
        
        return {
            "candidates": pruned_candidates,
            "scored_candidates": "clear",
            "depth": 1
        }
        
    def should_terminate(self, state: ToTState, config: Dict) -> Union[Literal["expand"], Literal["__end__"]]:
        """Determine if the process should continue or end."""
        # Get threshold and max depth from config
        threshold = config.get("configurable", {}).get("threshold", self.threshold)
        max_depth = config.get("configurable", {}).get("max_depth", self.max_depth)
        max_iterations = config.get("configurable", {}).get("max_iterations", self.max_iterations)
        max_failures = config.get("configurable", {}).get("max_failures", self.max_failures)
        
        # Check if we've reached max iterations
        if state.get("iteration_count", 0) >= max_iterations:
            print(f"\nReached maximum iterations ({max_iterations}). Ending process.")
            return "__end__"
        
        # Check if we've reached max depth
        if state.get("depth", 0) >= max_depth:
            print(f"\nReached maximum depth ({max_depth}). Ending process.")
            return "__end__"
            
        # Check if we've hit max consecutive failures
        if state.get("consecutive_failures", 0) >= max_failures:
            print(f"\nReached maximum consecutive failures ({max_failures}). Ending process.")
            return "__end__"
        
        # Check if we have any candidates
        if not state.get("candidates"):
            print("\nNo candidates to expand. Ending process.")
            return "__end__"
        
        # Check if the best candidate exceeds the threshold
        best_candidate = state.get("candidates", [])[0] if state.get("candidates") else None
        if best_candidate and best_candidate.score >= threshold:
            print(f"\nFound candidate with score {best_candidate.score:.4f} >= threshold {threshold}. Ending process.")
            return "__end__"
        
        # Continue with expansion
        return "expand"
    
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
            
    def _export_results_to_yaml(self, state: Dict, runtime_seconds: float) -> Dict:
        """Export the results to a standardized YAML-compatible dictionary format."""
        # Extract initial code
        initial_code = state.get("initial_code", state.get("model_code", ""))
        
        # Extract initial metrics from preserved state
        initial_metrics = {}
        if state.get("initial_metrics") and "model_old_score" in state.get("initial_metrics", {}):
            old_metrics = state["initial_metrics"]["model_old_score"]
            initial_metrics = {
                "old_distribution": old_metrics.get("on_old_data", 0),
                "new_distribution": old_metrics.get("on_new_data", 0)
            }
        
        # Extract final metrics and code
        final_metrics = {}
        final_code = ""
        
        # Try to get from improvement history
        if state.get("improvement_history"):
            last_entry = state["improvement_history"][-1]
            if "metrics" in last_entry:
                metrics = last_entry["metrics"]
                final_metrics = {
                    "old_distribution": metrics.get("on_old_data", 0),
                    "new_distribution": metrics.get("on_new_data", 0)
                }
            final_code = last_entry.get("code", "")
        # If no improvement history, try candidates
        elif state.get("candidates") and len(state["candidates"]) > 0:
            best_candidate = state["candidates"][0]
            if best_candidate.metrics:
                final_metrics = {
                    "old_distribution": best_candidate.metrics.get("on_old_data", 0),
                    "new_distribution": best_candidate.metrics.get("on_new_data", 0)
                }
            final_code = best_candidate.code
        # Final fallback to original code
        else:
            final_code = state.get("model_code", "")
        
        # Build improvement path with more detailed metadata
        improvement_path = []
        for i, entry in enumerate(state.get("improvement_history", [])):
            # Create reflection from feedback
            reflection = f"Tree of Thoughts iteration {i+1}"
            if "feedback" in entry:
                reflection = entry["feedback"]
            
            path_entry = {
                "iteration": i + 1,
                "code": entry.get("code", ""),
                "metrics": {
                    "old_distribution": entry.get("metrics", {}).get("on_old_data", 0),
                    "new_distribution": entry.get("metrics", {}).get("on_new_data", 0)
                },
                "changes": entry.get("changes", []),
                "reflection": reflection,
                "execution_time": entry.get("execution_time", 0),
                "score": entry.get("score", 0)
            }
            improvement_path.append(path_entry)
        
        # Get token usage from state or class counters
        token_usage = state.get("token_usage", {
            "prompt": self.token_counts["prompt"],
            "completion": self.token_counts["completion"],
            "total": self.token_counts["total"]
        })
        
        # Calculate number of successful and failed executions
        execution_attempts = state.get("iteration_count", 0) + state.get("consecutive_failures", 0)
        successful_executions = state.get("iteration_count", 0)
        failed_executions = state.get("consecutive_failures", 0)
        
        # Create the standardized output
        result = {
            "agent_name": "tot",
            "initial_code": initial_code,
            "initial_metrics": initial_metrics,
            "improvement_path": improvement_path,
            "final_code": final_code,
            "final_metrics": final_metrics,
            "runtime_statistics": {
                "total_time_seconds": runtime_seconds,
                "iterations": state.get("iteration_count", 0),
                "tokens_used": token_usage.get("total", 0),
                "prompt_tokens": token_usage.get("prompt", 0),
                "completion_tokens": token_usage.get("completion", 0),
                "beam_width": self.beam_width,
                "num_candidates": self.num_candidates,
                "iteration_times": state.get("iteration_times", []),
                "evaluation_timestamp": self._get_current_timestamp(),
                "execution_attempts": execution_attempts,
                "successful_executions": successful_executions,
                "failed_executions": failed_executions
            }
        }
        
        # Print debug information
        print(f"\nExporting results:")
        print(f"  Initial metrics: {initial_metrics}")
        print(f"  Final metrics: {final_metrics}")
        print(f"  Improvement path: {len(improvement_path)} entries")
        print(f"  Total tokens used: {token_usage.get('total', 0)}")
        print(f"  Execution stats: {successful_executions} successes, {failed_executions} failures")
        
        return result
    
    def _get_current_timestamp(self) -> str:
        """Get the current timestamp in ISO format."""
        return datetime.utcnow().isoformat() + "Z"
    
    def run(self, initial_state: Dict) -> Dict:
        """Run the Tree of Thoughts improvement process with enhanced error handling.
        
        Args:
            initial_state: Dictionary containing the initial state, should include:
                - model_code: The original model code to improve
                - metrics: Optional initial metrics (model_old_score)
                - max_iterations: (Optional) Override the default max iterations
                - beam_width: (Optional) Override the default beam width
                - num_candidates: (Optional) Override the default number of candidates
                - threshold: (Optional) Override the default threshold
                - max_depth: (Optional) Override the default max depth
                - max_failures: (Optional) Override the default max failures allowed
                - dataset_description: (Optional) Dataset description in JSON format
                
        Returns:
            The final state after improvement, including the improved code and metrics.
        """
        self.start_time = time.time()
        
        # Override parameters if provided
        if "max_iterations" in initial_state:
            self.max_iterations = initial_state.pop("max_iterations")
            print(f"Max iterations set to: {self.max_iterations}")
            
        if "beam_width" in initial_state:
            self.beam_width = initial_state.pop("beam_width")
            print(f"Beam width set to: {self.beam_width}")
            
        if "num_candidates" in initial_state:
            self.num_candidates = initial_state.pop("num_candidates")
            print(f"Number of candidates set to: {self.num_candidates}")
            
        if "threshold" in initial_state:
            self.threshold = initial_state.pop("threshold")
            print(f"Threshold set to: {self.threshold}")
            
        if "max_depth" in initial_state:
            self.max_depth = initial_state.pop("max_depth")
            print(f"Max depth set to: {self.max_depth}")
            
        if "max_failures" in initial_state:
            self.max_failures = initial_state.pop("max_failures")
            print(f"Max failures set to: {self.max_failures}")
        
        # Extract initial values
        initial_metrics = initial_state.get("metrics", {})
        initial_code = initial_state.get("model_code", "")
        
        # Extract dataset description if provided
        dataset_description = initial_state.get("dataset_description", {})
        
        # Create properly typed initial state
        typed_state = ToTState(
            model_code=initial_code,
            initial_code=initial_code,  # Store a copy of the original code
            initial_metrics=initial_metrics.copy() if initial_metrics else {},  # Store a copy of the original metrics
            candidates=[],
            scored_candidates=[],
            improvement_history=[],
            metrics=initial_metrics.copy() if initial_metrics else {},
            previous_metrics={},
            depth=0,
            iteration_count=0,
            dataset_description=dataset_description,
            token_usage={"prompt": 0, "completion": 0, "total": 0},
            iteration_times=[],
            start_time=self.start_time,
            iteration_start_time=None,
            consecutive_failures=0,
            last_successful_candidates=[],
            max_failures_allowed=self.max_failures
        )
        
        # Create config with parameters
        config = {
            "configurable": {
                "max_iterations": self.max_iterations,
                "beam_width": self.beam_width,
                "num_candidates": self.num_candidates,
                "threshold": self.threshold,
                "max_depth": self.max_depth,
                "max_failures": self.max_failures
            }
        }
        
        print("\n🚀 Starting Tree of Thoughts Model Improvement Process")
        print(f"Parameters: max_iterations={self.max_iterations}, beam_width={self.beam_width}, num_candidates={self.num_candidates}")
        print(f"Error handling: stopping after {self.max_failures} consecutive failures")
        
        if dataset_description:
            print(f"Dataset: {dataset_description.get('DATASET_TITLE', 'Unknown')}")
            
            if dataset_description.get('NUMERICAL_FEATURES') and dataset_description.get('CATEGORICAL_FEATURES'):
                print(f"Features: {len(dataset_description.get('NUMERICAL_FEATURES', [])) + len(dataset_description.get('CATEGORICAL_FEATURES', []))} total, {len(dataset_description.get('NUMERICAL_FEATURES', []))} numerical, {len(dataset_description.get('CATEGORICAL_FEATURES', []))} categorical")
        
        try:
            # Run the decision procedure
            final_output = None
            for output in self.graph.stream(typed_state, config=config):
                final_output = output
                
                # Log current state
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
            print("\n📊 Tree of Thoughts Improvement Process Complete")
            print(f"\nTotal runtime: {runtime_seconds:.2f} seconds")
            print(f"Execution attempts: successful={final_state.get('iteration_count', 0)}, " + 
                  f"failed={final_state.get('consecutive_failures', 0)}")
                  
            # Log final metrics
            initial_metrics_copy = initial_metrics.copy() if initial_metrics else {}
            if "model_old_score" in initial_metrics_copy:
                self._log_metrics(
                    {"model_new_score": final_state.get("improvement_history", [{}])[-1].get("metrics", {}) if final_state.get("improvement_history") else {}},
                    {"model_old_score": initial_metrics_copy["model_old_score"]}
                )
            
            # Create standardized YAML output
            final_state["yaml_output"] = self._export_results_to_yaml(final_state, runtime_seconds)
            
            return final_state
            
        except Exception as e:
            print(f"Error in graph execution: {str(e)}")
            import traceback
            traceback.print_exc()
            raise