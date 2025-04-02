from typing import TypedDict, Dict, Any, List, Optional, Literal
import yaml
import os
import json
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor
import time
from datetime import datetime
import re
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate

class SelfDiscoverState(TypedDict):
    """State for Self-Discovery based ML code improvement."""
    # Core code improvement state
    model_code: str
    improved_code: str
    execution_output: str
    metrics: Dict[str, Any]
    previous_metrics: Dict[str, Any]
    iteration_count: int
    improvement_history: List[Dict[str, Any]]
    changes: List[str]
    
    # Self-Discovery specific components
    reasoning_modules: List[str]
    selected_modules: Optional[str]
    adapted_modules: Optional[str]
    reasoning_structure: Optional[str]
    messages: List[Dict[str, Any]]
    
    # Tracking and error handling components
    dataset_description: Optional[Dict[str, Any]]
    token_usage: Dict[str, int]
    iteration_times: List[Dict[str, Any]]
    start_time: float
    iteration_start_time: Optional[float]
    consecutive_failures: int
    last_successful_state: Dict[str, Any]
    max_failures_allowed: int

REASONING_MODULES = [
    "1. How could I simplify the problem so that it is easier to solve?",
    "2. What are the key techniques for improving an ML model on distribution shifts?",
    "3. How can I implement a robust solution that works with the given datasets?",
    "4. What practical improvements would yield the best results with minimal complexity?"
]

# REASONING_MODULES = [
#     "1. How could I devise an experiment to help solve that problem?",
#     "2. Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
#     "4. How can I simplify the problem so that it is easier to solve?",
#     "5. What are the key assumptions underlying this problem?",
#     "6. What are the potential risks and drawbacks of each solution?",
#     "7. What are the alternative perspectives or viewpoints on this problem?",
#     "8. What are the long-term implications of this problem and its solutions?",
#     "9. How can I break down this problem into smaller, more manageable parts?",
#     "10. Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
#     "11. Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
#     "13. Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
#     "14. Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
#     "16. What is the core issue or problem that needs to be addressed?",
#     "17. What are the underlying causes or factors contributing to the problem?",
#     "18. Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
#     "19. What are the potential obstacles or challenges that might arise in solving this problem?",
#     "20. Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
#     "21. Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
#     "22. What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
#     "23. How can progress or success in solving the problem be measured or evaluated?",
#     "24. What indicators or metrics can be used?",
#     "25. Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
#     "26. Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
#     "27. Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
#     "28. Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
#     "29. Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
#     "30. Is the problem a design challenge that requires creative solutions and innovation?",
#     "31. Does the problem require addressing systemic or structural issues rather than just individual instances?",
#     "32. Is the problem time-sensitive or urgent, requiring immediate attention and action?",
#     "33. What kinds of solution typically are produced for this kind of problem specification?",
#     "34. Given the problem specification and the current best solution, have a guess about other possible solutions.",
#     "35. Let's imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?",
#     "36. What is the best way to modify this current best solution, given what you know about these kinds of problem specification?",
#     "37. Ignoring the current best solution, create an entirely new solution to the problem.",
#     "39. Let's make a step by step plan and implement it with good notation and explanation.",
# ]


def prompt_select_modules() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", """You are an expert ML architect. Select the 3 most relevant reasoning modules for improving model performance across data distributions.
Available Modules:
{modules}
Current Problem:
- Initial Code: {code}
- Initial Metrics: {metrics}
- Dataset Features: {dataset_features}
Selection Guidelines:
1. Choose modules that directly address the performance gap
2. Prioritize techniques with implementation feasibility
3. Consider computational constraints
Respond ONLY with comma-separated module numbers (e.g., "2,5,7")"""),
    ])


def prompt_adapt_modules() -> ChatPromptTemplate:
    """Create a simpler prompt for adapting the modules."""
    return ChatPromptTemplate.from_messages([
        ("system", """
        Adapt these selected modules to create a practical ML solution:

        Selected Modules: {selected}

        Problem Context:
        - Dataset Path: datasets/healthcare
        - Dataset Features: {dataset_features}
        
        Your adaptations should focus on:
        1. Simple, effective preprocessing techniques
        2. Model selection (RandomForest or GradientBoosting)
        3. Practical ways to combine old and new data
        4. Effective evaluation metrics

        Keep your adaptations concise and practical - no complex code snippets needed.
        """)
    ])


def prompt_structure_plan() -> ChatPromptTemplate:
    """Create prompt for structuring the implementation plan."""
    system_template = """
    Create a practical implementation plan from the adapted modules:

    {adapted}

    Dataset Info:
    - Features: {dataset_features}

    Focus on creating a simple, clear plan that:
    1. Loads and preprocesses both old and new data
    2. Implements a baseline model 
    3. Implements an improved model using the adapted modules
    4. Evaluates and compares both models
    5. Saves metrics in the correct format

    FORMAT YOUR PLAN IN 4 SECTIONS:
    1. Data Loading and Preparation
    2. Baseline Model Implementation
    3. Improved Model Implementation
    4. Evaluation and Metrics

    Keep your plan simple, concrete and executable - focus on essential steps only.
    Your plan will be implemented in Python code using sklearn components.
    """
    
    return ChatPromptTemplate.from_messages([
        ("system", system_template)
    ])



def prompt_generate_solution() -> ChatPromptTemplate:
    """Create a very explicit prompt for generating code."""
    system_template = """
    You are an expert ML engineer implementing a simple solution for a dataset with distribution shifts.

    Plan: {plan}
    Dataset Features: {dataset_features}
    Dataset Path: {dataset_folder}
    
    CRITICAL REQUIREMENTS:
    1. Follow this EXACT structure with NO deviations:
    ```python
    import pandas as pd
    import numpy as np
    import yaml
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier  # or RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Initialize metrics
    model_new_score = {{
        'on_old_data': 0.0,
        'on_new_data': 0.0
    }}
    
    # Load data - USE THESE EXACT PATHS
    dataset_folder = "{dataset_folder}"
    X_train_old = pd.read_csv(f"{{{{dataset_folder}}}}/X_train_old.csv")
    X_test_old = pd.read_csv(f"{{{{dataset_folder}}}}/X_test_old.csv")
    y_train_old = pd.read_csv(f"{{{{dataset_folder}}}}/y_train_old.csv").squeeze("columns")
    y_test_old = pd.read_csv(f"{{{{dataset_folder}}}}/y_test_old.csv").squeeze("columns")
    
    X_train_new = pd.read_csv(f"{{{{dataset_folder}}}}/X_train_new.csv")
    X_test_new = pd.read_csv(f"{{{{dataset_folder}}}}/X_test_new.csv")
    y_train_new = pd.read_csv(f"{{{{dataset_folder}}}}/y_train_new.csv").squeeze("columns")
    y_test_new = pd.read_csv(f"{{{{dataset_folder}}}}/y_test_new.csv").squeeze("columns")
    
    # [YOUR PREPROCESSING CODE HERE]
    
    # Combine datasets
    X_train_combined = pd.concat([X_train_old, X_train_new])
    y_train_combined = pd.concat([y_train_old, y_train_new])
    
    # Train model
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train_combined, y_train_combined)
    
    # Evaluate on old distribution
    old_score = accuracy_score(y_test_old, model.predict(X_test_old))
    print(f'Model evaluated on old distribution: {{{{old_score}}}}')
    model_new_score['on_old_data'] = float(old_score)
    
    # Evaluate on new distribution
    new_score = accuracy_score(y_test_new, model.predict(X_test_new))
    print(f'Model evaluated on new distribution: {{{{new_score}}}}')
    model_new_score['on_new_data'] = float(new_score)
    
    # Save metrics
    with open('metrics_self_discovery.yaml', 'w') as f:
        yaml.dump({{'model_new_score': model_new_score}}, f)
    ```
    
    2. Keep preprocessing SIMPLE:
    - Use StandardScaler only if needed
    - Avoid one-hot encoding unless you're certain it's needed
    - No complex grid search or cross-validation
    
    3. Use ONLY GradientBoostingClassifier or RandomForestClassifier

    4. Save metrics EXACTLY as shown above
    
    Provide ONLY the complete code, NO explanations.
    """
    
    return ChatPromptTemplate.from_messages([
        ("system", system_template)
    ])


class SelfDiscoverGraph:
    def __init__(self, llm, max_iterations=1, max_failures=3, debug=False):
        """Initialize the self-discovery graph with error handling.
        
        Args:
            llm: The language model to use for generation
            max_iterations: Maximum number of improvement iterations to run
            max_failures: Maximum consecutive execution failures allowed before stopping
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
        workflow = StateGraph(SelfDiscoverState)
        
        workflow.add_node("select", self.select_node)
        workflow.add_node("adapt", self.adapt_node)
        workflow.add_node("structure", self.structure_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("execute", self.execute_node)
        
        workflow.set_entry_point("select")
        
        workflow.add_edge("select", "adapt")
        workflow.add_edge("adapt", "structure")
        workflow.add_edge("structure", "generate")
        workflow.add_edge("generate", "execute")
        workflow.add_conditional_edges(
            "execute",
            self.should_continue,
            {"continue": "select", "end": END}
        )
        
        return workflow.compile()

    def _record_token_usage(self, state: SelfDiscoverState, prompt_tokens: int, completion_tokens: int):
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

    def _get_fallback_code(self, dataset_folder: str) -> str:
        """Generate reliable fallback code if all else fails."""
        return f"""
    import pandas as pd
    import numpy as np
    import yaml
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score

    # Initialize metrics
    model_new_score = {{
        'on_old_data': 0.0,
        'on_new_data': 0.0
    }}

    # Load data
    dataset_folder = "{dataset_folder}"
    X_train_old = pd.read_csv(f"{{dataset_folder}}/X_train_old.csv")
    X_test_old = pd.read_csv(f"{{dataset_folder}}/X_test_old.csv")
    y_train_old = pd.read_csv(f"{{dataset_folder}}/y_train_old.csv").squeeze("columns")
    y_test_old = pd.read_csv(f"{{dataset_folder}}/y_test_old.csv").squeeze("columns")

    X_train_new = pd.read_csv(f"{{dataset_folder}}/X_train_new.csv")
    X_test_new = pd.read_csv(f"{{dataset_folder}}/X_test_new.csv")
    y_train_new = pd.read_csv(f"{{dataset_folder}}/y_train_new.csv").squeeze("columns")
    y_test_new = pd.read_csv(f"{{dataset_folder}}/y_test_new.csv").squeeze("columns")

    # Combine training data
    X_train_combined = pd.concat([X_train_old, X_train_new])
    y_train_combined = pd.concat([y_train_old, y_train_new])

    # Train model
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_combined, y_train_combined)

    # Evaluate on old distribution
    old_score = accuracy_score(y_test_old, model.predict(X_test_old))
    print(f'Model evaluated on old distribution: {{old_score}}')
    model_new_score['on_old_data'] = float(old_score)

    # Evaluate on new distribution
    new_score = accuracy_score(y_test_new, model.predict(X_test_new))
    print(f'Model evaluated on new distribution: {{new_score}}')
    model_new_score['on_new_data'] = float(new_score)

    # Save metrics
    with open('metrics_self_discovery.yaml', 'w') as f:
        yaml.dump({{'model_new_score': model_new_score}}, f)
    """

    def _get_dataset_feature_str(self, state: SelfDiscoverState) -> str:
        """Get a formatted string of dataset features."""
        dataset_desc = state.get("dataset_description", {})
        if not dataset_desc:
            return "Not provided"
            
        numerical = dataset_desc.get("NUMERICAL_FEATURES", [])
        categorical = dataset_desc.get("CATEGORICAL_FEATURES", [])
        
        features_str = f"{len(numerical) + len(categorical)} total features: "
        features_str += f"{len(numerical)} numerical ({', '.join(numerical[:3])}{'...' if len(numerical) > 3 else ''}) and "
        features_str += f"{len(categorical)} categorical ({', '.join(categorical[:3])}{'...' if len(categorical) > 3 else ''})"
        return features_str

    def select_node(self, state: SelfDiscoverState) -> SelfDiscoverState:
        """Simplified select node that just picks some reasoning modules."""
        print("\n🔍 SELECTING REASONING MODULES")
        
        # Start iteration timing
        self.iteration_start_time = time.time()
        state["iteration_start_time"] = self.iteration_start_time
        
        # Select modules directly without LLM (simpler approach)
        selected_modules = REASONING_MODULES[:3]  # Just use the first three modules
        
        state["selected_modules"] = "\n".join(selected_modules)
        
        print(f"Selected modules: {len(selected_modules)}")
        for i, module in enumerate(selected_modules):
            print(f"- {module[:20]}...")
        
        return state

    def adapt_node(self, state: SelfDiscoverState) -> SelfDiscoverState:
        """Adapt selected modules to the problem."""
        print("\n🛠️ ADAPTING MODULES")
        
        dataset_features = self._get_dataset_feature_str(state)
        
        # Estimate prompt tokens
        prompt_text = f"selected:{state['selected_modules']}\nold_acc:{state['metrics']['model_old_score']['on_old_data']}\nnew_acc:{state['metrics']['model_old_score']['on_new_data']}\ndataset:{dataset_features}\nlimitations:{self._analyze_code_limitations(state['model_code'])}"
        prompt_tokens = len(prompt_text) // 4
        
        prompt = prompt_adapt_modules().format_messages(
            selected=state["selected_modules"],
            old_acc=state["metrics"]["model_old_score"]["on_old_data"],
            new_acc=state["metrics"]["model_old_score"]["on_new_data"],
            dataset_features=dataset_features,
            code_limitations=self._analyze_code_limitations(state["model_code"])
        )
        
        response = self.llm.invoke(prompt).content
        
        # Estimate completion tokens 
        completion_tokens = len(response) // 4
        
        # Record token usage in state
        state = self._record_token_usage(state, prompt_tokens, completion_tokens)
        
        state["adapted_modules"] = response
        print(f"Adapted modules: {state['adapted_modules']}")
        
        # Log token usage
        print(f"\nCurrent Token Usage:")
        print(f"Prompt: {state['token_usage'].get('prompt', 0)}")
        print(f"Completion: {state['token_usage'].get('completion', 0)}")
        print(f"Total: {state['token_usage'].get('total', 0)}")
        
        return state

    def structure_node(self, state: SelfDiscoverState) -> SelfDiscoverState:
        """Create implementation plan."""
        print("\n📝 STRUCTURING PLAN")
        
        dataset_features = self._get_dataset_feature_str(state)
        
        # Extract dataset folder for later use
        dataset_folder = self._extract_dataset_folder(state)
        state["dataset_folder"] = dataset_folder
        print(f"Using dataset folder: {dataset_folder}")
        
        # Estimate prompt tokens
        prompt_text = f"adapted:{state['adapted_modules']}\ndataset:{dataset_features}"
        prompt_tokens = len(prompt_text) // 4
        
        prompt = prompt_structure_plan().format_messages(
            adapted=state["adapted_modules"],
            dataset_features=dataset_features
        )
        
        response = self.llm.invoke(prompt).content
        
        # Estimate completion tokens 
        completion_tokens = len(response) // 4
        
        # Record token usage in state
        state = self._record_token_usage(state, prompt_tokens, completion_tokens)
        
        state["reasoning_structure"] = response
        print(f"Reasoning structure: {state['reasoning_structure']}")
        
        # Log token usage
        print(f"\nCurrent Token Usage:")
        print(f"Prompt: {state['token_usage'].get('prompt', 0)}")
        print(f"Completion: {state['token_usage'].get('completion', 0)}")
        print(f"Total: {state['token_usage'].get('total', 0)}")
        
        return state
    
    def _extract_dataset_folder(self, state: SelfDiscoverState) -> str:
        """
        Extract the dataset folder from the original model code.
        Default to "datasets/financial" if not found.
        """
        code = state.get("model_code", "")
        folder_match = re.search(r'dataset_folder\s*=\s*["\']([^"\']+)["\']', code)
        if folder_match:
            return folder_match.group(1)
        
        # Check different patterns
        path_match = re.search(r'["\']([^"\']*dataset[^"\']*)/[^"\']+["\']', code)
        if path_match:
            return path_match.group(1)
        
        # Default to financial
        return "datasets/healthcare"


    def generate_node(self, state: SelfDiscoverState) -> SelfDiscoverState:
        """Generate improved code."""
        print("\n💡 GENERATING SOLUTION")
        
        dataset_features = self._get_dataset_feature_str(state)
        dataset_folder = state.get("dataset_folder", "datasets/financial")
        
        # Estimate prompt tokens
        prompt_text = f"plan:{state['reasoning_structure']}\ndataset:{dataset_features}\ndataset_folder:{dataset_folder}"
        prompt_tokens = len(prompt_text) // 4
        
        # Create the prompt
        prompt = prompt_generate_solution().format(
            plan=state["reasoning_structure"],
            dataset_features=dataset_features,
            dataset_folder=dataset_folder
        )
        
        # Generate the solution
        response = self.llm.invoke(prompt)
        
        # Get the response content
        if hasattr(response, 'content'):
            response_content = response.content
        else:
            response_content = str(response)
        
        # Estimate completion tokens 
        completion_tokens = len(response_content) // 4
        
        # Record token usage in state
        state = self._record_token_usage(state, prompt_tokens, completion_tokens)
        
        # Extract the code using the simplified extractor
        code = self._extract_code(response_content)
        
        # Make sure we actually have code
        if not code or len(code.strip()) < 100:  # Basic check - real code should be longer
            print("⚠️ Generated content does not appear to be valid code")
            # Use a simple fallback template if needed
            code = self._get_fallback_code(dataset_folder)
        
        # Set the improved code in the state
        state["improved_code"] = code
        
        # Extract changes if possible
        changes = self._extract_changes(response_content)
        state["changes"] = changes
        
        print(f"Generated improved code with {len(changes)} changes")
        
        # Log token usage
        print(f"\nCurrent Token Usage:")
        print(f"Prompt: {state['token_usage'].get('prompt', 0)}")
        print(f"Completion: {state['token_usage'].get('completion', 0)}")
        print(f"Total: {state['token_usage'].get('total', 0)}")
        
        return state

    
    def execute_node(self, state: SelfDiscoverState) -> SelfDiscoverState:
        """Simplified execute node with better error handling."""
        print("\n⚙️ EXECUTING IMPROVED CODE")
        
        code = state.get("improved_code", "")
        dataset_folder = state.get("dataset_folder", "datasets/healthcare")
        
        if not code or len(code.strip()) < 50:
            print("No code to execute")
            state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            return state
        
        # Ensure basics are in place
        fixed_code = self._ensure_code_basics(code, dataset_folder)
        state["improved_code"] = fixed_code
        
        # Execute the code
        wrapped_code = f"```python\n{fixed_code}\n```"
        
        executor = LocalCommandLineCodeExecutor(timeout=60)
        code_executor_agent = ConversableAgent(
            "executor",
            llm_config=False,
            code_execution_config={"executor": executor}
        )
        
        try:
            execution_output = code_executor_agent.generate_reply(
                messages=[{"role": "user", "content": wrapped_code}]
            )
            
            state["execution_output"] = execution_output
            print(f"\nExecution output summary: {execution_output[:100]}...")
            
            if "error" in execution_output.lower() or "traceback" in execution_output.lower():
                print("Execution failed.")
                state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
            else:
                print("Execution succeeded!")
                state["consecutive_failures"] = 0
                
                # Process metrics
                if os.path.exists('metrics_self_discovery.yaml'):
                    with open('metrics_self_discovery.yaml', 'r') as f:
                        metrics = yaml.safe_load(f)
                        state["metrics"] = metrics
                        print(f"Metrics: {metrics}")
                
                # Add to history
                if "improvement_history" not in state:
                    state["improvement_history"] = []
                    
                state["improvement_history"].append({
                    "iteration": state.get("iteration_count", 0) + 1,
                    "metrics": state.get("metrics", {}),
                    "code": fixed_code,
                    "changes": state.get("changes", [])
                })
        except Exception as e:
            print(f"Error during execution: {e}")
            state["execution_output"] = f"Execution error: {str(e)}"
            state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
        
        # Increment iteration count
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        
        return state

    def _ensure_code_basics(self, code: str, dataset_folder: str) -> str:
        """Ensure code has the basic required elements."""
        imports = [
            "import pandas as pd", 
            "import numpy as np",
            "import yaml",
            "from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier",
            "from sklearn.metrics import accuracy_score"
        ]
        
        # Check imports
        for imp in imports:
            if imp not in code:
                code = imp + "\n" + code
        
        # Ensure dataset folder
        if f'dataset_folder = "{dataset_folder}"' not in code:
            code = f'dataset_folder = "{dataset_folder}"\n' + code
        
        # Ensure metrics initialization
        metrics_init = "model_new_score = {\n    'on_old_data': 0.0,\n    'on_new_data': 0.0\n}"
        if "model_new_score" not in code:
            code = code.replace("import yaml", f"import yaml\n\n# Initialize metrics\n{metrics_init}")
        
        # Ensure metrics saving
        metrics_save = "with open('metrics_self_discovery.yaml', 'w') as f:\n    yaml.dump({'model_new_score': model_new_score}, f)"
        if "metrics_self_discovery.yaml" not in code:
            code += f"\n\n# Save metrics\n{metrics_save}\n"
        
        return code
    
    def _fix_common_syntax_errors(self, code: str) -> str:
        """Fix common syntax errors in generated code."""
        # Split code into lines for processing
        lines = code.split('\n')
        fixed_lines = []
        
        # Track indentation level
        current_indent = 0
        in_function = False
        in_class = False
        
        for line in lines:
            stripped = line.lstrip()
            
            # Skip empty lines
            if not stripped:
                fixed_lines.append(line)
                continue
                
            # Calculate current line indentation
            indent = len(line) - len(stripped)
            
            # Check if we're starting a new block
            if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'with ', 'try:')) and stripped.endswith(':'):
                current_indent = indent
                in_function = stripped.startswith('def ')
                in_class = stripped.startswith('class ')
                fixed_lines.append(line)
                continue
                
            # Fix indentation for lines inside blocks
            if in_function or in_class:
                if stripped.startswith(('return ', 'yield ', 'break', 'continue', 'pass')):
                    # Ensure proper indentation for these statements
                    fixed_line = ' ' * (current_indent + 4) + stripped
                    fixed_lines.append(fixed_line)
                    continue
                    
            # Fix common dict definition issues
            if stripped.startswith('model_new_score = {') and not stripped.endswith('}'):
                # Fix multiline dict initialization
                fixed_line = ' ' * indent + stripped
                fixed_lines.append(fixed_line)
                # Ensure the next lines are properly indented
                current_indent = indent
                continue
                
            # Add line with original indentation
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)

    def _has_execution_errors(self, output: str) -> bool:
        """Check if execution output contains errors."""
        error_indicators = [
            'error', 'exception', 'failed', 'failure',
            'traceback', 'exitcode: 1'
        ]
        return any(indicator in output.lower() for indicator in error_indicators)
        
    def _extract_code(self, text: str) -> str:
        """Extract code from response text using a simpler approach."""
        # If the text is already just code, return it
        if text.strip().startswith(("import ", "from ")):
            return text
        
        # Look for code blocks
        import re
        code_blocks = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
        
        # Look for code blocks without language specification
        code_blocks = re.findall(r'```\n(.*?)\n```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
        
        # Fallback - just return the text
        return text
            
    def _validate_code(self, code: str) -> bool:
        """Validate that the code is syntactically valid Python."""
        try:
            # Try to compile the code to check for syntax errors
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError as e:
            print(f"⚠️ Syntax error in generated code: {str(e)}")
            return False

    
    def _extract_changes(self, text: str) -> List[str]:
        """Extract list of changes from response text."""
        changes_section = re.search(r'changes (I\'ve|made).*?:(.*?)(?:$|```)', text, re.DOTALL | re.IGNORECASE)
        
        if changes_section:
            changes_text = changes_section.group(2)
            changes = re.findall(r'[-*]\s*(.*?)(?:\n|$)', changes_text)
            return changes
        
        changes = re.findall(r'[-*]\s*(.*?)(?:\n|$)', text)
        changes = [c for c in changes if len(c) > 10 and any(keyword in c.lower() for keyword in 
                                                           ['add', 'improv', 'chang', 'updat', 'modif', 'implement', 'us'])]
        
        # If we couldn't extract changes, create some based on the code
        if not changes:
            code = self._extract_code(text)
            if "GradientBoosting" in code:
                changes.append("Implemented GradientBoosting for better handling of distribution shifts")
            if "StandardScaler" in code:
                changes.append("Added feature scaling using StandardScaler")
            if "concat" in code:
                changes.append("Combined old and new training data")
        
        return changes[:5]
    
    def _extract_metrics_from_output(self, output: str) -> Dict[str, float]:
        """Extract metrics from execution output as a fallback."""
        metrics = {}
        # Look for patterns like "Old distribution: 0.91" or "accuracy on old: 0.91"
        old_match = re.search(r'old (?:distribution|data).*?(\d+\.\d+)', output, re.IGNORECASE)
        new_match = re.search(r'new (?:distribution|data).*?(\d+\.\d+)', output, re.IGNORECASE)
        
        if old_match:
            metrics["on_old_data"] = float(old_match.group(1))
        if new_match:
            metrics["on_new_data"] = float(new_match.group(1))
            
        return metrics
    
    def _analyze_code_limitations(self, code: str) -> str:
        """Identify code limitations for adaptation."""
        limitations = []
        if "RandomForest" in code:
            limitations.append("Using basic RandomForest without adaptation")
        if "StandardScaler" not in code:
            limitations.append("No feature scaling/normalization")
        if "concat" not in code:
            limitations.append("Not combining datasets")
        if any(lib in code for lib in ["keras", "tensorflow", "torch"]):
            limitations.append("Using non-sklearn dependencies")
        return "\n".join(limitations) if limitations else "No major limitations identified"
    
    def should_continue(self, state: SelfDiscoverState) -> Literal["continue", "end"]:
        """Determine if improvement process should continue or end."""
        # Check if we've hit the max consecutive failures
        if state.get("consecutive_failures", 0) >= self.max_failures:
            print(f"\nReached maximum consecutive failures ({self.max_failures}). Ending process.")
            return "end"
            
        if state.get("iteration_count", 0) >= self.max_iterations:
            print(f"\nReached maximum iterations ({self.max_iterations}). Ending process.")
            return "end"
        
        if "execution_output" in state and "ModuleNotFoundError" in state["execution_output"]:
            print("\nCritical dependency error detected. Ending process.")
            return "end"
        
        current = state.get("metrics", {}).get('model_new_score', {})
        previous = state.get("previous_metrics", {}).get('model_new_score', {})
        
        if not previous:
            return "continue"
        
        improvement_new = current.get('on_new_data', 0) - previous.get('on_new_data', 0)
        improvement_old = current.get('on_old_data', 0) - previous.get('on_old_data', 0)
        
        print(f"\nImprovements this iteration:")
        print(f"New Distribution: {improvement_new:+.4f}")
        print(f"Old Distribution: {improvement_old:+.4f}")
        
        if improvement_new <= 0 and improvement_old <= 0:
            print("\nNo improvement detected on either distribution")
            return "end"
        
        return "continue"
    
    def _export_results_to_yaml(self, state: SelfDiscoverState, runtime: float) -> Dict:
        """Format results matching a standardized output format."""
        # Ensure we have valid metrics data for output
        if "model_old_score" not in state.get("metrics", {}):
            print("Warning: No model_old_score found in metrics")
            # Create default values if missing
            state["metrics"]["model_old_score"] = {
                "on_old_data": 0.0,
                "on_new_data": 0.0
            }
            
        if "model_new_score" not in state.get("metrics", {}):
            print("Warning: No model_new_score found in metrics")
            # Set model_new_score equal to model_old_score if missing
            state["metrics"]["model_new_score"] = state["metrics"]["model_old_score"]
        
        # Make a copy of improvement_history if it exists
        improvement_path = []
        if "improvement_history" in state and state["improvement_history"]:
            for entry in state["improvement_history"]:
                path_entry = {
                    "iteration": entry.get("iteration", 1),
                    "code": entry.get("code", state["improved_code"]),
                    "metrics": {
                        "old_distribution": entry.get("metrics", {}).get("model_new_score", {}).get("on_old_data", state["metrics"]["model_new_score"]["on_old_data"]),
                        "new_distribution": entry.get("metrics", {}).get("model_new_score", {}).get("on_new_data", state["metrics"]["model_new_score"]["on_new_data"])
                    },
                    "changes": entry.get("changes", state["changes"]),
                    "reflection": entry.get("reflection", ""),
                    "execution_time": entry.get("execution_time", 0)
                }
                improvement_path.append(path_entry)
        else:
            # Create a default improvement path if none exists
            improvement_path = [{
                "iteration": 1,
                "code": state["improved_code"],
                "metrics": {
                    "old_distribution": state["metrics"]["model_new_score"]["on_old_data"],
                    "new_distribution": state["metrics"]["model_new_score"]["on_new_data"]
                },
                "changes": state["changes"],
                "reflection": f"Selected Modules:\n{state.get('selected_modules', '')}\n\nAdapted Plan:\n{state.get('adapted_modules', '')}\n\nStructure:\n{state.get('reasoning_structure', '')}",
                "execution_time": state.get("iteration_times", [{}])[0].get("time", 0) if state.get("iteration_times") else 0
            }]
        
        # Get token usage from state or use estimation
        token_usage = state.get("token_usage", {
            "prompt": 0,
            "completion": 0,
            "total": len(state["improved_code"])//4
        })
        
        # Build the output following the standard structure
        return {
            "agent_name": "self_discovery",
            "initial_code": state["model_code"],
            "initial_metrics": {
                "old_distribution": state["metrics"]["model_old_score"]["on_old_data"],
                "new_distribution": state["metrics"]["model_old_score"]["on_new_data"]
            },
            "improvement_path": improvement_path,
            "final_code": state["improved_code"],
            "final_metrics": {
                "old_distribution": state["metrics"]["model_new_score"]["on_old_data"],
                "new_distribution": state["metrics"]["model_new_score"]["on_new_data"]
            },
            "runtime_statistics": {
                "total_time_seconds": runtime,
                "iterations": state["iteration_count"],
                "tokens_used": token_usage.get("total", 0),
                "prompt_tokens": token_usage.get("prompt", 0),
                "completion_tokens": token_usage.get("completion", 0),
                "iteration_times": state.get("iteration_times", []),
                "evaluation_timestamp": datetime.utcnow().isoformat()+"Z"
            }
        }
    
    def run(self, initial_state: Dict) -> Dict:
        """Run the self-discovery improvement process with enhanced error handling.
        
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
        typed_state = SelfDiscoverState(
            model_code=initial_state.get("model_code", ""),
            improved_code="",
            execution_output="",
            metrics=initial_metrics.copy() if initial_metrics else {},
            previous_metrics={},
            iteration_count=0,
            improvement_history=[],
            changes=[],
            reasoning_modules=REASONING_MODULES,
            selected_modules="",
            adapted_modules="",
            reasoning_structure="",
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
        
        print("\n🚀 Starting Self-Discovery Model Improvement Process")
        print(f"Dataset: {dataset_description.get('DATASET_TITLE', 'Unknown')}")
        print(f"Error handling: stopping after {self.max_failures} consecutive failures")
        
        if dataset_description.get('NUMERICAL_FEATURES') and dataset_description.get('CATEGORICAL_FEATURES'):
            print(f"Features: {len(dataset_description.get('NUMERICAL_FEATURES', [])) + len(dataset_description.get('CATEGORICAL_FEATURES', []))} total, {len(dataset_description.get('NUMERICAL_FEATURES', []))} numerical, {len(dataset_description.get('CATEGORICAL_FEATURES', []))} categorical")
        
        try:
            # Run the decision procedure
            config = {"recursion_limit": 50}  # Increase recursion limit
            final_output = None
            for output in self.graph.stream(typed_state, config=config):
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
            
            # Get the final state
            final_state = final_output[list(final_output.keys())[-1]]
            
            # Calculate total runtime
            end_time = time.time()
            runtime_seconds = end_time - self.start_time
            
            # Generate final report
            print("\n📊 Self-Discovery Improvement Process Complete")
            print(f"\nTotal runtime: {runtime_seconds:.2f} seconds")
            print(f"Execution attempts: successful={len(final_state.get('improvement_history', []))}, " + 
                  f"failed={final_state.get('consecutive_failures', 0)}")
            
            # Summary of iterations
            if final_state.get("iteration_times"):
                print("\nIteration Times:")
                for iter_time in final_state["iteration_times"]:
                    print(f"  Iteration {iter_time['iteration']}: {iter_time['time']:.2f} seconds")
            
            # Create standardized YAML output
            yaml_output = self._export_results_to_yaml(final_state, runtime_seconds)
            final_state["yaml_output"] = yaml_output
            
            return final_state
            
        except Exception as e:
            print(f"Error in graph execution: {str(e)}")
            import traceback
            traceback.print_exc()
            raise