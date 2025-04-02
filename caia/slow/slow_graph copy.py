from typing import TypedDict, Dict, List, Optional, Annotated, Any
import textwrap
import yaml
from time import sleep
from rich import print
from rich.panel import Panel
from rich.text import Text
import re

from langgraph.graph import StateGraph, END
from IPython.display import Image, display
from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor, DockerCommandLineCodeExecutor
from caia.memory import ModelScore, ImprovementEntry, create_improvement_entry

from caia.memory import WorkingMemory, SemanticMemory, EpisodicMemory 
from caia.utils import print_function_name
from caia.prompts import (
    prompt_distill_memories,
    prompt_generate_tiny_change,
    prompt_evaluate_change, prompt_summarize_model_docs,
    prompt_distill_memories,
    prompt_analyze_improvement_needs,  # New prompt
    prompt_model_selection_change,     # New prompt 
    prompt_hyperparameter_tuning,      # New prompt
    prompt_ensemble_method,            # New prompt
    prompt_parse_strategy_output,     # New prompt
    # prompt_summarize_monitoring_report, # New prompt
    prompt_stats_data,
    prompt_evaluate_change,
    # prompt_fix_code_slow,
    prompt_fix_code,
)

class SlowGraph:
    def __init__(self, llm, debug=False):
        """Initialize the slow improvement graph.
        
        Args:
            llm: The language model to use for generation
            debug: Whether to run in debug mode
        """
        self.llm = llm
        self.graph = EnhancedStateGraph(WorkingMemory)
        self.build_plan()
        self.decision_procedure = self.graph.compile(debug=debug)

    def draw_graph(self):
        """Visualize the graph structure."""
        return display(Image(self.decision_procedure.get_graph().draw_mermaid_png()))

    def build_plan(self):
        """Build the graph structure with nodes and edges."""
        # Add nodes for each step of the process
        self.graph.add_node('distill_memories', self.distill_memories)
        self.graph.add_node('analyze_needs', self.analyze_needs)
        
        # Strategy nodes
        self.graph.add_node('model_selection', self.generate_model_selection_change)
        self.graph.add_node('hyperparameter_tuning', self.generate_hyperparameter_tuning)
        self.graph.add_node('ensemble_method', self.generate_ensemble_method)
        
        # Execution and evaluation nodes
        self.graph.add_node('apply_change', self.apply_change)
        self.graph.add_node('evaluate_change', self.evaluate_change)
        
        # Set entry point
        self.graph.set_entry_point('distill_memories')
        
        # Add edges for the main flow
        self.graph.add_edge('distill_memories', 'analyze_needs')
        
        # Add conditional edge to route to appropriate strategy
        self.graph.add_conditional_edges(
            'analyze_needs',
            self.route_to_strategy,
            {
                'model_selection': 'model_selection',
                'hyperparameter_tuning': 'hyperparameter_tuning',
                'ensemble_method': 'ensemble_method',
                'end': END  # Add direct end path if no strategies available
            }
        )
        
        # Connect all strategy nodes to apply_change
        strategy_nodes = ['model_selection', 'hyperparameter_tuning', 'ensemble_method']
        for node in strategy_nodes:
            self.graph.add_edge(node, 'apply_change')
        
        # Add conditional edge for apply_change
        self.graph.add_conditional_edges(
            'apply_change',
            self.should_evaluate_code,
            {
                'evaluate': 'evaluate_change',
                'retry': 'analyze_needs',  # Changed: failed executions go back to analyze_needs
                'end': END
            }
        )
        # Add conditional edge for evaluate_change
        self.graph.add_conditional_edges(
            'evaluate_change',
            self.should_continue_improving,
            {
                'continue': 'analyze_needs',
                'end': END
            }
        )



    def initialize_generations(self) -> Dict[str, Any]:
        """Initialize the generations dictionary with strategy tracking."""
        return {
            # Core insights and changes
            'distilled_insights': '',
            'tiny_change': '',
            
            # Execution tracking
            'execution_output': '',
            'execution_success': False,
            
            # Strategy tracking
            'current_strategy': '',
            'strategy_analysis': {
                'recommended_strategy': '',
                'reasoning': '',
                'tried_strategies': [],
                'performance_gaps': [],
                'next_steps': []
            },
            'strategy_results': {
                'model_selection': {
                    'tried': False,
                    'models_tried': []
                },
                'hyperparameter_tuning': {
                    'tried': False,
                    'best_params': {}
                },
                'ensemble_method': {
                    'tried': False,
                    'best_ensemble': ''
                }
            }
        }
    
    def stats_data(self, state: WorkingMemory) -> dict:
        """Generate and execute dataset statistics code"""
        semantic_memory = state['semantic_memory']
        episodic_memory = state['episodic_memory'][-1]
        
        input_content = {
            "old": {
                "description": semantic_memory.dataset_old.description,
                "X_train": semantic_memory.dataset_old.X_train,
                "y_train": semantic_memory.dataset_old.y_train
            },
            "new": {
                "description": episodic_memory.new_dataset.description,
                "X_train": episodic_memory.new_dataset.X_train,
                "y_train": episodic_memory.new_dataset.y_train
            }
        }
        
        # Generate analysis code
        prompt = prompt_stats_data()
        chain = prompt | self.llm
        code_output = chain.invoke({'input': yaml.dump(input_content)}).content
        
        # Parse the YAML output to extract the Python code
        try:
            parsed_yaml = yaml.safe_load(code_output)
            stats_code = parsed_yaml['stats_code']
        except Exception as e:
            print(f"Error parsing YAML output: {str(e)}")
            return {}
        
        # Execute the extracted Python code
        executor = LocalCommandLineCodeExecutor(timeout=20)
        code_executor_agent = ConversableAgent(
            "stats_agent",
            llm_config=False,
            code_execution_config={"executor": executor}
        )
        
        
        # Execute the generated code
        execution_output = code_executor_agent.generate_reply(
            messages=[{"role": "user", "content": f"```python\n{stats_code}\n```"}]
        )
        
        print(execution_output)

        # Read the YAML file from disk
        try:
            with open('dataset_stats.yaml', 'r') as f:
                results = yaml.safe_load(f)
            return results
        except Exception as e:
            print(f"Error reading stats YAML file: {str(e)}")
            return {}

    def distill_memories(self, state: WorkingMemory) -> WorkingMemory:
        """Updated distill_memories to focus on core model performance analysis"""
        state['generations_slow_graph'] = self.initialize_generations()
        
        # Prepare semantic memory inputs
        semantic_memory = state['semantic_memory']
        model_params_summary = (prompt_summarize_model_docs() | self.llm).invoke(
            {'input': semantic_memory.model_object.__doc__}
        ).content
        
        # Build focused input YAML with only required components
        yaml_content = {
            'execution_output': state['episodic_memory'][-1].quick_insight['execution_output'],
            'model_code': semantic_memory.model_code
        }
        
        # Generate insights with core data
        chain = prompt_distill_memories() | self.llm
        output = chain.invoke({'input': yaml.dump(yaml_content)}).content
        
        # Store structured results
        try:
            insights = yaml.safe_load(output)
        except yaml.YAMLError:
            insights = {'error': 'Failed to parse insights YAML'}
        
        state['generations_slow_graph'].update({
            'distilled_insights': insights,
            'model_metadata': {
                'params_summary': model_params_summary,
                'data_paths': {
                    'old_data': semantic_memory.dataset_old.X_train,
                    'new_data': state['episodic_memory'][-1].dataset_new.X_train
                }
            }
        })
        
        return state



    def analyze_needs(self, state: WorkingMemory) -> WorkingMemory:
        """Analyzes current model performance and determines next best strategy."""
        
        state['generations_slow_graph']['execution_attempts'] = 0
        # Get current state information
        execution_output = state['generations_slow_graph'].get('execution_output', '')
        strategy_results = state['generations_slow_graph'].get('strategy_results', {})
        improvement_history = state.get('improvement_history', [])
        
        # Get current performance metrics
        latest_metrics = None
        if improvement_history:
            latest_eval = improvement_history[-1].get('evaluation', {}).get('evaluation', {})
            latest_metrics = latest_eval.get('metrics', {})
        
        # Prepare input YAML
        yaml_content = {
            "current_performance": execution_output,
            "improvement_history": improvement_history,
            "strategy_results": strategy_results,
        }
        
        # Add metrics if available
        if latest_metrics:
            yaml_content["latest_metrics"] = latest_metrics
        
        # Get strategy recommendation
        prompt = prompt_analyze_improvement_needs()
        chain = prompt | self.llm
        analysis = chain.invoke({'input': yaml.dump(yaml_content)}).content
        
        try:
            # Try to parse as YAML first
            analysis_dict = yaml.safe_load(analysis)
            if analysis_dict and isinstance(analysis_dict, dict):
                parsed_analysis = analysis_dict
            else:
                raise yaml.YAMLError("Invalid YAML structure")
                
        except yaml.YAMLError:
            # If YAML parsing fails, use the parsing prompt
            parse_prompt = prompt_parse_strategy_output()
            parse_chain = parse_prompt | self.llm
            parsed_output = parse_chain.invoke({'input': analysis}).content
            
            try:
                parsed_analysis = yaml.safe_load(parsed_output)
            except yaml.YAMLError:
                parsed_analysis = self._extract_strategy_heuristic(analysis)
        
        # Get suggested strategy from next_steps if available
        next_steps = parsed_analysis.get('next_steps', [])
        recommended_strategy = None
        
        strategy_keywords = {
            'model_selection': ['different model', 'new model', 'model architecture'],
            'hyperparameter_tuning': ['parameter', 'tune', 'hyperparameter'],
            'ensemble_method': ['ensemble', 'combine', 'voting']
        }
        
        # Look for strategy keywords in next steps
        for step in next_steps:
            for strategy, keywords in strategy_keywords.items():
                if any(keyword in step.lower() for keyword in keywords):
                    if not strategy_results.get(strategy, {}).get('tried', False):
                        recommended_strategy = strategy
                        break
            if recommended_strategy:
                break
        
        # If no untried strategy found in next steps, pick first untried strategy
        if not recommended_strategy:
            for strategy in ['model_selection', 'hyperparameter_tuning', 'ensemble_method']:
                if not strategy_results.get(strategy, {}).get('tried', False):
                    recommended_strategy = strategy
                    break
        
        # Default to model_selection if all else fails
        if not recommended_strategy:
            recommended_strategy = 'model_selection'
        
        # Update state
        state['generations_slow_graph'].update({
            'current_strategy': recommended_strategy,
            'strategy_analysis': parsed_analysis
        })
        
        print("\nStrategy Analysis:", "-"*50)
        print(f"Recommended Strategy: {recommended_strategy}")
        print(f"Next Steps: {next_steps}")
        print(f"Strategies Tried: {[s for s, r in strategy_results.items() if r.get('tried', False)]}")
        
        return state
    
    def generate_model_selection_change(self, state: WorkingMemory) -> WorkingMemory:
        """Generates changes focused on trying different model architectures."""
        try:
            # Get current strategy results
            strategy_results = state['generations_slow_graph'].get('strategy_results', {})
            models_tried = strategy_results.get('model_selection', {}).get('models_tried', [])
            
            # Get previous metrics from improvement history
            prev_metrics = None
            if state.get('improvement_history'):
                prev_metrics = state['improvement_history'][-1]['metrics']
            
            prompt = prompt_model_selection_change()
            chain = prompt | self.llm
            
            # Prepare input YAML
            yaml_content = {
                "current_code": state['semantic_memory'].model_code,
                "execution_output": state['generations_slow_graph'].get('execution_output', ''),
                "models_tried": models_tried,
                "previous_performance": prev_metrics if prev_metrics else {}
            }
            
            # Get model selection changes
            change_output = chain.invoke({'input': yaml.dump(yaml_content)}).content
            
            # Store the raw output
            state['generations_slow_graph']['tiny_change'] = change_output
            
            # Parse YAML output for strategy results update
            try:
                change_dict = yaml.safe_load(change_output)
                if isinstance(change_dict, dict):
                    # Update strategy results with new model information
                    new_generations = state['generations_slow_graph'].copy()
                    new_generations['strategy_results']['model_selection'] = {
                        'tried': True,
                        'models_tried': models_tried + [change_dict.get('model_name', 'unknown')],
                        'latest_changes': {
                            'model_name': change_dict.get('model_name', ''),
                            'parameters': change_dict.get('parameters', {}),
                            'rationale': change_dict.get('rationale', '')
                        }
                    }
                    state['generations_slow_graph'].update(new_generations)
                    
            except yaml.YAMLError as e:
                print(f"Error parsing model selection output: {str(e)}")
                
        except Exception as e:
            print(f"Error in generate_model_selection_change: {str(e)}")
            state['generations_slow_graph']['error'] = str(e)
        
        return state


    def generate_hyperparameter_tuning(self, state: WorkingMemory) -> WorkingMemory:
        """Generates hyperparameter optimization changes."""
        try:
            # Get current strategy results
            strategy_results = state['generations_slow_graph'].get('strategy_results', {})
            current_params = strategy_results.get('hyperparameter_tuning', {}).get('best_params', {})
            
            # Get previous metrics from improvement history
            prev_metrics = None
            if state.get('improvement_history'):
                prev_metrics = state['improvement_history'][-1]['metrics']
            
            prompt = prompt_hyperparameter_tuning()
            chain = prompt | self.llm
            
            # Prepare input YAML
            yaml_content = {
                "current_code": state['semantic_memory'].model_code,
                "execution_output": state['generations_slow_graph'].get('execution_output', ''),
                "current_params": current_params,
                "previous_performance": prev_metrics if prev_metrics else {}
            }
            
            # Get hyperparameter tuning changes
            change_output = chain.invoke({'input': yaml.dump(yaml_content)}).content
            
            # Store the raw output
            state['generations_slow_graph']['tiny_change'] = change_output
            
            try:
                # Parse YAML output
                change_dict = yaml.safe_load(change_output)
                
                if isinstance(change_dict, dict):
                    new_generations = state['generations_slow_graph'].copy()
                    new_generations['strategy_results']['hyperparameter_tuning'] = {
                        'tried': True,
                        'best_params': change_dict.get('hyperparameters', {}),
                        'latest_changes': {
                            'parameters': change_dict.get('hyperparameters', {}),
                            'rationale': change_dict.get('rationale', '')
                        }
                    }
                    state['generations_slow_graph'].update(new_generations)
                    
            except yaml.YAMLError as e:
                print(f"Error parsing hyperparameter tuning output: {str(e)}")
                
        except Exception as e:
            print(f"Error in generate_hyperparameter_tuning: {str(e)}")
            state['generations_slow_graph']['error'] = str(e)
        
        return state


    def generate_ensemble_method(self, state: WorkingMemory) -> WorkingMemory:
        """Generates ensemble-based improvements."""
        try:
            # Get current strategy results
            strategy_results = state['generations_slow_graph'].get('strategy_results', {})
            
            # Get previous metrics from improvement history
            prev_metrics = None
            if state.get('improvement_history'):
                prev_metrics = state['improvement_history'][-1]['metrics']
            
            prompt = prompt_ensemble_method()
            chain = prompt | self.llm
            
            # Prepare input YAML
            yaml_content = {
                "current_code": state['semantic_memory'].model_code,
                "execution_output": state['generations_slow_graph'].get('execution_output', ''),
                "strategy_results": strategy_results,
                "previous_performance": prev_metrics if prev_metrics else {}
            }
            
            # Get ensemble method changes
            change_output = chain.invoke({'input': yaml.dump(yaml_content)}).content
            
            # Store the raw output
            state['generations_slow_graph']['tiny_change'] = change_output
            
            try:
                # Parse YAML output
                change_dict = yaml.safe_load(change_output)
                
                if isinstance(change_dict, dict):
                    new_generations = state['generations_slow_graph'].copy()
                    new_generations['strategy_results']['ensemble_method'] = {
                        'tried': True,
                        'best_ensemble': change_dict.get('ensemble_type', 'unknown'),
                        'latest_changes': {
                            'ensemble_type': change_dict.get('ensemble_type', ''),
                            'estimators': change_dict.get('estimators', []),
                            'rationale': change_dict.get('rationale', '')
                        }
                    }
                    state['generations_slow_graph'].update(new_generations)
                    
            except yaml.YAMLError as e:
                print(f"Error parsing ensemble method output: {str(e)}")
                
        except Exception as e:
            print(f"Error in generate_ensemble_method: {str(e)}")
            state['generations_slow_graph']['error'] = str(e)
        
        return state


    def route_to_strategy(self, state: WorkingMemory) -> str:
        """Routes to appropriate strategy node based on analysis.
        
        Args:
            state: Current working memory state
            
        Returns:
            Name of the next strategy to try
        """
        return state['generations_slow_graph'].get('current_strategy', 'model_selection')

    def should_evaluate_code(self, state: WorkingMemory) -> str:
        """Determine if code should be evaluated or retried.
        
        Args:
            state: Current working memory state
            
        Returns:
            'evaluate' if code executed successfully,
            'retry' if should retry with different strategy,
            'end' if should stop attempts
        """
        # Get current attempt count
        attempts = state['generations_slow_graph'].get('execution_attempts', 0)
        state['generations_slow_graph']['execution_attempts'] = attempts + 1
        
        # Check if execution was successful
        if state['generations_slow_graph'].get('execution_success', False):
            return 'evaluate'
        
        # If too many attempts, end the process
        if attempts >= 3:  # Limit retries
            print("Maximum execution attempts reached. Ending process.")
            return 'end'
        
        # Otherwise, retry with a different strategy
        print(f"Execution failed. Attempt {attempts + 1}/3. Retrying with different strategy.")
        return 'retry'

    def should_continue_improving(self, state: WorkingMemory) -> str:
        """Determine if improvement process should continue."""
        print("\nEvaluating improvement continuation...", "-"*50)
        
        # Get improvement history
        improvement_history = state.get('improvement_history', [])
        if not improvement_history:
            print("No improvement history found, continuing...")
            return 'continue'
        
        # Get latest improvement info
        latest_improvement = improvement_history[-1]
        
        # Get strategy information
        generations = state['generations_slow_graph']
        strategy_results = generations.get('strategy_results', {})
        current_strategy = generations.get('current_strategy')
        
        strategies_tried = [
            strategy for strategy, result in strategy_results.items() 
            if result.get('tried', False)
        ]
        
        # Get evaluation details from latest improvement
        evaluation = latest_improvement.get('evaluation', {})
        recommendation = evaluation.get('recommendation', {})
        next_steps = evaluation.get('next_steps', [])
        
        # Get performance metrics
        improvements = latest_improvement.get('improvements', {})
        metrics = latest_improvement.get('metrics', {})
        
        # Log decision factors
        self._log_improvement_decision_factors(
            strategies_tried=strategies_tried,
            latest_metrics=metrics,
            improvements=improvements,
            recommendation=recommendation
        )
        
        # Increment iteration count
        generations['iteration_count'] = generations.get('iteration_count', 0) + 1
        
        # Check basic stopping conditions
        max_iterations_reached = generations.get('iteration_count', 0) >= 10
        all_strategies_tried = len(strategies_tried) >= 3
        no_improvement = self._check_no_improvement(improvement_history)
        current_strategy_exhausted = self._is_strategy_exhausted(
            strategy_results.get(current_strategy, {}),
            latest_improvement
        )
        
        # Check if we've reached hard limits
        if max_iterations_reached:
            print("Maximum iterations reached.")
            return 'end'
        
        if all_strategies_tried and no_improvement:
            print("All strategies tried without significant improvement.")
            return 'end'
        
        # Handle successful improvements
        if latest_improvement['outcome'] == 'success':
            if current_strategy_exhausted:
                if not next_steps or all_strategies_tried:
                    print("Strategy exhausted and no more steps needed.")
                    return 'end'
                print("Strategy exhausted but other strategies available.")
                return 'continue'
            
            # Check if current strategy should continue
            significant_improvement = self._has_significant_improvement(latest_improvement)
            if significant_improvement and not current_strategy_exhausted:
                print("Continuing with current successful strategy.")
                return 'continue'
        
        # Check if we should try other strategies
        if len(strategies_tried) < 3:
            print("Still have untried strategies available.")
            return 'continue'
        
        print("No clear improvement path found.")
        return 'end'

    def _check_no_improvement(self, improvement_history: List[Dict]) -> bool:
        """Check if recent improvements show no progress."""
        if len(improvement_history) < 2:
            return False
        
        # Look at last 3 improvements or all if less
        recent_improvements = improvement_history[-3:]
        threshold = 0.01  # 1% improvement threshold
        
        # Check both distributions for improvements
        for imp in recent_improvements:
            improvements = imp.get('improvements', {})
            new_dist_improvement = abs(improvements.get('new_distribution', 0))
            old_dist_improvement = abs(improvements.get('old_distribution', 0))
            
            # Consider improvement if either distribution shows progress
            if new_dist_improvement > threshold or old_dist_improvement > threshold:
                return False
        
        return True

    def _is_strategy_exhausted(self, strategy_results: Dict, latest_improvement: Dict) -> bool:
        """Determine if current strategy should be considered exhausted."""
        if not strategy_results:
            return False
            
        strategy_type = latest_improvement.get('strategy_type')
        
        if strategy_type == 'model_selection':
            # Consider exhausted if we've tried more than 3 models
            return len(strategy_results.get('models_tried', [])) >= 3
            
        elif strategy_type == 'hyperparameter_tuning':
            # Consider exhausted after 3 attempts without significant improvement
            return strategy_results.get('attempts', 0) >= 3
            
        elif strategy_type == 'ensemble_method':
            # Consider exhausted if we've tried the main ensemble types
            return strategy_results.get('tried', False)
            
        return False

    def _has_significant_improvement(self, improvement: Dict) -> bool:
        """Check if the improvement is significant enough to continue with current strategy."""
        threshold = 0.02  # 2% improvement threshold
        improvements = improvement.get('improvements', {})
        
        new_dist_improvement = improvements.get('new_distribution', 0)
        old_dist_improvement = improvements.get('old_distribution', 0)
        
        # Consider improvement significant if either distribution improves notably
        # without severely degrading the other
        if new_dist_improvement > threshold and old_dist_improvement > -threshold:
            return True
        if old_dist_improvement > threshold and new_dist_improvement > -threshold:
            return True
            
        return False

    def _log_improvement_decision_factors(
        self,
        strategies_tried: List[str],
        latest_metrics: Dict,
        improvements: Dict,
        recommendation: Dict
    ):
        """Log factors influencing the improvement decision."""
        print("\nImprovement Decision Factors:", "-"*50)
        print(f"Strategies Tried: {', '.join(strategies_tried)}")
        print("\nLatest Performance:")
        print(f"  Old Distribution: {latest_metrics.get('new_model', {}).get('on_old_data', 0):.4f}")
        print(f"  New Distribution: {latest_metrics.get('new_model', {}).get('on_new_data', 0):.4f}")
        print("\nImprovements:")
        print(f"  Old Distribution: {improvements.get('old_distribution', 0):.4f}")
        print(f"  New Distribution: {improvements.get('new_distribution', 0):.4f}")
        print(f"\nRecommendation: {recommendation.get('action', 'unknown')}")
        print(f"Confidence: {recommendation.get('confidence', 'unknown')}")


    def apply_change(self, state: WorkingMemory) -> WorkingMemory:
        """Apply the generated change and execute the code."""
        try:
            # Extract and parse code from tiny_change
            current_code = self._extract_code(state['generations_slow_graph']['tiny_change'])
            max_retries = 4
            current_try = 0
            
            while current_try < max_retries:
                # Execute the code
                execution_output = self._execute_code(current_code)
                
                # Check for execution errors
                if not self._has_execution_errors(execution_output):
                    break
                    
                # Handle code fixing if needed
                if current_try < max_retries - 1:
                    current_code = self._fix_code(current_code, execution_output, state)
                
                current_try += 1
            
            # Process execution results and update improvement history
            state = self._process_execution_results(state, execution_output, current_code)
            
        except Exception as e:
            print(f"Error in apply_change: {str(e)}")
            state['generations_slow_graph'].update({
                'execution_success': False,
                'execution_output': str(e)
            })
        
        return state

    def _extract_code(self, code_yaml: str) -> str:
        """Extract code from YAML structure."""
        try:
            parsed_yaml = yaml.safe_load(code_yaml)
            if isinstance(parsed_yaml, dict) and 'new_training_code' in parsed_yaml:
                return parsed_yaml['new_training_code']
            return code_yaml
        except yaml.YAMLError:
            return code_yaml

    def _execute_code(self, code: str) -> str:
        """Execute code using the code executor."""
        wrapped_code = f"```python\n{code}\n```"
        executor = LocalCommandLineCodeExecutor(
            timeout=10,
            work_dir=".",
        )
        code_executor_agent = ConversableAgent(
            "code_executor_agent",
            llm_config=False,
            code_execution_config={"executor": executor},
            human_input_mode="NEVER",
        )
        return code_executor_agent.generate_reply(
            messages=[{"role": "user", "content": wrapped_code}]
        )

    def _has_execution_errors(self, output: str) -> bool:
        """Check if execution output contains errors."""
        error_indicators = ['error', 'failed', 'TypeError', 'ValueError', 'NameError']
        return any(err in output.lower() for err in error_indicators)

    def _fix_code(self, code: str, error_output: str, state: WorkingMemory) -> str:
        """Fix code using the fix code prompt."""
        yaml_content = {
            "current_code": code,
            "error_output": error_output
        }
        
        prompt = prompt_fix_code()
        chain = prompt | self.llm
        fixed_output = chain.invoke({'input': yaml.dump(yaml_content)}).content
        
        try:
            fixed_dict = yaml.safe_load(fixed_output)
            if isinstance(fixed_dict, dict) and 'fixed_code' in fixed_dict:
                if 'validation_steps' in fixed_dict:
                    state['generations_slow_graph']['validation_steps'] = fixed_dict['validation_steps']
                return fixed_dict['fixed_code']
        except yaml.YAMLError:
            print("Failed to parse fixed code output")
        
        return code

    def _process_execution_results(self, state: WorkingMemory, execution_output: str, current_code: str) -> WorkingMemory:
        """Process execution results and update improvement history."""
        print("Execution Output:", "-"*100)
        print(execution_output)
        
        # Store raw output
        state['generations_slow_graph']['execution_output'] = execution_output
        
        # Extract metrics from both YAML files
        try:
            # Read old model metrics
            with open('old_metrics.yaml', 'r') as f:
                old_metrics = yaml.safe_load(f)
                old_model_score = old_metrics.get('model_old_score', {})
            
            # Read new model metrics
            with open('slow_graph_metrics.yaml', 'r') as f:
                new_metrics = yaml.safe_load(f)
                new_model_score = new_metrics.get('model_new_score', {})
                
            # Get current strategy and changes information
            current_strategy = state['generations_slow_graph'].get('current_strategy')
            strategy_results = state['generations_slow_graph'].get('strategy_results', {})
            
            # Prepare changes made dictionary based on strategy
            changes_made = {
                'strategy': current_strategy,
                'iteration_count': state['generations_slow_graph'].get('iteration_count', 0)
            }
            
            # Add strategy-specific changes
            if current_strategy == 'model_selection':
                changes_made.update({'models_tried': strategy_results.get('model_selection', {}).get('models_tried', [])})
            elif current_strategy == 'hyperparameter_tuning':
                changes_made.update({'parameters': strategy_results.get('hyperparameter_tuning', {}).get('best_params', {})})
            elif current_strategy == 'ensemble_method':
                changes_made.update({'ensemble_type': strategy_results.get('ensemble_method', {}).get('best_ensemble', '')})

            # Create improvement entry
            improvement_entry = create_improvement_entry(
                previous_code=state['semantic_memory'].model_code,
                new_code=current_code,
                graph_type='slow',
                strategy_type=current_strategy,
                old_model_score=old_model_score,
                new_model_score=new_model_score,
                changes_made=changes_made
            )

            # Initialize improvement_history if it doesn't exist
            if 'improvement_history' not in state:
                state['improvement_history'] = []
            
            # Add the new improvement entry
            state['improvement_history'].append(improvement_entry)
                
            # Update state with metrics and success status
            state['generations_slow_graph'].update({
                'execution_success': True,
                'model_new_score': new_model_score,
                'model_old_score': old_model_score
            })
            
        except (yaml.YAMLError, FileNotFoundError) as e:
            print(f"Error processing metrics: {str(e)}")
            state['generations_slow_graph']['execution_success'] = False
        
        return state


    def evaluate_change(self, state: WorkingMemory) -> WorkingMemory:
        """Evaluate model changes and update improvement history."""
        print("\nEvaluating model changes...", "-"*50)
        
        # Get latest improvement entry
        if not state.get('improvement_history'):
            print("No improvement history found. Cannot evaluate changes.")
            return state
            
        latest_improvement = state['improvement_history'][-1]
        
        # Get metrics from the latest improvement
        current_metrics = latest_improvement['metrics']['new_model']
        previous_metrics = latest_improvement['metrics']['old_model']
        
        # Get code versions
        current_code = latest_improvement['new_code']
        previous_code = latest_improvement['previous_code']
        
        # Prepare evaluation input
        yaml_content = {
            'current_code': current_code,
            'execution_output': state['generations_slow_graph']['execution_output'],
            'current_metrics': current_metrics,
            'previous_metrics': previous_metrics,
            'strategy_type': latest_improvement['strategy_type'],
            'improvements': latest_improvement['improvements']
        }
        
        # Get evaluation from LLM
        evaluation_result = self._get_evaluation(yaml_content)
        
        # Log evaluation metrics
        self._log_evaluation_metrics(current_metrics, previous_metrics)
        
        # Update improvement entry with evaluation results, with proper error handling
        try:
            recommendation = evaluation_result.get('recommendation', {})
            action = recommendation.get('action', 'reject')  # Default to reject if not found
            
            updated_improvement = {
                **latest_improvement,
                'evaluation': evaluation_result,
                'final_outcome': action
            }
            
            # Replace the latest improvement with updated version
            state['improvement_history'][-1] = updated_improvement
            
            # Update strategy results
            current_strategy = state['generations_slow_graph'].get('current_strategy')
            if current_strategy:
                strategy_results = state['generations_slow_graph'].get('strategy_results', {})
                if current_strategy in strategy_results:
                    strategy_results[current_strategy]['tried'] = True
                    
                    # Update strategy-specific information
                    if current_strategy == 'model_selection':
                        model_name = latest_improvement['changes_made'].get('model_name', '')
                        if model_name:
                            models_tried = strategy_results[current_strategy].get('models_tried', [])
                            if model_name not in models_tried:
                                models_tried.append(model_name)
                            strategy_results[current_strategy]['models_tried'] = models_tried
                            
                    elif current_strategy == 'hyperparameter_tuning':
                        best_params = latest_improvement['changes_made'].get('parameters', {})
                        if best_params:
                            strategy_results[current_strategy]['best_params'] = best_params
                            
                    elif current_strategy == 'ensemble_method':
                        ensemble_type = latest_improvement['changes_made'].get('ensemble_type', '')
                        if ensemble_type:
                            strategy_results[current_strategy]['best_ensemble'] = ensemble_type
                    
                    state['generations_slow_graph']['strategy_results'] = strategy_results
            
            # Store evaluation in state
            state['generations_slow_graph']['evaluation'] = evaluation_result
            
        except Exception as e:
            print(f"Error updating improvement entry: {str(e)}")
            # Create a minimal evaluation result if there's an error
            state['generations_slow_graph']['evaluation'] = {
                'recommendation': {'action': 'reject', 'confidence': 'low'},
                'analysis': [f'Error in evaluation: {str(e)}'],
                'next_steps': ['Retry with different approach']
            }
        
        return state


    def _extract_current_metrics(self, state: WorkingMemory) -> Dict[str, float]:
        """Extract current model metrics."""
        try:
            model_new_score = state['generations_slow_graph'].get('model_new_score', {})
            return {
                'on_old_data': float(model_new_score.get('on_old_data', 0)),
                'on_new_data': float(model_new_score.get('on_new_data', 0))
            }
        except (TypeError, ValueError):
            return {'on_old_data': 0.0, 'on_new_data': 0.0}

    def _extract_previous_metrics(self, state: WorkingMemory) -> Dict[str, float]:
        """Extract previous model metrics."""
        if state['improvement_history']:
            last_entry = state['improvement_history'][-1]
            return {
                'on_old_data': float(last_entry.get('metrics_change', {}).get('old_distribution', 0)),
                'on_new_data': float(last_entry.get('metrics_change', {}).get('new_distribution', 0))
            }
        return {'on_old_data': 0.0, 'on_new_data': 0.0}


    def _get_previous_code(self, state: WorkingMemory) -> str:
        """Get previous code version."""
        if not state['improvement_history']:
            return state['semantic_memory'].model_code
        return state['improvement_history'][-1].get('new_code', '')


    def _get_evaluation(self, yaml_content: Dict) -> Dict:
        """Get evaluation from LLM with better error handling."""
        try:
            prompt = prompt_evaluate_change()
            chain = prompt | self.llm
            evaluation_output = chain.invoke({'input': yaml.dump(yaml_content)}).content
            
            result = yaml.safe_load(evaluation_output)
            
            # Ensure the result has the expected structure
            if not isinstance(result, dict):
                raise ValueError("Evaluation result is not a dictionary")
                
            # Ensure minimum required structure
            if 'recommendation' not in result:
                result['recommendation'] = {'action': 'reject', 'confidence': 'low'}
            if 'analysis' not in result:
                result['analysis'] = ['No analysis provided']
            if 'next_steps' not in result:
                result['next_steps'] = ['Retry with different approach']
                
            return result
            
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            # Return a default evaluation result
            return {
                'recommendation': {'action': 'reject', 'confidence': 'low'},
                'analysis': [f'Error in evaluation: {str(e)}'],
                'next_steps': ['Retry with different approach']
            }
        

    def _log_evaluation_metrics(self, current_metrics: Dict[str, float], previous_metrics: Dict[str, float]):
        """Log evaluation metrics for debugging."""
        print("\nEvaluation Metrics:", "-"*50)
        print("Current Performance:")
        print(f"  Old Distribution: {current_metrics['on_old_data']:.4f}")
        print(f"  New Distribution: {current_metrics['on_new_data']:.4f}")
        print("\nPrevious Performance:")
        print(f"  Old Distribution: {previous_metrics['on_old_data']:.4f}")
        print(f"  New Distribution: {previous_metrics['on_new_data']:.4f}")
        print("\nImprovements:")
        print(f"  Old Distribution: {current_metrics['on_old_data'] - previous_metrics['on_old_data']:.4f}")
        print(f"  New Distribution: {current_metrics['on_new_data'] - previous_metrics['on_new_data']:.4f}")

    def _update_state_with_evaluation(self, state: WorkingMemory, 
                                    improvement_entry: Dict,
                                    current_metrics: Dict) -> WorkingMemory:
        """Update state with evaluation results."""
        if 'improvement_history' not in state:
            state['improvement_history'] = []
        state['improvement_history'].append(improvement_entry)
        
        # Update state tracking
        state['generations_slow_graph']['evaluation'] = improvement_entry['evaluation']
        
        # Update strategy results
        current_strategy = state['generations_slow_graph'].get('current_strategy')
        if current_strategy:
            strategy_results = state['generations_slow_graph'].get('strategy_results', {})
            if current_strategy in strategy_results:
                strategy_results[current_strategy].update({
                    'tried': True,
                    'best_accuracy': {
                        'on_old_data': max(
                            strategy_results[current_strategy].get('best_accuracy', {}).get('on_old_data', 0),
                            current_metrics['on_old_data']
                        ),
                        'on_new_data': max(
                            strategy_results[current_strategy].get('best_accuracy', {}).get('on_new_data', 0),
                            current_metrics['on_new_data']
                        )
                    }
                })
                state['generations_slow_graph']['strategy_results'] = strategy_results
        
        # Log strategy status
        print("\nStrategy Status:", "-"*50)
        print(f"Current Strategy: {current_strategy}")
        print("Strategy Results:", state['generations_slow_graph']['strategy_results'])
        
        return state

    def should_evaluate_code(self, state: WorkingMemory) -> str:
        """Determine if code should be evaluated or retried."""
        return 'evaluate' if state['generations_slow_graph'].get('execution_success', False) else 'retry'


    def run(self, initial_state: WorkingMemory):
        """Run the slow improvement process with enhanced logging."""
        MAX_ITERATIONS = 20
        
        if 'generations_slow_graph' not in initial_state:
            initial_state['generations_slow_graph'] = self.initialize_generations()
            
        output_keys = ['generations_slow_graph', 'improvement_history']
        visited_states = set()
        iteration_count = 0
        
        try:
            for output in self.decision_procedure.stream(
                initial_state, 
                output_keys=output_keys, 
                debug=False
            ):
                iteration_count += 1
                if iteration_count >= MAX_ITERATIONS:
                    print(f"Reached maximum iterations limit: {MAX_ITERATIONS}")
                    break
                    
                for node_name, state in output.items():
                    # Create a more unique state key
                    current_strategy = state['generations_slow_graph'].get('current_strategy', '')
                    history_len = len(state.get('improvement_history', []))
                    strategies_tried = tuple(sorted(
                        [s for s, r in state['generations_slow_graph'].get('strategy_results', {}).items() 
                        if r.get('tried', False)]
                    ))
                    
                    state_key = (node_name, current_strategy, history_len, strategies_tried)
                    
                    if state_key in visited_states:
                        print(f"Skipping repeated state: {state_key}")
                        continue
                        
                    visited_states.add(state_key)
                
                    
                    # Log node execution
                    print(f"\nExecuting Node: {node_name}", "="*50)
                    
                    # Log generations updates
                    self._log_generations_updates(state)
                    
                    # Log improvement history updates
                    self._log_improvement_history(state)
                    
                    # Log strategy progress
                    self._log_strategy_progress(state)
                    
        except Exception as e:
            print(f"Error in graph execution: {str(e)}")
            raise
        
        return output

    def _log_generations_updates(self, state: Dict):
        """Log updates to generations dictionary."""
        generations = state.get('generations_slow_graph', {})
        for key, value in generations.items():
            if key not in {'strategy_results', 'strategy_analysis'}:  # Skip complex nested structures
                title = Text(f"Generation Update: {key}", style="bold green")
                content = Text(str(value))
                panel = Panel(content, title=title)
                print(panel)

    def _log_improvement_history(self, state: Dict):
        """Log updates to improvement history."""
        history = state.get('improvement_history', [])
        if history:
            latest = history[-1]
            title = Text("Latest Improvement", style="bold blue")
            content = Text(
                f"Strategy: {latest.get('strategy_type', 'unknown')}\n"
                f"Outcome: {latest.get('outcome', 'unknown')}\n"
                f"Improvements:\n"
                f"  New Distribution: {latest.get('improvements', {}).get('new_distribution', 0):.4f}\n"
                f"  Old Distribution: {latest.get('improvements', {}).get('old_distribution', 0):.4f}\n"
                f"Evaluation: {latest.get('evaluation', {}).get('recommendation', {}).get('action', 'unknown')}"
            )
            panel = Panel(content, title=title)
            print(panel)

    def _log_strategy_progress(self, state: Dict):
        """Log progress of strategy execution."""
        strategy_results = state.get('generations_slow_graph', {}).get('strategy_results', {})
        current_strategy = state.get('generations_slow_graph', {}).get('current_strategy')
        
        title = Text("Strategy Progress", style="bold yellow")
        content = []
        
        for strategy, results in strategy_results.items():
            status = "✓" if results.get('tried', False) else "○"
            current = "→" if strategy == current_strategy else " "
            content.append(f"{current} [{status}] {strategy}")
        
        panel = Panel("\n".join(content), title=title)
        print(panel)

class EnhancedStateGraph(StateGraph):
    """Enhanced StateGraph with function name printing capability."""
    def add_node(self, node_name, function):
        decorated_function = print_function_name(function)
        super().add_node(node_name, decorated_function)