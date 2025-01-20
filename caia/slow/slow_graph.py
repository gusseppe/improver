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
    prompt_summarize_monitoring_report, # New prompt
    prompt_evaluate_change,
    prompt_fix_code_slow,
    # prompt_fix_code,
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
                'ensemble_method': 'ensemble_method'
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
                'retry': 'analyze_needs'  # Go back to analysis if error
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

    # def build_plan(self):
    #     """Build the graph structure with nodes and edges."""
    #     print("\nBuilding Graph Plan...", "-"*50)
        
    #     # Add nodes for each step of the process
    #     self.graph.add_node('distill_memories', self.distill_memories)
    #     self.graph.add_node('analyze_needs', self.analyze_needs)
        
    #     # Strategy nodes
    #     self.graph.add_node('model_selection', self.generate_model_selection_change)
    #     self.graph.add_node('hyperparameter_tuning', self.generate_hyperparameter_tuning)
    #     self.graph.add_node('ensemble_method', self.generate_ensemble_method)
        
    #     # Execution and evaluation nodes
    #     self.graph.add_node('apply_change', self.apply_change)
    #     self.graph.add_node('evaluate_change', self.evaluate_change)
        
    #     print("Added all nodes...")
        
    #     # Set entry point
    #     self.graph.set_entry_point('distill_memories')
    #     print("Set entry point...")
        
    #     # Add edges for the main flow
    #     self.graph.add_edge('distill_memories', 'analyze_needs')
        
    #     # Add conditional edge to route to appropriate strategy
    #     self.graph.add_conditional_edges(
    #         'analyze_needs',
    #         self.route_to_strategy,
    #         {
    #             'model_selection': 'model_selection',
    #             'hyperparameter_tuning': 'hyperparameter_tuning',
    #             'ensemble_method': 'ensemble_method'
    #         }
    #     )
    #     print("Added strategy routing edges...")
        
    #     # Connect all strategy nodes to apply_change
    #     strategy_nodes = ['model_selection', 'hyperparameter_tuning', 'ensemble_method']
    #     for node in strategy_nodes:
    #         self.graph.add_edge(node, 'apply_change')
    #     print("Connected strategies to apply_change...")
        
    #     # Add conditional edge for apply_change
    #     self.graph.add_conditional_edges(
    #         'apply_change',
    #         self.should_evaluate_code,
    #         {
    #             'evaluate': 'evaluate_change',
    #             'retry': 'analyze_needs'
    #         }
    #     )
    #     print("Added apply_change edges...")
        
    #     # Create a wrapper for should_continue_improving that ensures it's called
    #     def should_continue_wrapper(state):
    #         print("\nCalling should_continue_improving wrapper...")
    #         result = self.should_continue_improving(state)
    #         print(f"should_continue_improving returned: {result}")
    #         return result
        
    #     # Add conditional edge for evaluate_change with wrapped function
    #     print("Adding evaluate_change conditional edges...")
    #     self.graph.add_conditional_edges(
    #         'evaluate_change',
    #         should_continue_wrapper,  # Use wrapped version
    #         {
    #             'continue': 'analyze_needs',
    #             'end': END
    #         }
    #     )
    #     print("Added evaluate_change edges...")
    #     print("Graph build complete!", "-"*50)


    def initialize_generations(self) -> Dict[str, Any]:
        """Initialize the generations dictionary with strategy tracking."""
        return {
            'distilled_insights': '',
            'tiny_change': '',
            'execution_output': '',
            'execution_success': False,
            'new_accuracy': 0.0,
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
                    'best_accuracy': 0.0,
                    'models_tried': []
                },
                'hyperparameter_tuning': {
                    'tried': False,
                    'best_params': {},
                    'best_accuracy': 0.0
                },
                'ensemble_method': {
                    'tried': False,
                    'best_ensemble': '',
                    'best_accuracy': 0.0
                }
            }
        }
    def distill_memories(self, state: WorkingMemory) -> WorkingMemory:
        """Distill insights from semantic and episodic memories.
        
        Args:
            state: Current working memory state
            
        Returns:
            Updated working memory with distilled insights
        """
        # Ensure generations dictionary is initialized
        # if 'generations' not in state:
        state['generations_slow_graph'] = self.initialize_generations()

        # Get model documentation summary
        semantic_memory = state['semantic_memory']
        doc_prompt = prompt_summarize_model_docs()
        doc_chain = doc_prompt | self.llm
        model_params_summary = doc_chain.invoke({'input': semantic_memory.model_object.__doc__}).content
        
        # Summarize monitoring report
        # monitoring_prompt = prompt_summarize_monitoring_report()
        # monitoring_chain = monitoring_prompt | self.llm
        # monitoring_summary = monitoring_chain.invoke(
        #     {'input': yaml.dump(state['monitoring_report'])}
        # ).content
        
        # Prepare semantic memory
        semantic_memory_dict = {
            'old_training_code': semantic_memory.model_code,
            'model_documentation': model_params_summary,
            'dataset_description': semantic_memory.reference_dataset.description
        }

        # Get episodic memory
        episodic_memory = state['episodic_memory'][-1].quick_insight
        
        # Create the prompt for the LLM
        prompt = prompt_distill_memories()
        
        # Prepare the input in YAML format
        yaml_content = {
            'semantic_memory': str(semantic_memory_dict),
            'episodic_memory': str(episodic_memory),
            # 'episodic_memory': str(episodic_memory),
            # 'monitoring_summary': monitoring_summary
        }
        
        # Invoke the LLM with yaml content
        chain = prompt | self.llm
        output = chain.invoke({'input': yaml.dump(yaml_content)}).content
        
        # Update the state with the distilled insights
        if 'generations_slow_graph' not in state:
            state['generations_slow_graph'] = {}
        state['generations_slow_graph']['distilled_insights'] = output
        
        return state
    
    # def analyze_needs(self, state: WorkingMemory) -> WorkingMemory:
    #     """Analyzes current model performance and determines next best strategy."""
        
    #     # Extract key metrics
    #     execution_output = state['generations_slow_graph'].get('execution_output', '')
    #     current_accuracy = state['generations_slow_graph'].get('new_accuracy', 0.0)
    #     previous_accuracy = state.get('previous_accuracy', 0.0)
        
    #     # Extract accuracy values from execution output
    #     ref_accuracy = 0.0
    #     new_accuracy = 0.0
    #     combined_accuracy = 0.0
        
    #     try:
    #         for line in execution_output.split('\n'):
    #             if 'reference distribution' in line:
    #                 ref_accuracy = float(line.split(':')[-1].strip())
    #             elif 'new distribution' in line:
    #                 new_accuracy = float(line.split(':')[-1].strip())
    #             elif 'average score' in line:
    #                 combined_accuracy = float(line.split(':')[-1].strip())
    #     except:
    #         pass
        
    #     # Prepare concise input YAML
    #     yaml_content = {
    #         "current_metrics": {
    #             "reference_accuracy": ref_accuracy,
    #             "new_data_accuracy": new_accuracy,
    #             "combined_accuracy": combined_accuracy,
    #             "accuracy_change": current_accuracy - previous_accuracy if current_accuracy else 0
    #         },
    #         "strategies_tried": []
    #     }
        
    #     # Add strategy history
    #     for strategy, results in state['generations_slow_graph'].get('strategy_results', {}).items():
    #         if results.get('tried', False):
    #             yaml_content["strategies_tried"].append({
    #                 strategy: {
    #                     "models": results.get('models_tried', []) if strategy == 'model_selection' else None,
    #                     "best_accuracy": results.get('best_accuracy', 0.0)
    #                 }
    #             })
        
    #     # Get analysis from LLM
    #     prompt = prompt_analyze_improvement_needs()
    #     chain = prompt | self.llm
    #     analysis = chain.invoke({'input': yaml.dump(yaml_content)}).content
        
    #     try:
    #         # Parse YAML output
    #         analysis_dict = yaml.safe_load(analysis)
            
    #         # Update state with analysis results
    #         state['generations_slow_graph'].update({
    #             'current_strategy': analysis_dict.get('recommended_strategy', 'model_selection'),
    #             'strategy_analysis': {
    #                 'recommended_strategy': analysis_dict.get('recommended_strategy', ''),
    #                 'reasoning': analysis_dict.get('reasoning', ''),
    #                 'performance_gaps': analysis_dict.get('performance_gaps', []),
    #                 'next_steps': analysis_dict.get('next_steps', [])
    #             }
    #         })
            
    #     except yaml.YAMLError as e:
    #         print(f"Error parsing analysis output: {e}")
    #         state['generations_slow_graph'].update({
    #             'current_strategy': 'model_selection',
    #             'strategy_analysis': {
    #                 'recommended_strategy': 'model_selection',
    #                 'reasoning': 'Failed to parse analysis, defaulting to model selection',
    #                 'performance_gaps': [],
    #                 'next_steps': ['Try different model architecture']
    #             }
    #         })
        
    #     # Print analysis for debugging
    #     print("\nStrategy Analysis:", "-"*50)
    #     print(yaml.dump(state['generations_slow_graph']['strategy_analysis'], default_flow_style=False))
        
    #     return state

    # def analyze_needs(self, state: WorkingMemory) -> WorkingMemory:
    #     """Analyzes current model performance and determines next best strategy.
        
    #     Args:
    #         state: Current working memory state
            
    #     Returns:
    #         Updated working memory with analysis results
    #     """
    #     # Ensure generations dictionary is initialized
    #     # if 'generations' not in state:
    #     #     state['generations_slow_graph'] = self.initialize_generations()
            
    #     execution_output = state['generations_slow_graph'].get('execution_output', '')
    #     strategy_results = state['generations_slow_graph'].get('strategy_results', {})
        
    #     # Create prompt for analysis
    #     prompt = prompt_analyze_improvement_needs()
    #     chain = prompt | self.llm
        
    #     # Prepare input YAML
    #     yaml_content = {
    #         "current_performance": execution_output,
    #         "improvement_history": state['improvement_history'],
    #         "strategy_results": strategy_results,
    #         # "threshold": state['threshold'],
    #         # "initial_distilled_memory": state['monitoring_report']
    #         # "monitoring_report": state['monitoring_report']
    #     }
        
    #     # Get analysis from LLM with properly formatted input
    #     analysis = chain.invoke({'input': yaml.dump(yaml_content)}).content
        
    #     # Parse YAML output
    #     analysis_dict = yaml.safe_load(analysis)
        
    #     # Update generations with analysis
    #     # new_generations = state['generations_slow_graph'].copy()
    #     state['generations_slow_graph'].update({
    #         'current_strategy': analysis_dict['recommended_strategy'],
    #         'strategy_analysis': analysis_dict
    #     })
    #     # state['generations_slow_graph'] = new_generations
    #     return state
    #     # return {**state, 'generations': new_generations}

    def analyze_needs(self, state: WorkingMemory) -> WorkingMemory:
        """Analyzes current model performance and determines next best strategy."""
        
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
        strategy_results = state['generations_slow_graph'].get('strategy_results', {})
        models_tried = strategy_results.get('model_selection', {}).get('models_tried', [])
        
        prompt = prompt_model_selection_change()
        chain = prompt | self.llm
        
        # Prepare input YAML
        yaml_content = {
            "current_code": state['semantic_memory'].model_code,
            "execution_output": state['generations_slow_graph'].get('execution_output', ''),
            "models_tried": models_tried,
            # "dataset_representation": state['dataset_representation']
        }
        
        # Get model selection changes from LLM with properly formatted input
        change_output = chain.invoke({'input': yaml.dump(yaml_content)}).content
        
        # Store the raw model selection output
        state['generations_slow_graph']['tiny_change'] = change_output
        
        # Parse YAML output for strategy results update
        try:
            change_dict = yaml.safe_load(change_output)
            # Update strategy results if valid YAML
            if isinstance(change_dict, dict) and 'model_name' in change_dict:
                new_generations = state['generations_slow_graph'].copy()
                new_generations['strategy_results']['model_selection'] = {
                    'tried': True,
                    'models_tried': models_tried + [change_dict['model_name']],
                    'best_accuracy': max(
                        strategy_results.get('model_selection', {}).get('best_accuracy', 0),
                        state['generations_slow_graph'].get('new_accuracy', 0)
                    )
                }
                state['generations_slow_graph'].update(new_generations)
        except yaml.YAMLError:
            # If YAML parsing fails, continue with just the raw output
            pass

        return state
        # return {**state, 'generations': new_generations}

    def generate_hyperparameter_tuning(self, state: WorkingMemory) -> WorkingMemory:
        """Generates hyperparameter optimization changes."""
        strategy_results = state['generations_slow_graph'].get('strategy_results', {})
        current_params = strategy_results.get('hyperparameter_tuning', {}).get('best_params', {})
        
        prompt = prompt_hyperparameter_tuning()
        chain = prompt | self.llm
        
        # Prepare input YAML
        yaml_content = {
            "current_code": state['semantic_memory'].model_code,
            "execution_output": state['generations_slow_graph'].get('execution_output', ''),
            "current_params": current_params,
            # "dataset_representation": state['dataset_representation']
        }
        
        # Properly format input for the chain with the 'input' key
        change_output = chain.invoke({'input': yaml.dump(yaml_content)}).content
        
        try:
            # Parse YAML output
            change_dict = yaml.safe_load(change_output)
            
            # Update generations
            new_generations = state['generations_slow_graph'].copy()
            
            # Store the raw hyperparameter tuning output
            state['generations_slow_graph']['tiny_change'] = change_output
            
            if isinstance(change_dict, dict):
                # Update strategy results with hyperparameter information
                new_generations['strategy_results']['hyperparameter_tuning'] = {
                    'tried': True,
                    'best_params': change_dict.get('hyperparameters', {}),
                    'best_accuracy': max(
                        strategy_results.get('hyperparameter_tuning', {}).get('best_accuracy', 0),
                        state['generations_slow_graph'].get('new_accuracy', 0)
                    )
                }
                state['generations_slow_graph'].update(new_generations)
                
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {str(e)}")
            # If YAML parsing fails, still store the raw output
            state['generations_slow_graph']['tiny_change'] = change_output

        return state

        # return {**state, 'generations': new_generations}

    def generate_ensemble_method(self, state: WorkingMemory) -> WorkingMemory:
        """Generates ensemble-based improvements.
        
        Args:
            state: Current working memory state
            
        Returns:
            Updated working memory with ensemble method changes
        """
        strategy_results = state['generations_slow_graph'].get('strategy_results', {})
        
        prompt = prompt_ensemble_method()
        chain = prompt | self.llm
        
        # Prepare input YAML
        yaml_content = {
            "current_code": state['semantic_memory'].model_code,
            "execution_output": state['generations_slow_graph'].get('execution_output', ''),
            "strategy_results": strategy_results,
            # "dataset_representation": state['dataset_representation']
        }
        
        # Properly format input for the chain with the 'input' key
        change_output = chain.invoke({'input': yaml.dump(yaml_content)}).content
        
        try:
            # Parse YAML output
            change_dict = yaml.safe_load(change_output)
            
            # Store the raw ensemble method output
            state['generations_slow_graph']['tiny_change'] = change_output
            
            if isinstance(change_dict, dict):
                if 'new_training_code' in change_dict:
                    # Update strategy results with ensemble information
                    new_generations = state['generations_slow_graph'].copy()
                    new_generations['strategy_results']['ensemble_method'] = {
                        'tried': True,
                        'best_ensemble': change_dict.get('ensemble_type', 'unknown'),
                        'best_accuracy': max(
                            strategy_results.get('ensemble_method', {}).get('best_accuracy', 0),
                            state['generations_slow_graph'].get('new_accuracy', 0)
                        )
                    }
                    state['generations_slow_graph'].update(new_generations)
            else:
                print("Warning: Invalid YAML structure in ensemble method output")
                
        except yaml.YAMLError as e:
            print(f"Error parsing YAML in ensemble method: {str(e)}")
            # If YAML parsing fails, still store the raw output
            state['generations_slow_graph']['tiny_change'] = change_output

        return state
        
        # return {**state, 'generations': new_generations}

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
            'evaluate' if code executed successfully, 'retry' otherwise
        """
        return 'evaluate' if state['generations_slow_graph'].get('execution_success', False) else 'retry'

    # def should_continue_improving(self, state: WorkingMemory) -> str:
    #     """Determine if improvement process should continue.
        
    #     Args:
    #         state: Current working memory state
            
    #     Returns:
    #         'continue' if more improvements needed, 'end' otherwise
    #     """
    #     if not state['improvement_history']:
    #         return 'continue'
            
    #     latest_improvement = state['improvement_history'][-1]
    #     iteration_count = len(state['improvement_history'])
        
    #     # Get strategy results
    #     strategy_results = state['generations_slow_graph'].get('strategy_results', {})
    #     strategies_tried = [
    #         strategy for strategy, result in strategy_results.items() 
    #         if result.get('tried', False)
    #     ]
        
    #     # Stop conditions:
    #     # 1. Reached max iterations (10)
    #     # 2. Found significant improvement (> threshold)
    #     # 3. Tried all strategies with no improvement
    #     # 4. Multiple failed attempts in a row (3)
    #     if (
    #         iteration_count >= 10 or
    #         latest_improvement['accuracy_change'] > state['threshold'] or
    #         len(strategies_tried) >= 3 or  # We implemented 3 strategies
    #         (iteration_count > 2 and all(
    #             h['evaluation']['recommendation']['action'] == 'reject' 
    #             for h in state['improvement_history'][-3:]
    #         ))
    #     ):
    #         return 'end'
            
    #     return 'continue'

    def should_continue_improving(self, state: WorkingMemory) -> str:
        """Determine if improvement process should continue."""

        print("\nshould_continue_improving called!", "-"*50)
        print("State keys:", state.keys())
        
        if not state.get('improvement_history'):
            print("No improvement history found, continuing...")
            return 'continue'
        
        # Get latest improvement info
        improvement_history = state['improvement_history']
        latest_improvement = improvement_history[-1]
        print(f"\nLatest improvement:", latest_improvement.keys())
        
        # Get strategy information
        strategy_results = state['generations_slow_graph'].get('strategy_results', {})
        strategies_tried = [
            strategy for strategy, result in strategy_results.items() 
            if result.get('tried', False)
        ]
        print(f"Strategies tried: {strategies_tried}")
        
        # Get evaluation details
        latest_eval = latest_improvement.get('evaluation', {}).get('evaluation', {})
        recommendation = latest_eval.get('recommendation', {})
        next_steps = latest_eval.get('next_steps', [])
        print(f"Recommendation: {recommendation.get('action')}")
        print(f"Next steps: {next_steps}")
        
        # Check conditions
        should_stop = (
            len(improvement_history) >= 10 or  # Max iterations
            len(strategies_tried) >= 3  # All strategies tried
        )
        
        should_continue = (
            len(strategies_tried) < 3 or  # Still have untried strategies
            (recommendation.get('action') == 'accept' and  # Current change accepted
            len(next_steps) > 0)  # Have next steps
        )
        
        print(f"Should stop: {should_stop}")
        print(f"Should continue: {should_continue}")
        
        decision = 'continue' if (should_continue and not should_stop) else 'end'
        print(f"Final decision: {decision}")
        return decision
 

    def apply_change(self, state: WorkingMemory) -> WorkingMemory:
        """Apply the generated change and execute the code."""
        try:
            # Get the current code with tiny change
            current_code_yaml = state['generations_slow_graph']['tiny_change']
            
            # Try to parse as YAML first
            try:
                parsed_yaml = yaml.safe_load(current_code_yaml)
                if isinstance(parsed_yaml, dict) and 'new_training_code' in parsed_yaml:
                    current_code = parsed_yaml['new_training_code']
                else:
                    current_code = current_code_yaml
            except yaml.YAMLError:
                current_code = current_code_yaml

            max_retries = 2
            current_try = 0
            
            while current_try < max_retries:
                # Attempt to execute the code
                wrapped_code = f"```python\n{current_code}\n```"
                executor = DockerCommandLineCodeExecutor(
                    image="agent-env:latest",
                    timeout=10,
                    work_dir=".",
                )
                
                code_executor_agent = ConversableAgent(
                    "code_executor_agent",
                    llm_config=False,
                    code_execution_config={"executor": executor},
                    human_input_mode="NEVER",
                )
                
                execution_output = code_executor_agent.generate_reply(
                    messages=[{"role": "user", "content": wrapped_code}]
                )
                
                # If execution succeeds, break the loop
                if not any(err in execution_output.lower() for err in ['error', 'failed', 'TypeError', 'ValueError', 'NameError']):
                    break
                    
                # If execution fails and we have retries left
                if current_try < max_retries - 1:
                    # Prepare input for code fixing
                    yaml_content = {
                        "current_code": current_code,
                        "error_output": execution_output
                    }
                    
                    # Create fix code prompt and get fixed code
                    prompt = prompt_fix_code()
                    chain = prompt | self.llm
                    fixed_output = chain.invoke({'input': yaml.dump(yaml_content)}).content
                    
                    try:
                        fixed_dict = yaml.safe_load(fixed_output)
                        if isinstance(fixed_dict, dict) and 'fixed_code' in fixed_dict:
                            current_code = fixed_dict['fixed_code']
                            # Store validation steps for debugging
                            if 'validation_steps' in fixed_dict:
                                state['generations_slow_graph']['validation_steps'] = fixed_dict['validation_steps']
                    except yaml.YAMLError:
                        print("Failed to parse fixed code output")
                        break
                        
                current_try += 1
            
            # Store final execution results
            print("Execution Output:", "-"*100)
            print(execution_output)
            state['generations_slow_graph']['execution_output'] = execution_output
            
            # Check if execution was successful
            success_indicators = [
                "succeeded",
                "new model evaluated",
                "accuracy:",
                "score:",
                "completed successfully"
            ]
            
            if any(indicator in execution_output.lower() for indicator in success_indicators):
                state['generations_slow_graph']['execution_success'] = True
                
                # Extract accuracy using various patterns
                accuracy_patterns = [
                    r"New model evaluated on new distribution: ([\d.]+)",
                    r"New Model Accuracy: ([\d.]+)",
                    r"Final accuracy: ([\d.]+)",
                    r"Test accuracy: ([\d.]+)",
                    r"Score: ([\d.]+)",
                    r"Accuracy: ([\d.]+)"
                ]
                
                for pattern in accuracy_patterns:
                    try:
                        match = re.search(pattern, execution_output)
                        if match:
                            accuracy = float(match.group(1))
                            state['generations_slow_graph']['new_accuracy'] = accuracy
                            break
                    except (AttributeError, ValueError):
                        continue
                
                if 'new_accuracy' not in state['generations_slow_graph']:
                    state['generations_slow_graph']['execution_success'] = False
                    
            else:
                state['generations_slow_graph']['execution_success'] = False
                
        except Exception as e:
            print(f"Error in apply_change: {str(e)}")
            state['generations_slow_graph']['execution_success'] = False
            state['generations_slow_graph']['execution_output'] = str(e)
        
        return state

    def evaluate_change(self, state: WorkingMemory) -> WorkingMemory:
        """Evaluate the applied change and update improvement history."""
        print("\nEntering evaluate_change...", "-"*50)
        
        # Get current accuracy from execution output
        execution_output = state['generations_slow_graph']['execution_output']
        current_accuracy = None
        
        # Try to extract new model accuracy
        try:
            for line in execution_output.split('\n'):
                if 'New model evaluated on new distribution:' in line:
                    current_accuracy = float(line.split(':')[1].strip())
                    break
        except:
            current_accuracy = state['generations_slow_graph'].get('new_accuracy', 0)
        
        # Get previous accuracy - from either improvement history or semantic memory
        if state['improvement_history']:
            previous_metrics = state['improvement_history'][-1].get('evaluation', {}).get('evaluation', {}).get('metrics', {})
            previous_accuracy = previous_metrics.get('accuracy_new_model', 0)
        else:
            # Get initial model accuracy from semantic memory or first run
            previous_accuracy = state.get('previous_accuracy', 0)
        
        # Get current and previous code
        current_code = state['generations_slow_graph']['tiny_change']
        previous_code = state['semantic_memory'].model_code if len(state['improvement_history']) == 0 else state['improvement_history'][-1].get('new_code', '')
        
        # Prepare evaluation input
        yaml_content = {
            'current_code': current_code,
            'execution_output': execution_output,
            'previous_accuracy': previous_accuracy
        }
        
        # Get evaluation from LLM
        prompt = prompt_evaluate_change()
        chain = prompt | self.llm
        evaluation_output = chain.invoke({'input': yaml.dump(yaml_content)}).content
        evaluation_result = yaml.safe_load(evaluation_output)
        
        # Print debug information
        print("\nEvaluation Metrics:", "-"*50)
        print(f"Current Accuracy: {current_accuracy}")
        print(f"Previous Accuracy: {previous_accuracy}")
        print(f"Improvement: {current_accuracy - previous_accuracy if current_accuracy else 0}")
        
        # Update improvement history
        improvement_entry = {
            'previous_code': previous_code,
            'new_code': current_code,
            'accuracy_change': current_accuracy - previous_accuracy if current_accuracy else 0,
            'evaluation': evaluation_result
        }
        
        if 'improvement_history' not in state:
            state['improvement_history'] = []
        state['improvement_history'].append(improvement_entry)
        
        # Update state
        state['previous_accuracy'] = current_accuracy
        state['previous_code'] = current_code
        state['generations_slow_graph']['evaluation'] = evaluation_result
        
        # Update strategy results with actual metrics
        current_strategy = state['generations_slow_graph'].get('current_strategy')
        if current_strategy and current_accuracy:
            strategy_results = state['generations_slow_graph'].get('strategy_results', {})
            if current_strategy in strategy_results:
                strategy_results[current_strategy].update({
                    'tried': True,
                    'best_accuracy': max(
                        strategy_results[current_strategy].get('best_accuracy', 0),
                        current_accuracy
                    )
                })
                state['generations_slow_graph']['strategy_results'] = strategy_results
        
        print("\nStrategy Status:", "-"*50)
        print(f"Current Strategy: {current_strategy}")
        print("Strategy Results:", state['generations_slow_graph']['strategy_results'])
        
        return state


    # def evaluate_change(self, state: WorkingMemory) -> WorkingMemory:
    #     """Evaluate the applied change and update improvement history."""
    #     print("\nEntering evaluate_change...", "-"*50)
        
    #     # Get current metrics
    #     current_accuracy = state['generations_slow_graph'].get('new_accuracy')
    #     previous_accuracy = state.get('previous_accuracy', 0)
        
    #     # Get current and previous code
    #     current_code = state['generations_slow_graph']['tiny_change']
    #     previous_code = state['semantic_memory'].model_code if len(state['improvement_history']) == 0 else state['improvement_history'][-1].get('new_code', '')
        
    #     # Create evaluation prompt
    #     prompt = prompt_evaluate_change()
        
    #     # Prepare input YAML
    #     yaml_content = {
    #         'current_code': current_code,
    #         'execution_output': state['generations_slow_graph']['execution_output'],
    #         'previous_accuracy': previous_accuracy
    #     }
        
    #     # Get evaluation from LLM
    #     chain = prompt | self.llm
    #     evaluation_output = chain.invoke({'input': yaml.dump(yaml_content)}).content
    #     evaluation_result = yaml.safe_load(evaluation_output)
        
    #     # Print debug information about the evaluation
    #     print("\nEvaluation Result:", "-"*50)
    #     print(f"Current Accuracy: {current_accuracy}")
    #     print(f"Previous Accuracy: {previous_accuracy}")
    #     print(f"Recommendation: {evaluation_result.get('evaluation', {}).get('recommendation', {}).get('action')}")
    #     print(f"Next Steps: {evaluation_result.get('evaluation', {}).get('next_steps', [])}")
        
    #     # Prepare improvement history entry
    #     improvement_entry = {
    #         'previous_code': previous_code,
    #         'new_code': current_code,
    #         'accuracy_change': current_accuracy - previous_accuracy if current_accuracy else 0,
    #         'evaluation': evaluation_result
    #     }
        
    #     # Update improvement history
    #     if 'improvement_history' not in state:
    #         state['improvement_history'] = []
    #     state['improvement_history'].append(improvement_entry)
        
    #     # Print improvement history statistics
    #     print("\nImprovement History:", "-"*50)
    #     print(f"Total Improvements Attempted: {len(state['improvement_history'])}")
    #     print(f"Latest Accuracy Change: {improvement_entry['accuracy_change']}")
        
    #     # Update current state for next iteration
    #     state['previous_accuracy'] = current_accuracy
    #     state['previous_code'] = current_code
    #     state['generations_slow_graph']['evaluation'] = evaluation_result
        
    #     # Update strategy results
    #     current_strategy = state['generations_slow_graph'].get('current_strategy')
    #     if current_strategy:
    #         strategy_results = state['generations_slow_graph'].get('strategy_results', {})
    #         if current_strategy in strategy_results:
    #             strategy_results[current_strategy]['tried'] = True
    #             state['generations_slow_graph']['strategy_results'] = strategy_results
        
    #     print("\nExiting evaluate_change...", "-"*50)
    #     print(f"Current Strategy: {current_strategy}")
    #     print(f"Strategies Tried: {[s for s, r in state['generations_slow_graph'].get('strategy_results', {}).items() if r.get('tried', False)]}")
        
    #     return state

    def should_evaluate_code(self, state: WorkingMemory) -> str:
        """Determine if code should be evaluated or retried.
        
        Args:
            state: Current working memory state
            
        Returns:
            'evaluate' if code executed successfully, 'retry' otherwise
        """
        return 'evaluate' if state['generations_slow_graph'].get('execution_success', False) else 'retry'



    def run(self, initial_state: WorkingMemory):
        """Run the slow improvement process.
        
        Args:
            initial_state: Initial working memory state
            
        Returns:
            Final state after improvements
        """
        # Initialize generations if not present
        if 'generations_slow_graph' not in initial_state:
            initial_state['generations_slow_graph'] = self.initialize_generations()
            
        output_keys = ['generations_slow_graph', 'improvement_history']
        visited_keys = []
        
        for output in self.decision_procedure.stream(
            initial_state, 
            output_keys=output_keys, 
            debug=False
        ):
            for node_name, state in output.items():
                # Print generations updates
                for k, v in state['generations_slow_graph'].items():
                    if state['generations_slow_graph'][k] and k not in visited_keys:
                        title = Text(k, style="bold green")
                        content = Text(str(v))
                        panel = Panel(content, title=title)
                        print(panel)
                        visited_keys.append(k)
                        
                # Print improvement history updates
                if 'improvement_history' in state and state['improvement_history']:
                    latest_improvement = state['improvement_history'][-1]
                    if 'latest_improvement' not in visited_keys:
                        title = Text("Latest Improvement", style="bold blue")
                        content = Text(str(latest_improvement))
                        panel = Panel(content, title=title)
                        print(panel)
                        visited_keys.append('latest_improvement')
        
        return output
 

class EnhancedStateGraph(StateGraph):
    """Enhanced StateGraph with function name printing capability."""
    def add_node(self, node_name, function):
        decorated_function = print_function_name(function)
        super().add_node(node_name, decorated_function)