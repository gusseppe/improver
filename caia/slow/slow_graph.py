from typing import TypedDict, Dict, List, Optional, Annotated, Any
import textwrap
import os
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
from datetime import datetime
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
    def __init__(self, llm, max_iterations=3, max_failures=3, debug=False):
        """Initialize the slow improvement graph.
        
        Args:
            llm: The language model to use for generation
            max_iterations: Default maximum number of improvement iterations to run
            max_failures: Maximum number of consecutive execution failures allowed
            debug: Whether to run in debug mode
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.max_failures = max_failures
        self.graph = EnhancedStateGraph(WorkingMemory)
        self.build_plan()
        self.decision_procedure = self.graph.compile(debug=debug)
        self.start_time = datetime.now()
        self.token_counts = {"prompt": 0, "completion": 0, "total": 0}
        
    def draw_graph(self):
        """Visualize the graph structure."""
        return display(Image(self.decision_procedure.get_graph().draw_mermaid_png()))
        
    def build_plan(self):
        """Build the graph structure with nodes and edges."""
        # Add nodes for each step of the process
        self.graph.add_node('check_fast_graph_results', self.check_fast_graph_results)
        self.graph.add_node('distill_memories', self.distill_memories)
        self.graph.add_node('analyze_needs', self.analyze_needs)
        
        # Strategy nodes
        self.graph.add_node('model_selection', self.generate_model_selection_change)
        self.graph.add_node('hyperparameter_tuning', self.generate_hyperparameter_tuning)
        self.graph.add_node('ensemble_method', self.generate_ensemble_method)
        
        # Execution and evaluation nodes
        self.graph.add_node('apply_change', self.apply_change)
        self.graph.add_node('evaluate_change', self.evaluate_change)
        
        # Set entry point to check for fast graph results first
        self.graph.set_entry_point('check_fast_graph_results')
        
        # Add edges for the main flow
        self.graph.add_edge('check_fast_graph_results', 'distill_memories')
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
                'retry': 'analyze_needs',
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
            'consecutive_failures': 0,
            'last_successful_state': {},
            
            # Token tracking
            'token_usage': {
                'prompt': 0,
                'completion': 0,
                'total': 0
            },
            
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
            },
            # Fast graph integration tracking
            'fast_graph_integrated': False,
            'fast_graph_metrics': {},
            'fast_graph_code': ''
        }
    
    # Updated check_fast_graph_results method with proper handling of episodic memory
    def check_fast_graph_results(self, state: WorkingMemory) -> WorkingMemory:
        """Check if fast graph has run and extract its results from either
        improvement_history or generations_fast_graph."""
        state['generations_slow_graph'] = self.initialize_generations()
        
        # First check if there are direct fast graph generation results
        fast_graph_results = state.get('generations_fast_graph', {})
        
        if fast_graph_results and fast_graph_results.get('execution_success', False):
            # Extract information from generations_fast_graph
            print("\nDetected Fast Graph Results from generations_fast_graph:", "-"*50)
            
            # Parse the new_training_code from YAML format
            fast_code = ""
            try:
                if 'new_training_code' in fast_graph_results:
                    yaml_content = fast_graph_results['new_training_code']
                    parsed_yaml = yaml.safe_load(yaml_content)
                    if isinstance(parsed_yaml, dict) and 'new_training_code' in parsed_yaml:
                        fast_code = parsed_yaml['new_training_code']
            except yaml.YAMLError:
                print("Warning: Could not parse new_training_code YAML")
            
            # Extract metrics
            model_old_score = fast_graph_results.get('model_old_score', {})
            model_new_score = fast_graph_results.get('model_new_score', {})
            
            # Update the slow graph generations with fast graph results
            state['generations_slow_graph'].update({
                'fast_graph_integrated': True,
                'fast_graph_metrics': {
                    'old_model': model_old_score,
                    'new_model': model_new_score
                },
                'fast_graph_code': fast_code,
                'execution_output': fast_graph_results.get('execution_output', ''),
                'model_old_score': model_old_score,
                'model_new_score': model_new_score
            })
            
            print(f"Fast Graph Code Length: {len(fast_code)} characters")
            print("Fast Graph Metrics:")
            print(f"  New Distribution: {model_new_score.get('on_new_data', 0):.4f}")
            print(f"  Old Distribution: {model_new_score.get('on_old_data', 0):.4f}")
            
            # FIXED: Safely check if episodic_memory exists and has elements with quick_insight
            if (state.get('episodic_memory') and 
                isinstance(state['episodic_memory'], list) and 
                len(state['episodic_memory']) > 0 and 
                hasattr(state['episodic_memory'][-1], 'quick_insight') and
                state['episodic_memory'][-1].quick_insight):
                print("Found additional fast graph insights in episodic memory quick_insight")
                state['generations_slow_graph']['quick_insight'] = state['episodic_memory'][-1].quick_insight
        
        # As a fallback, also check improvement_history as originally designed
        else:
            fast_graph_improvements = [
                entry for entry in state.get('improvement_history', [])
                if entry.get('graph_type') == 'fast'
            ]
            
            if fast_graph_improvements:
                # Get the most recent fast graph improvement
                latest_fast_improvement = fast_graph_improvements[-1]
                
                # Extract metrics and code
                fast_metrics = latest_fast_improvement.get('metrics', {})
                fast_code = latest_fast_improvement.get('new_code', '')
                
                # Update state with fast graph information
                state['generations_slow_graph'].update({
                    'fast_graph_integrated': True,
                    'fast_graph_metrics': fast_metrics,
                    'fast_graph_code': fast_code,
                    'execution_output': "",  # Initialize with empty string
                })
                
                # FIXED: Safely check episodic_memory
                if (state.get('episodic_memory') and 
                    isinstance(state['episodic_memory'], list) and 
                    len(state['episodic_memory']) > 0 and 
                    hasattr(state['episodic_memory'][-1], 'quick_insight')):
                    state['generations_slow_graph']['execution_output'] = state['episodic_memory'][-1].quick_insight.get('execution_output', '')
                
                print("\nDetected Fast Graph Results from improvement_history:", "-"*50)
                print(f"Fast Graph Code Length: {len(fast_code)} characters")
                print("Fast Graph Metrics:")
                print(f"  New Distribution: {fast_metrics.get('new_model', {}).get('on_new_data', 0):.4f}")
                print(f"  Old Distribution: {fast_metrics.get('new_model', {}).get('on_old_data', 0):.4f}")
            else:
                print("\nNo Fast Graph Results Found - Starting from scratch")
        
        # Always check if YAML metrics files exist from fast graph and load them
        if os.path.exists('old_metrics.yaml') and os.path.exists('fast_graph_metrics.yaml'):
            try:
                with open('old_metrics.yaml', 'r') as f:
                    old_metrics = yaml.safe_load(f)
                    state['generations_slow_graph']['model_old_score'] = old_metrics.get('model_old_score', {})
                
                with open('fast_graph_metrics.yaml', 'r') as f:
                    new_metrics = yaml.safe_load(f)
                    state['generations_slow_graph']['model_new_score'] = new_metrics.get('model_new_score', {})
                    
                print("Loaded metrics from Fast Graph execution files")
            except Exception as e:
                print(f"Error loading Fast Graph metrics files: {str(e)}")
                
        return state
    
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
        """Updated distill_memories to handle missing semantic memory and leverage fast graph results."""
        # Initialize if not already done
        if 'generations_slow_graph' not in state:
            state['generations_slow_graph'] = self.initialize_generations()
        
        # Check if semantic memory exists
        if 'semantic_memory' not in state or state['semantic_memory'] is None:
            print("\n⚠️ Missing semantic memory in distill_memories - creating fallback")
            # Create fallback objects to prevent errors
            state['generations_slow_graph'].update({
                'distilled_insights': {'error': 'Missing semantic memory'},
                'model_metadata': {
                    'params_summary': 'No model parameters available',
                    'data_paths': {
                        'old_data': 'unknown',
                        'new_data': 'unknown'
                    },
                    'base_code': state['generations_slow_graph'].get('fast_graph_code', '')
                }
            })
            return state
        
        # Prepare semantic memory inputs with error handling
        semantic_memory = state['semantic_memory']
        
        # Get model parameters summary with fallback
        try:
            if hasattr(semantic_memory, 'model_object') and semantic_memory.model_object is not None:
                model_params_summary = (prompt_summarize_model_docs() | self.llm).invoke(
                    {'input': semantic_memory.model_object.__doc__}
                ).content
            else:
                print("\n⚠️ Missing model_object in semantic_memory")
                model_params_summary = "No model parameters available"
        except Exception as e:
            print(f"\n⚠️ Error getting model parameters: {str(e)}")
            model_params_summary = f"Error in model parameters: {str(e)}"
        
        # Determine which code to use as base - fast graph code if available, otherwise original
        try:
            base_code = state['generations_slow_graph'].get('fast_graph_code', '')
            if not base_code and hasattr(semantic_memory, 'model_code'):
                base_code = semantic_memory.model_code
            
            if not base_code:
                print("\n⚠️ No base code found in either fast_graph or semantic_memory")
                base_code = "# No base code available"
        except Exception as e:
            print(f"\n⚠️ Error getting base code: {str(e)}")
            base_code = f"# Error getting base code: {str(e)}"
        
        # Build focused input YAML with required components
        yaml_content = {}
        
        # Get data paths with error handling
        old_data_path = "unknown"
        new_data_path = "unknown"
        
        try:
            if hasattr(semantic_memory, 'dataset_old') and hasattr(semantic_memory.dataset_old, 'X_train'):
                old_data_path = semantic_memory.dataset_old.X_train
                
            if 'episodic_memory' in state and state['episodic_memory'] and len(state['episodic_memory']) > 0:
                episodic = state['episodic_memory'][-1]
                if hasattr(episodic, 'dataset_new') and hasattr(episodic.dataset_new, 'X_train'):
                    new_data_path = episodic.dataset_new.X_train
        except Exception as e:
            print(f"\n⚠️ Error accessing data paths: {str(e)}")
        
        # If fast graph has been run, include its metrics and execution output
        if state['generations_slow_graph'].get('fast_graph_integrated', False):
            # Use fast graph performance data
            fast_metrics = state['generations_slow_graph'].get('fast_graph_metrics', {})
            execution_output = state['generations_slow_graph'].get('execution_output', '')
            
            # Format might be different depending on where we got the metrics from
            # Check and adapt to both possible structures
            if 'old_model' in fast_metrics and 'new_model' in fast_metrics:
                # Standard structure from improvement_history
                model_metrics = fast_metrics
            else:
                # Structure from generations_fast_graph
                model_metrics = {
                    'old_model': state['generations_slow_graph'].get('model_old_score', {}),
                    'new_model': state['generations_slow_graph'].get('model_new_score', {})
                }
            
            # Get model type with fallback
            model_type = "Unknown"
            if hasattr(semantic_memory, 'model_object') and semantic_memory.model_object is not None:
                model_type = semantic_memory.model_object.__class__.__name__
            
            yaml_content = {
                'execution_output': execution_output,
                'model_code': base_code,
                'fast_graph_metrics': model_metrics,
                'model_type': model_type
            }
            
            print("\nDistilling insights from Fast Graph results")
        else:
            # Use original episodic memory output if no fast graph data
            execution_output = ""
            if 'episodic_memory' in state and state['episodic_memory'] and len(state['episodic_memory']) > 0:
                episodic = state['episodic_memory'][-1]
                if hasattr(episodic, 'quick_insight'):
                    execution_output = episodic.quick_insight.get('execution_output', '')
            
            yaml_content = {
                'execution_output': execution_output,
                'model_code': base_code
            }
            
            print("\nDistilling insights from scratch (no Fast Graph results)")
        
        # Generate insights with core data
        try:
            chain = prompt_distill_memories() | self.llm
            output = chain.invoke({'input': yaml.dump(yaml_content)}).content
            
            # Store structured results
            try:
                insights = yaml.safe_load(output)
            except yaml.YAMLError:
                insights = {'error': 'Failed to parse insights YAML'}
        except Exception as e:
            print(f"\n⚠️ Error generating insights: {str(e)}")
            insights = {'error': f'Error generating insights: {str(e)}'}
        
        # Update state with distilled insights and metadata
        state['generations_slow_graph'].update({
            'distilled_insights': insights,
            'model_metadata': {
                'params_summary': model_params_summary,
                'data_paths': {
                    'old_data': old_data_path,
                    'new_data': new_data_path
                },
                'base_code': base_code
            }
        })
        
        return state
        
    def analyze_needs(self, state: WorkingMemory) -> WorkingMemory:
        """Analyzes current model performance and determines next best strategy,
        considering fast graph results when available."""
        
        state['generations_slow_graph']['execution_attempts'] = 0
        
        # Get current state information
        execution_output = state['generations_slow_graph'].get('execution_output', '')
        strategy_results = state['generations_slow_graph'].get('strategy_results', {})
        improvement_history = state.get('improvement_history', [])
        fast_graph_integrated = state['generations_slow_graph'].get('fast_graph_integrated', False)
        
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
        
        # NEW: If fast graph results are available, include them for better strategy selection
        if fast_graph_integrated:
            yaml_content["fast_graph_metrics"] = state['generations_slow_graph'].get('fast_graph_metrics', {})
            yaml_content["fast_graph_improved"] = True
            
            # Add distribution gap information from fast graph
            fast_metrics = state['generations_slow_graph'].get('fast_graph_metrics', {})
            if 'new_model' in fast_metrics and 'old_model' in fast_metrics:
                new_model = fast_metrics['new_model']
                old_model = fast_metrics['old_model']
                
                # Calculate gaps in fast graph improvement
                new_dist_gap = new_model.get('on_new_data', 0) - old_model.get('on_new_data', 0)
                old_dist_gap = new_model.get('on_old_data', 0) - old_model.get('on_old_data', 0)
                
                yaml_content["distribution_gaps"] = {
                    "new_distribution": float(new_dist_gap),
                    "old_distribution": float(old_dist_gap)
                }
        
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
        
        # Get suggested strategy from analysis
        recommended_strategy = parsed_analysis.get('recommended_strategy', None)
        
        # If no clear recommendation, infer from context
        if not recommended_strategy:
            # If fast graph showed significant improvement on new distribution
            # but slight regression on old, prioritize hyperparameter tuning
            if fast_graph_integrated:
                distrib_gaps = yaml_content.get("distribution_gaps", {})
                if distrib_gaps.get("new_distribution", 0) > 0.05 and distrib_gaps.get("old_distribution", 0) < 0:
                    recommended_strategy = "hyperparameter_tuning"
                elif distrib_gaps.get("new_distribution", 0) > 0.1:
                    # If large improvement on new distribution, try ensemble to balance
                    recommended_strategy = "ensemble_method"
                else:
                    # Default to model selection for other cases
                    recommended_strategy = "model_selection"
            else:
                # Default strategy ordering if no fast graph results
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
        print(f"Fast Graph Integration: {'Yes' if fast_graph_integrated else 'No'}")
        
        # Print next steps and tried strategies
        next_steps = parsed_analysis.get('next_steps', [])
        print(f"Next Steps: {next_steps}")
        print(f"Strategies Tried: {[s for s, r in strategy_results.items() if r.get('tried', False)]}")
        
        return state
    
    def generate_model_selection_change(self, state: WorkingMemory) -> WorkingMemory:
        """Generates changes focused on trying different model architectures,
        using fast graph code as base when available."""
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
            
            # Use fast graph code as current code if available and not already used
            current_code = state['generations_slow_graph'].get('model_metadata', {}).get('base_code', '')
            if not current_code:
                current_code = state['semantic_memory'].model_code
            
            # Prepare input YAML
            yaml_content = {
                "current_code": current_code,
                "execution_output": state['generations_slow_graph'].get('execution_output', ''),
                "models_tried": models_tried,
                "previous_metrics": prev_metrics if prev_metrics else {}
            }
            
            # Add fast graph information if available
            if state['generations_slow_graph'].get('fast_graph_integrated', False):
                yaml_content["fast_graph_improved"] = True
                yaml_content["fast_graph_metrics"] = state['generations_slow_graph'].get('fast_graph_metrics', {})
            
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
        
        # Check if we've hit max consecutive failures
        if state['generations_slow_graph'].get('consecutive_failures', 0) >= self.max_failures:
            print(f"Maximum consecutive failures ({self.max_failures}) reached. Ending process.")
            return 'end'
        
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
        
        # Get current iteration count
        iterations = generations.get('iteration_count', 0)
        generations['iteration_count'] = iterations + 1  # Increment for next time
        
        # Critical termination: check if we've reached max iterations
        if iterations >= self.max_iterations:
            print(f"Reached maximum iterations ({iterations}/{self.max_iterations})")
            return 'end'
    

        # Performance analysis: check if there was improvement
        new_dist_improvement = improvements.get('new_distribution', 0)
        old_dist_improvement = improvements.get('old_distribution', 0)
        
        # If both distributions got worse, stop iterations
        if new_dist_improvement < 0 and old_dist_improvement < 0:
            print("Performance degraded on both distributions, stopping iterations")
            return 'end'
        
        # Strategy exhaustion check
        all_strategies_tried = len(strategies_tried) >= 3
        current_strategy_exhausted = self._is_strategy_exhausted(
            strategy_results.get(current_strategy, {}),
            latest_improvement
        )
        
        # If all strategies tried and current is exhausted, stop
        if all_strategies_tried and current_strategy_exhausted:
            print("All strategies tried and current strategy exhausted")
            return 'end'
        
        # Decision based on current performance
        if latest_improvement['outcome'] == 'success':
            # Continue if we had success and have more iterations available
            print(f"Successful improvement, continuing ({iterations}/{self.max_iterations})")
            return 'continue'
        else:
            # For unsuccessful outcomes, check if we have untried strategies
            if len(strategies_tried) < 3:
                print("Unsuccessful improvement but still have untried strategies")
                return 'continue'
            else:
                print("Unsuccessful improvement and all strategies tried")
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
        
    def _extract_strategy_heuristic(self, text: str) -> Dict:
        """Extract strategy information using heuristics when YAML parsing fails."""
        result = {
            "recommended_strategy": "model_selection",  # Default
            "reasoning": "Fallback reasoning due to parsing error",
            "next_steps": []
        }
        
        # Look for strategy mentions
        strategies = {
            "model_selection": ["model selection", "different model", "model architecture"],
            "hyperparameter_tuning": ["hyperparameter", "tuning", "parameters"],
            "ensemble_method": ["ensemble", "voting", "stacking", "combine models"]
        }
        
        for strategy, keywords in strategies.items():
            if any(keyword in text.lower() for keyword in keywords):
                result["recommended_strategy"] = strategy
                break
        
        # Extract next steps
        step_markers = ["next steps", "recommended steps", "next actions"]
        for marker in step_markers:
            if marker in text.lower():
                steps_section = text.lower().split(marker)[1].split("\n\n")[0]
                steps = re.findall(r'[-*] (.*?)(?:\n|$)', steps_section)
                if steps:
                    result["next_steps"] = steps
                    break
        
        return result
        
    def apply_change(self, state: WorkingMemory) -> WorkingMemory:
        """Apply the generated change and execute the code with better error handling."""
        try:
            # Extract and parse code from tiny_change
            current_code = self._extract_code(state['generations_slow_graph']['tiny_change'])
            max_retries = min(self.max_failures, 3)  # Use the smaller of max_failures or 3
            current_try = 0
            
            while current_try < max_retries:
                # Execute the code
                execution_output = self._execute_code(current_code)
                
                # Check for execution errors
                if not self._has_execution_errors(execution_output):
                    # Reset consecutive failures on success
                    state['generations_slow_graph']['consecutive_failures'] = 0
                    break
                    
                # Increment failure count
                state['generations_slow_graph']['consecutive_failures'] = state['generations_slow_graph'].get('consecutive_failures', 0) + 1
                
                # Log failure information
                print(f"⚠️ Execution failed. Attempt {current_try + 1}/{max_retries}")
                print(f"⚠️ Consecutive failures: {state['generations_slow_graph']['consecutive_failures']}/{self.max_failures}")
                
                # Handle code fixing if needed and under the limit
                if current_try < max_retries - 1:
                    print("🔧 Attempting to fix code...")
                    current_code = self._fix_code(current_code, execution_output, state)
                
                current_try += 1
            
            # Check if we've reached max failures
            if state['generations_slow_graph']['consecutive_failures'] >= self.max_failures:
                print(f"❌ Reached maximum consecutive failures ({self.max_failures}). Stopping execution attempts.")
                
                # Use the last successful state if available
                if state['generations_slow_graph'].get('last_successful_state'):
                    print("📥 Restoring last successful state...")
                    # Update metrics but keep the consecutive_failures count
                    last_successful = state['generations_slow_graph']['last_successful_state']
                    for key, value in last_successful.items():
                        if key not in ['consecutive_failures']:
                            state['generations_slow_graph'][key] = value
                    
                    # Add note to execution output
                    state['generations_slow_graph']['execution_output'] = execution_output + f"\n\nReached maximum failures ({self.max_failures}). Restored last successful state."
                    return state
            
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
            timeout=60,
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
        """Process execution results and update improvement history with enhanced metrics tracking."""
        print("Execution Output:", "-"*100)
        print(execution_output)
        
        # Store raw output
        state['generations_slow_graph']['execution_output'] = execution_output
        
        # Extract metrics from both YAML files
        try:
            # Read old model metrics
            with open('old_metrics.yaml', 'r') as f:
                old_metrics = yaml.safe_load(f)
                baseline_old_model_score = old_metrics.get('model_old_score', {})
            
            # Use Fast Graph metrics as baseline if available and better than old model
            fast_graph_metrics = state['generations_slow_graph'].get('fast_graph_metrics', {})
            if fast_graph_metrics and 'new_model' in fast_graph_metrics:
                fast_graph_score = fast_graph_metrics['new_model']
                # Compare Fast Graph vs baseline and use the better one as comparison baseline
                if (fast_graph_score.get('on_new_data', 0) >= baseline_old_model_score.get('on_new_data', 0) and
                    fast_graph_score.get('on_old_data', 0) >= baseline_old_model_score.get('on_old_data', 0)):
                    old_model_score = fast_graph_score
                    print("Using Fast Graph metrics as baseline for comparison")
                else:
                    old_model_score = baseline_old_model_score
                    print("Using original old model metrics as baseline")
            else:
                old_model_score = baseline_old_model_score
                print("No Fast Graph metrics available, using original baseline")
            
            # Read new model metrics - prioritize the slow graph metric file if available
            if os.path.exists('slow_graph_metrics.yaml'):
                with open('slow_graph_metrics.yaml', 'r') as f:
                    new_metrics = yaml.safe_load(f)
                    new_model_score = new_metrics.get('model_new_score', {})
            elif os.path.exists('fast_graph_metrics.yaml'):
                with open('fast_graph_metrics.yaml', 'r') as f:
                    new_metrics = yaml.safe_load(f)
                    new_model_score = new_metrics.get('model_new_score', {})
            else:
                raise FileNotFoundError("No metrics files found for new model")
                    
            # Get current strategy and changes information
            current_strategy = state['generations_slow_graph'].get('current_strategy')
            strategy_results = state['generations_slow_graph'].get('strategy_results', {})
            
            # Prepare changes made dictionary based on strategy
            changes_made = {
                'strategy': current_strategy,
                'iteration_count': state['generations_slow_graph'].get('iteration_count', 0),
                'based_on_fast_graph': state['generations_slow_graph'].get('fast_graph_integrated', False)
            }
            
            # Add strategy-specific changes
            if current_strategy == 'model_selection':
                changes_made.update({'models_tried': strategy_results.get('model_selection', {}).get('models_tried', [])})
            elif current_strategy == 'hyperparameter_tuning':
                changes_made.update({'parameters': strategy_results.get('hyperparameter_tuning', {}).get('best_params', {})})
            elif current_strategy == 'ensemble_method':
                changes_made.update({'ensemble_type': strategy_results.get('ensemble_method', {}).get('best_ensemble', '')})
            
            # Create improvement entry
            iteration_count = state['generations_slow_graph'].get('iteration_count', 1)
            improvement_entry = create_improvement_entry(
                previous_code=state['semantic_memory'].model_code,
                new_code=current_code,
                graph_type='slow',
                strategy_type=current_strategy,
                old_model_score=old_model_score,
                new_model_score=new_model_score,
                changes_made=changes_made,
                iteration=iteration_count
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
            
            # Save current state as last successful state
            state['generations_slow_graph']['last_successful_state'] = {
                'execution_success': True,
                'model_new_score': new_model_score,
                'model_old_score': old_model_score,
                'tiny_change': state['generations_slow_graph'].get('tiny_change', ''),
                'current_strategy': current_strategy
            }
            
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
            # Check methodology validation first
            methodology_check = evaluation_result.get('methodology_check', {})
            valid_evaluation = methodology_check.get('valid_evaluation', True)  # Default to True for backward compatibility
            
            recommendation = evaluation_result.get('recommendation', {})
            action = recommendation.get('action', 'reject')  # Default to reject if not found
            
            # If methodology is invalid, force rejection regardless of performance
            if not valid_evaluation:
                print(f"WARNING: Invalid evaluation methodology detected. Forcing rejection.")
                print(f"Issues found: {methodology_check.get('issues_found', [])}")
                action = 'reject'
                
                # Update recommendation in evaluation result
                evaluation_result['recommendation'] = {
                    **recommendation,
                    'action': 'reject',
                    'confidence': 'high',
                    'reasoning': f"Invalid evaluation methodology: {methodology_check.get('issues_found', [])}"
                }
            
            updated_improvement = {
                **latest_improvement,
                'evaluation': evaluation_result,
                'final_outcome': action,
                'methodology_valid': valid_evaluation  # Track methodology validity
            }
            
            # Replace the latest improvement with updated version
            state['improvement_history'][-1] = updated_improvement
            
            # Update strategy results
            current_strategy = state['generations_slow_graph'].get('current_strategy')
            if current_strategy:
                strategy_results = state['generations_slow_graph'].get('strategy_results', {})
                if current_strategy in strategy_results:
                    strategy_results[current_strategy]['tried'] = True
                    
                    # Track methodology issues at strategy level
                    if not valid_evaluation:
                        methodology_issues = strategy_results[current_strategy].get('methodology_issues', [])
                        methodology_issues.extend(methodology_check.get('issues_found', []))
                        strategy_results[current_strategy]['methodology_issues'] = methodology_issues
                    
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
            
            # Log methodology validation results
            if not valid_evaluation:
                print(f"❌ Methodology validation failed: {methodology_check.get('issues_found', [])}")
            else:
                print(f"✅ Methodology validation passed")
                
            print(f"Final recommendation: {action}")
            
        except Exception as e:
            print(f"Error updating improvement entry: {str(e)}")
            # Create a minimal evaluation result if there's an error
            state['generations_slow_graph']['evaluation'] = {
                'methodology_check': {'valid_evaluation': False, 'issues_found': [f'Evaluation error: {str(e)}']},
                'recommendation': {'action': 'reject', 'confidence': 'low'},
                'analysis': [f'Error in evaluation: {str(e)}'],
                'next_steps': ['Retry with different approach']
            }
            
            # Also update the improvement history with error information
            try:
                state['improvement_history'][-1]['evaluation'] = state['generations_slow_graph']['evaluation']
                state['improvement_history'][-1]['final_outcome'] = 'reject'
                state['improvement_history'][-1]['methodology_valid'] = False
            except:
                pass  # Fail silently if we can't update improvement history
        
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
        
    def export_results_to_yaml(self, state: WorkingMemory, runtime_seconds: float) -> Dict:
        """Format results to match standardized YAML output."""
        # Get the initial code
        initial_code = state['semantic_memory'].model_code if 'semantic_memory' in state else ""
        
        # Get final code from the last improvement entry
        final_code = initial_code
        if state.get('improvement_history'):
            final_code = state['improvement_history'][-1].get('new_code', initial_code)
        
        # Extract initial metrics
        initial_metrics = {}
        if 'model_old_score' in state.get('generations_slow_graph', {}):
            old_model_score = state['generations_slow_graph']['model_old_score']
            initial_metrics = {
                "old_distribution": old_model_score.get("on_old_data", 0),
                "new_distribution": old_model_score.get("on_new_data", 0)
            }
        
        # Extract final metrics
        final_metrics = {}
        if 'model_new_score' in state.get('generations_slow_graph', {}):
            new_model_score = state['generations_slow_graph']['model_new_score']
            final_metrics = {
                "old_distribution": new_model_score.get("on_old_data", 0),
                "new_distribution": new_model_score.get("on_new_data", 0)
            }
        
        # Group improvement history by iteration
        iteration_map = {}
        max_iteration = self.max_iterations
        
        if state.get('improvement_history'):
            # First, group entries by iteration number
            for entry in state['improvement_history']:
                # Get the iteration this entry belongs to
                # In the new implementation, each entry should have an iteration field
                iteration = min(entry.get('iteration', 1), max_iteration)
                
                if iteration not in iteration_map:
                    iteration_map[iteration] = []
                
                iteration_map[iteration].append(entry)
        
        # Build improvement path with one entry per iteration (best result from that iteration)
        improvement_path = []
        
        for iteration in range(1, max_iteration + 1):
            if iteration not in iteration_map:
                continue
                
            entries = iteration_map[iteration]
            if not entries:
                continue
                
            # Use the last entry from each iteration as the representative
            best_entry = entries[-1]
            
            # Extract metrics
            metrics = best_entry.get('metrics', {})
            new_model = metrics.get('new_model', {})
            
            # Validate metrics are numeric
            old_dist = new_model.get("on_old_data", 0)
            new_dist = new_model.get("on_new_data", 0)
            try:
                old_dist = float(old_dist)
                new_dist = float(new_dist)
            except (ValueError, TypeError):
                old_dist = 0.0
                new_dist = 0.0
            
            # Extract changes
            changes = []
            changes_made = best_entry.get('changes_made', {})
            
            if 'strategy' in changes_made:
                changes.append(f"Applied {changes_made['strategy']} strategy")
                
            # Add strategy-specific changes
            strategy_type = best_entry.get('strategy_type', '')
            if strategy_type == 'model_selection':
                models = changes_made.get('models_tried', [])
                if models:
                    model_name = models[-1] if models else 'unknown model'
                    changes.append(f"Changed model to {model_name}")
            elif strategy_type == 'hyperparameter_tuning':
                params = changes_made.get('parameters', {})
                if params:
                    params_str = ', '.join([f"{k}={v}" for k, v in params.items()])
                    changes.append(f"Tuned hyperparameters: {params_str}")
            elif strategy_type == 'ensemble_method':
                ensemble_type = changes_made.get('ensemble_type', '')
                if ensemble_type:
                    changes.append(f"Applied {ensemble_type} ensemble method")
                    
            # Get iteration time if available
            iteration_time = 0
            if 'iteration_times' in state['generations_slow_graph']:
                for time_entry in state['generations_slow_graph']['iteration_times']:
                    if time_entry['iteration'] == iteration:
                        iteration_time = time_entry['time']
                        break
                    
            # Create path entry
            path_entry = {
                "iteration": iteration,
                "code": best_entry.get('new_code', ''),
                "metrics": {
                    "old_distribution": old_dist,
                    "new_distribution": new_dist
                },
                "changes": changes,
                "reflection": f"Strategy: {strategy_type}\nEvaluation: {best_entry.get('evaluation', {})}",
                "execution_time": iteration_time
            }
            improvement_path.append(path_entry)
        
        # If no improvement path, create minimal entry
        if not improvement_path and final_metrics:
            improvement_path = [{
                "iteration": 1,
                "code": final_code,
                "metrics": final_metrics,
                "changes": ["Used best strategy from analysis"],
                "reflection": "No detailed evaluations available",
                "execution_time": 0
            }]
                
        # Calculate the actual number of iterations that were run
        completed_iterations = max(iteration_map.keys()) if iteration_map else 0
        
        # Get token usage from state or use class counter as fallback
        token_usage = state['generations_slow_graph'].get('token_usage', {
            "prompt": self.token_counts.get("prompt", 0),
            "completion": self.token_counts.get("completion", 0),
            "total": self.token_counts.get("total", 0)
        })
        
        # UPDATED: Better token calculation using new helper methods
        token_count = self._estimate_total_tokens(state, final_code)
                
        # Build the standardized output
        output = {
            "agent_name": "slow",
            "initial_code": initial_code,
            "initial_metrics": initial_metrics,
            "improvement_path": improvement_path,
            "final_code": final_code,
            "final_metrics": final_metrics,
            "runtime_statistics": {
                "total_time_seconds": runtime_seconds,
                "iterations": completed_iterations,
                "tokens_used": token_count,
                "prompt_tokens": token_usage.get("prompt", 0),
                "completion_tokens": token_usage.get("completion", 0),
                "iteration_times": state['generations_slow_graph'].get('iteration_times', []),
                "evaluation_timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        # Check if we should revert to Fast Graph metrics if they're better
        fast_graph_metrics = state['generations_slow_graph'].get('fast_graph_metrics', {})
        if fast_graph_metrics and 'new_model' in fast_graph_metrics:
            fast_graph_score = fast_graph_metrics['new_model']
            fast_graph_final_metrics = {
                "old_distribution": fast_graph_score.get("on_old_data", 0),
                "new_distribution": fast_graph_score.get("on_new_data", 0)
            }
            
            # Compare Fast Graph vs current final metrics
            current_total = final_metrics.get("old_distribution", 0) + final_metrics.get("new_distribution", 0)
            fast_graph_total = fast_graph_final_metrics.get("old_distribution", 0) + fast_graph_final_metrics.get("new_distribution", 0)
            
            if fast_graph_total > current_total:
                print(f"Reverting to Fast Graph metrics: Fast Graph total={fast_graph_total:.4f} > Slow Graph total={current_total:.4f}")
                output["final_metrics"] = fast_graph_final_metrics
                output["final_code"] = state['generations_slow_graph'].get('fast_graph_code', final_code)
                output["reverted_to_fast_graph"] = True
            else:
                print(f"Keeping Slow Graph results: Slow Graph total={current_total:.4f} >= Fast Graph total={fast_graph_total:.4f}")
                output["reverted_to_fast_graph"] = False
        
        return output

    def _estimate_total_tokens(self, state: WorkingMemory, final_code: str) -> int:
        """Estimate total tokens used throughout the slow graph execution."""
        # Start with a base token count for final code (using improved estimation)
        token_count = len(final_code) // 3  # Better character-to-token ratio for code
        
        # Add tokens for all prompt completions and generations
        if 'generations_slow_graph' in state:
            generations = state['generations_slow_graph']
            
            # Count tokens for insights, analysis, and other text generations
            for key, value in generations.items():
                if isinstance(value, str):
                    token_count += len(value) // 4  # Estimate for text
                elif isinstance(value, dict):
                    # Estimate tokens for nested dictionaries
                    token_count += self._estimate_dict_tokens(value)
            
            # Add tokens for any execution outputs
            if 'execution_output' in generations:
                token_count += len(str(generations['execution_output'])) // 4
        
        # Add tokens for improvement history entries
        for entry in state.get('improvement_history', []):
            # Add tokens for code changes
            if 'new_code' in entry:
                token_count += len(entry['new_code']) // 3
            
            # Add tokens for evaluation and analysis
            if 'evaluation' in entry:
                token_count += self._estimate_dict_tokens(entry['evaluation'])
        
        return token_count

    def _estimate_dict_tokens(self, data: dict) -> int:
        """Recursively estimate tokens in a nested dictionary structure."""
        token_count = 0
        
        if not isinstance(data, dict):
            return len(str(data)) // 4
            
        for key, value in data.items():
            # Count the key
            token_count += len(str(key)) // 4
            
            # Count the value based on type
            if isinstance(value, str):
                token_count += len(value) // 4
            elif isinstance(value, dict):
                token_count += self._estimate_dict_tokens(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        token_count += len(item) // 4
                    elif isinstance(item, dict):
                        token_count += self._estimate_dict_tokens(item)
                    else:
                        # Handle numbers, booleans, etc.
                        token_count += len(str(item)) // 4
            else:
                # Handle numbers, booleans, etc.
                token_count += len(str(value)) // 4
        
        return token_count
        
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

    def _should_terminate_after_iteration(self, state: WorkingMemory) -> bool:
        """Determine if we should terminate after the current iteration."""
        # Check if we have improvement history
        improvement_history = state.get('improvement_history', [])
        if len(improvement_history) < 2:
            return False  # Not enough history to make a decision
        
        # Get the last two improvement entries
        current = improvement_history[-1]
        previous = improvement_history[-2]
        
        # Compare performance metrics
        current_metrics = current.get('metrics', {}).get('new_model', {})
        previous_metrics = previous.get('metrics', {}).get('new_model', {})
        
        # Calculate changes in performance
        new_dist_change = current_metrics.get('on_new_data', 0) - previous_metrics.get('on_new_data', 0)
        old_dist_change = current_metrics.get('on_old_data', 0) - previous_metrics.get('on_old_data', 0)
        
        # Terminate if both metrics got worse
        if new_dist_change < -0.01 and old_dist_change < -0.01:
            return True
        
        # Terminate if no significant improvement (less than 0.5%)
        if abs(new_dist_change) < 0.005 and abs(old_dist_change) < 0.005:
            return True
        
        # Don't terminate otherwise
        return False

    def run(self, initial_state: WorkingMemory):
        """Run the slow improvement process with enhanced logging and standardized output."""
        self.start_time = datetime.now()
        
        # Get max_iterations from working memory if available
        if "max_iterations" in initial_state:
            self.max_iterations = initial_state["max_iterations"]
            print(f"Max iterations set to: {self.max_iterations}")
        else:
            print(f"Using default max iterations: {self.max_iterations}")
            
        # Get max_failures from working memory if available
        if "max_failures" in initial_state:
            self.max_failures = initial_state["max_failures"]
            print(f"Max consecutive failures set to: {self.max_failures}")
        else:
            print(f"Using default max failures: {self.max_failures}")
        
        # Initialize generations dictionary if not present
        if 'generations_slow_graph' not in initial_state:
            initial_state['generations_slow_graph'] = self.initialize_generations()
        
        # Set up tracking for iterations at the graph level (not node level)
        current_state = initial_state
        output_keys = ['generations_slow_graph', 'improvement_history']
        final_state = None
        
        # Run for specified number of iterations
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n\n{'='*20} STARTING ITERATION {iteration}/{self.max_iterations} {'='*20}\n")
            
            current_iteration_output = None
            iteration_start_time = datetime.now()
            try:
                # Run one complete pass through the decision procedure
                for output in self.decision_procedure.stream(
                    current_state, 
                    output_keys=output_keys, 
                    debug=False
                ):
                    current_iteration_output = output
                    
                    # Log the current node execution
                    for node_name, state in output.items():
                        # Log node execution
                        print(f"\nExecuting Node: {node_name}", "="*50)
                        
                        # Log generations updates
                        self._log_generations_updates(state)
                        
                        # Log improvement history updates
                        self._log_improvement_history(state)
                        
                        # Log strategy progress
                        self._log_strategy_progress(state)
                
                # Store the final state of this iteration for the next iteration
                if current_iteration_output:
                    final_key = list(current_iteration_output.keys())[-1]
                    current_state = current_iteration_output[final_key]
                    final_state = current_state  # Keep track of the very last state
                    
                    # Add iteration-specific information
                    if 'generations_slow_graph' in current_state:
                        current_state['generations_slow_graph']['iteration_count'] = iteration
                        
                        # Record iteration time
                        iteration_end_time = datetime.now()
                        iteration_time = (iteration_end_time - iteration_start_time).total_seconds()
                        
                        if 'iteration_times' not in current_state['generations_slow_graph']:
                            current_state['generations_slow_graph']['iteration_times'] = []
                            
                        current_state['generations_slow_graph']['iteration_times'].append({
                            'iteration': iteration,
                            'time': iteration_time
                        })
                        
                        print(f"\nIteration {iteration} time: {iteration_time:.2f} seconds")
                    
                    # Check if we should continue to the next iteration
                    if self._should_terminate_after_iteration(current_state):
                        print(f"\nTerminating after iteration {iteration} due to convergence or no improvement")
                        break
            
            except Exception as e:
                print(f"Error in iteration {iteration}: {str(e)}")
                import traceback
                traceback.print_exc()
                break
        
        # Calculate runtime
        end_time = datetime.now()
        runtime_seconds = (end_time - self.start_time).total_seconds()
        
        # Generate standardized YAML output
        if final_state:
            yaml_output = self.export_results_to_yaml(final_state, runtime_seconds)
            final_state['yaml_output'] = yaml_output
            return final_state
        
        # If no final state, return last output with minimal YAML
        if current_iteration_output:
            minimal_yaml = {
                "agent_name": "slow_improvement",
                "initial_code": initial_state['semantic_memory'].model_code if 'semantic_memory' in initial_state else "",
                "initial_metrics": {"old_distribution": 0, "new_distribution": 0},
                "improvement_path": [],
                "final_code": initial_state['semantic_memory'].model_code if 'semantic_memory' in initial_state else "",
                "final_metrics": {"old_distribution": 0, "new_distribution": 0},
                "runtime_statistics": {
                    "total_time_seconds": runtime_seconds,
                    "iterations": 0,
                    "tokens_used": 0,
                    "evaluation_timestamp": datetime.utcnow().isoformat() + "Z"
                }
            }
            for key in current_iteration_output:
                if isinstance(current_iteration_output[key], dict):
                    current_iteration_output[key]['yaml_output'] = minimal_yaml
            
            return current_iteration_output
        
        return initial_state


class EnhancedStateGraph(StateGraph):
    """Enhanced StateGraph with function name printing capability."""
    def add_node(self, node_name, function):
        decorated_function = print_function_name(function)
        super().add_node(node_name, decorated_function)