import textwrap
import yaml
from time import sleep
from rich import print
from rich.panel import Panel
from rich.text import Text
from langgraph.graph import StateGraph, END
from IPython.display import Image, display
from caia.memory import WorkingMemory
from caia.utils import print_function_name
from caia.prompts import (
    # prompt_measure_criticality,
    prompt_generate_retraining_code,
    prompt_generate_retraining_code_with_insights,  # Added new prompt function
    prompt_execute_and_fix_retraining_code
)
from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor
from typing import TypedDict, Dict, Optional
from caia.memory import create_improvement_entry
from datetime import datetime

class FastGraph:
    def __init__(self, llm, debug=False):
        self.llm = llm
        self.graph = EnhancedStateGraph(WorkingMemory)
        self.build_plan()
        self.decision_procedure = self.graph.compile(debug=debug)
        self.start_time = None
        
    def draw_graph(self):
        return display(Image(self.decision_procedure.get_graph().draw_mermaid_png()))
        
    def build_plan(self):
        # Add nodes
        # self.graph.add_node('measure_criticality', self.measure_criticality)
        self.graph.add_node('generate_retraining_code', self.generate_retraining_code)
        self.graph.add_node('execute_retraining_code', self.execute_retraining_code)
        self.graph.add_node('fix_retraining_code', self.fix_retraining_code)
        
        # Set entry point
        self.graph.set_entry_point('generate_retraining_code')
        # self.graph.set_entry_point('measure_criticality')
        
        # self.graph.add_edge('measure_criticality', 'generate_retraining_code')
        # self.graph.add_edge('measure_model_retraining_speed', 'generate_retraining_code')
        self.graph.add_edge('generate_retraining_code', 'execute_retraining_code')
        
        # Add conditional edge for execute_retraining_code
        self.graph.add_conditional_edges(
            'execute_retraining_code',
            self.should_fix_code,
            {
                'fix': 'fix_retraining_code',
                'end': END
            }
        )
        
        # Add edge from fix_retraining_code back to execute_retraining_code
        self.graph.add_edge('fix_retraining_code', 'execute_retraining_code')
    
    def generate_retraining_code(self, state: WorkingMemory) -> WorkingMemory:
        """Generate retraining code, utilizing insights from slow graph if available."""
        try:
            episodic_memory = state['episodic_memory']
            semantic_memory = state['semantic_memory']
            
            if not semantic_memory.model_code:
                raise ValueError("No model code found in semantic memory")
            
            # Initialize generations dictionary if not present
            if 'generations_fast_graph' not in state:
                state['generations_fast_graph'] = {}
                
            # Check if we have slow graph insights to leverage
            has_slow_graph_insights = (
                'generations_slow_graph' in state and 
                isinstance(state['generations_slow_graph'], dict) and 
                'yaml_output' in state['generations_slow_graph']
            )
            
            # Get the appropriate training code
            training_code = semantic_memory.model_code
            dataset_folder = "/".join(semantic_memory.dataset_old.X_train.split("/")[:2])
            new_data = (
                f'X_train_new = pd.read_csv(f"{dataset_folder}/X_train_new.csv")\n'
                f'X_test_new = pd.read_csv(f"{dataset_folder}/X_test_new.csv")\n'
                f'y_train_new = pd.read_csv(f"{dataset_folder}/y_train_new.csv").squeeze("columns")\n'
                f'y_test_new = pd.read_csv(f"{dataset_folder}/y_test_new.csv").squeeze("columns")\n'
            )
            
            # Choose the appropriate prompt based on availability of slow graph insights
            if has_slow_graph_insights:
                print("Using insights from slow graph to enhance retraining code generation")
                # Get the improved model code from slow graph output
                improved_model_code = state['generations_slow_graph']['yaml_output']['final_code']
                
                # Prepare YAML content with improved model code
                yaml_content = self._prepare_yaml_content_with_insights(training_code, improved_model_code)
                
                # Use the enhanced prompt
                prompt = prompt_generate_retraining_code_with_insights()
                state['generations_fast_graph']['has_slow_graph_insights'] = True
            else:
                print("No slow graph insights available, using basic retraining approach")
                # Prepare basic YAML content
                yaml_content = self._prepare_yaml_content(training_code, new_data)
                
                # Use the standard prompt
                prompt = prompt_generate_retraining_code()
                state['generations_fast_graph']['has_slow_graph_insights'] = False
            
            # Invoke the LLM
            chain = prompt | self.llm
            output = chain.invoke({'input': yaml_content}).content
            
            # Validate output format
            try:
                yaml.safe_load(output)
            except yaml.YAMLError:
                print("Warning: Generated code is not valid YAML format")
                
            # Update the state
            state['generations_fast_graph']['new_training_code'] = output
            
            return state
            
        except Exception as e:
            print(f"Error in generate_retraining_code: {str(e)}")
            state['generations_fast_graph']['error'] = str(e)
            return state
            
    def _prepare_yaml_content_with_insights(self, training_code: str, improved_model_code: str) -> str:
        """Helper method to prepare YAML content with improved model code from slow graph."""
        # Clean and indent the code
        cleaned_training_code = textwrap.dedent(training_code).strip()
        indented_training_code = textwrap.indent(cleaned_training_code, '  ')
        
        cleaned_improved_code = textwrap.dedent(improved_model_code).strip()
        indented_improved_code = textwrap.indent(cleaned_improved_code, '  ')
        
        # Create the YAML content with old training code and improved model code
        return (
            f"old_training_code: |\n"
            f"{indented_training_code}\n"
            f"improved_model_code: |\n"
            f"{indented_improved_code}\n"
        )

    def _prepare_yaml_content(self, training_code: str, new_data: str) -> str:
        """Helper method to prepare YAML content."""
        cleaned_code = textwrap.dedent(training_code).strip()
        indented_code = textwrap.indent(cleaned_code, '  ')
        
        cleaned_new_data = textwrap.dedent(new_data).strip()
        indented_new_data = textwrap.indent(cleaned_new_data, '  ')
        
        return (
            f"old_training_code: |\n"
            f"{indented_code}\n"
            f"new_data: |\n"
            f"{indented_new_data}\n"
        )
        
    def execute_retraining_code(self, state: WorkingMemory) -> WorkingMemory:
        """Execute retraining code and track improvements"""
        # Get the current code from the most recent episodic memory
        current_code_yaml = state['generations_fast_graph'].get('fixed_code') or state['generations_fast_graph']['new_training_code']
        
        # Parse the YAML to extract the 'new_training_code' content
        parsed_yaml = yaml.safe_load(current_code_yaml)
        current_code = parsed_yaml['new_training_code']
        
        # Wrap the code in Python code block
        wrapped_code = f"```python\n{current_code}\n```"
        
        executor = LocalCommandLineCodeExecutor(
            timeout=100,
            work_dir=".",
        )
        code_executor_agent = ConversableAgent(
            "code_executor_agent",
            llm_config=False,
            code_execution_config={"executor": executor},
            human_input_mode="NEVER",
        )
        execution_output = code_executor_agent.generate_reply(messages=[{"role": "user", "content": wrapped_code}])
        
        # Store the error output in the state
        state['generations_fast_graph']['execution_output'] = execution_output
        
        # Check if execution was successful and read metrics from both files
        if "succeeded" in execution_output.lower():
            try:
                # Read old model metrics
                with open('old_metrics.yaml', 'r') as f:
                    old_metrics = yaml.safe_load(f)
                    old_model_score = old_metrics['model_old_score']
                    state['generations_fast_graph'].update(old_metrics)
                
                # Read new model metrics
                with open('fast_graph_metrics.yaml', 'r') as f:
                    new_metrics = yaml.safe_load(f)
                    new_model_score = new_metrics['model_new_score']
                    state['generations_fast_graph'].update(new_metrics)
                
                # When using slow graph insights, we need to compare against the right baseline
                # Use slow graph metrics as baseline if available and better than old model
                baseline_old_score = old_model_score
                if (state['generations_fast_graph'].get('has_slow_graph_insights', False) and 
                    'generations_slow_graph' in state and 
                    'yaml_output' in state['generations_slow_graph']):
                    slow_graph_metrics = state['generations_slow_graph']['yaml_output']['final_metrics']
                    slow_graph_baseline = {
                        'on_old_data': slow_graph_metrics.get('old_distribution', 0),
                        'on_new_data': slow_graph_metrics.get('new_distribution', 0)
                    }
                    
                    # Compare slow graph vs original baseline and use the better one
                    if (slow_graph_baseline.get('on_new_data', 0) >= baseline_old_score.get('on_new_data', 0) and
                        slow_graph_baseline.get('on_old_data', 0) >= baseline_old_score.get('on_old_data', 0)):
                        baseline_old_score = slow_graph_baseline
                        print("Using Slow Graph metrics as baseline for comparison")
                    else:
                        print("Using original old model metrics as baseline")
                
                # Create improvement entry
                improvement_entry = create_improvement_entry(
                    previous_code=state['semantic_memory'].model_code,
                    new_code=current_code,
                    graph_type='fast',
                    strategy_type=None,
                    old_model_score=baseline_old_score,
                    new_model_score=new_model_score,
                    changes_made={
                        'retrained_on_combined_data': True,
                        'iteration_count': state['generations_fast_graph'].get('iteration_count', 0),
                        'used_slow_graph_insights': state['generations_fast_graph'].get('has_slow_graph_insights', False)
                    }
                )
                
                # Initialize improvement_history if it doesn't exist
                if 'improvement_history' not in state:
                    state['improvement_history'] = []
                
                # Add the new improvement entry
                state['improvement_history'].append(improvement_entry)
                
                state['generations_fast_graph']['execution_success'] = True
                
                # Also extract metrics directly from console output as a backup verification
                extracted_metrics = self._extract_metrics_from_output(execution_output)
                if extracted_metrics:
                    state['generations_fast_graph']['extracted_metrics'] = extracted_metrics
                
            except Exception as e:
                print(f"Error processing metrics and creating improvement entry: {str(e)}")
                state['generations_fast_graph']['execution_success'] = False
        else:
            state['generations_fast_graph']['execution_success'] = False
        
        # Increment the iteration count
        state['generations_fast_graph']['iteration_count'] = state['generations_fast_graph'].get('iteration_count', 0) + 1
        
        return state
        
    def fix_retraining_code(self, state: WorkingMemory) -> WorkingMemory:
        current_code_yaml = state['generations_fast_graph'].get('fixed_code') or state['generations_fast_graph']['new_training_code']
        parsed_yaml = yaml.safe_load(current_code_yaml)
        current_code = parsed_yaml['new_training_code']
        
        execution_output = state['generations_fast_graph']['execution_output']
        
        # Create the prompt for the LLM
        prompt = prompt_execute_and_fix_retraining_code()
        
        # Prepare the input in YAML format
        cleaned_code = textwrap.dedent(current_code).strip()
        indented_code = textwrap.indent(cleaned_code, '  ')
        
        cleaned_error = textwrap.dedent(execution_output).strip()
        indented_error = textwrap.indent(cleaned_error, '  ')
        
        yaml_content = (
            f"training_code: |\n"
            f"{indented_code}\n"
            f"error_output: |\n"
            f"{indented_error}\n"
        )
        
        # Invoke the LLM
        chain = prompt | self.llm
        output = chain.invoke({'input': yaml_content}).content
        
        # Update the state with the fixed code
        state['generations_fast_graph']['fixed_code'] = output
        
        return state
        
    def should_fix_code(self, state: WorkingMemory) -> str:
        """Determine if code should be fixed or execution should end"""
        if state['generations_fast_graph']['execution_success']:
            # Before ending, check if the improvement was successful
            if state.get('improvement_history'):
                latest_improvement = state['improvement_history'][-1]
                if latest_improvement['outcome'] == 'success':
                    # Store insights in episodic memory before ending
                    state['episodic_memory'][-1].quick_insight = {
                        'execution_output': state['generations_fast_graph']['execution_output'],
                        'metrics': latest_improvement['metrics'],
                        'improvements': latest_improvement['improvements']
                    }
                    return 'end'
            return 'end'
        elif state['generations_fast_graph']['iteration_count'] >= 3:
            # If we've tried 3 times and failed, store failure information
            if state.get('improvement_history'):
                state['episodic_memory'][-1].quick_insight = {
                    'execution_output': state['generations_fast_graph']['execution_output'],
                    'failure': True,
                    'iterations_attempted': state['generations_fast_graph']['iteration_count']
                }
            return 'end'
        else:
            return 'fix'
    
    def _extract_metrics_from_output(self, output: str) -> Dict:
        """Extract metrics from execution output as a verification mechanism."""
        import re
        
        metrics = {
            'model_old_score': {'on_old_data': 0.0, 'on_new_data': 0.0},
            'model_new_score': {'on_old_data': 0.0, 'on_new_data': 0.0}
        }
        
        # Extract old model scores
        old_on_old_match = re.search(r'Old model trained and evaluated on the old distribution: (\d+\.\d+)', output)
        if old_on_old_match:
            metrics['model_old_score']['on_old_data'] = float(old_on_old_match.group(1))
            
        old_on_new_match = re.search(r'Old model evaluated on the new distribution: (\d+\.\d+)', output)
        if old_on_new_match:
            metrics['model_old_score']['on_new_data'] = float(old_on_new_match.group(1))
            
        # Extract new model scores
        new_on_old_match = re.search(r'New model (trained and )?evaluated on old distribution: (\d+\.\d+)', output)
        if new_on_old_match:
            metrics['model_new_score']['on_old_data'] = float(new_on_old_match.group(2))
            
        new_on_new_match = re.search(r'New model evaluated on new distribution: (\d+\.\d+)', output)
        if new_on_new_match:
            metrics['model_new_score']['on_new_data'] = float(new_on_new_match.group(1))
            
        return metrics
            
    def export_results_to_yaml(self, state: WorkingMemory, runtime_seconds: float) -> Dict:
        """Format results to match standardized YAML output."""
        # Get the initial code from semantic memory
        initial_code = state['semantic_memory'].model_code if 'semantic_memory' in state else ""
        
        # Get the current retraining code
        current_code_yaml = state['generations_fast_graph'].get('fixed_code') or state['generations_fast_graph']['new_training_code']
        parsed_yaml = yaml.safe_load(current_code_yaml)
        final_code = parsed_yaml['new_training_code']
        
        # Prioritize metrics in this order:
        # 1. metrics extracted from console output (most reliable)
        # 2. metrics from the model_*_score dictionaries
        
        # Start with console-extracted metrics as most reliable source
        console_metrics = state['generations_fast_graph'].get('extracted_metrics', {})
        
        # Extract initial metrics
        initial_metrics = {}
        if console_metrics and 'model_old_score' in console_metrics:
            initial_metrics = {
                "old_distribution": console_metrics['model_old_score'].get("on_old_data", 0),
                "new_distribution": console_metrics['model_old_score'].get("on_new_data", 0)
            }
        elif 'model_old_score' in state['generations_fast_graph']:
            old_model_score = state['generations_fast_graph']['model_old_score']
            initial_metrics = {
                "old_distribution": old_model_score.get("on_old_data", 0),
                "new_distribution": old_model_score.get("on_new_data", 0)
            }
            
        # Extract final metrics
        final_metrics = {}
        if console_metrics and 'model_new_score' in console_metrics:
            final_metrics = {
                "old_distribution": console_metrics['model_new_score'].get("on_old_data", 0),
                "new_distribution": console_metrics['model_new_score'].get("on_new_data", 0)
            }
        elif 'model_new_score' in state['generations_fast_graph']:
            new_model_score = state['generations_fast_graph']['model_new_score']
            final_metrics = {
                "old_distribution": new_model_score.get("on_old_data", 0),
                "new_distribution": new_model_score.get("on_new_data", 0)
            }
            
        # Build improvement path from improvement history
        improvement_path = []
        if 'improvement_history' in state and state['improvement_history']:
            for i, entry in enumerate(state['improvement_history']):
                # Extract changes
                changes = []
                if 'changes_made' in entry:
                    for key, value in entry['changes_made'].items():
                        if isinstance(value, bool) and value:
                            changes.append(f"Applied {key}")
                        elif not isinstance(value, bool):
                            changes.append(f"{key}: {value}")
                
                # Create standard path entry
                path_entry = {
                    "iteration": i + 1,
                    "code": entry.get('new_code', final_code),
                    "metrics": {
                        "old_distribution": entry.get('metrics', {}).get('old_distribution', final_metrics.get("old_distribution", 0)),
                        "new_distribution": entry.get('metrics', {}).get('new_distribution', final_metrics.get("new_distribution", 0))
                    },
                    "changes": changes,
                    "reflection": f"Fast retraining execution output:\n{state['generations_fast_graph'].get('execution_output', '')}"
                }
                improvement_path.append(path_entry)
        
        # If no improvement history, create a default entry
        if not improvement_path and final_metrics:
            default_changes = ["Applied retraining on combined data"]
            if state['generations_fast_graph'].get('iteration_count', 0) > 1:
                default_changes.append(f"Fixed code execution issues ({state['generations_fast_graph'].get('iteration_count', 0)} iterations)")
                
            improvement_path = [{
                "iteration": 1,
                "code": final_code,
                "metrics": final_metrics,
                "changes": default_changes,
                "reflection": f"Fast retraining execution output:\n{state['generations_fast_graph'].get('execution_output', '')}"
            }]
            
        # Build the standardized output
        agent_name = "improver" if state['generations_fast_graph']['has_slow_graph_insights'] else "fast"
        output = {
            "agent_name": agent_name,
            "initial_code": initial_code,
            "initial_metrics": initial_metrics,
            "improvement_path": improvement_path,
            "final_code": final_code,
            "final_metrics": final_metrics,
            "runtime_statistics": {
                "total_time_seconds": runtime_seconds,
                "iterations": state['generations_fast_graph'].get('iteration_count', 0),
                "tokens_used": len(final_code) // 4,
                "evaluation_timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        # When using slow graph insights, compare final metrics with both fast graph and slow graph baselines
        if state['generations_fast_graph'].get('has_slow_graph_insights', False):
            # Get potential baseline metrics to compare against
            baselines = []
            
            # Add fast graph metrics as baseline (current initial_metrics)
            if initial_metrics:
                baselines.append({
                    'name': 'Fast Graph',
                    'metrics': initial_metrics,
                    'code': initial_code
                })
            
            # Add slow graph metrics as baseline if available
            if ('generations_slow_graph' in state and 
                'yaml_output' in state['generations_slow_graph'] and 
                'final_metrics' in state['generations_slow_graph']['yaml_output']):
                slow_graph_metrics = state['generations_slow_graph']['yaml_output']['final_metrics']
                slow_graph_code = state['generations_slow_graph']['yaml_output']['final_code']
                baselines.append({
                    'name': 'Slow Graph',
                    'metrics': slow_graph_metrics,
                    'code': slow_graph_code
                })
            
            # Find the best baseline and current result
            current_total = final_metrics.get("old_distribution", 0) + final_metrics.get("new_distribution", 0)
            best_baseline = None
            best_total = current_total
            
            for baseline in baselines:
                baseline_total = baseline['metrics'].get("old_distribution", 0) + baseline['metrics'].get("new_distribution", 0)
                if baseline_total > best_total:
                    best_baseline = baseline
                    best_total = baseline_total
            
            # Revert to best baseline if it's better than current result
            if best_baseline:
                print(f"Reverting to {best_baseline['name']} metrics: {best_baseline['name']} total={best_total:.4f} > Current total={current_total:.4f}")
                output["final_metrics"] = best_baseline['metrics']
                output["final_code"] = best_baseline['code']
                output["reverted_to_baseline"] = best_baseline['name']
            else:
                print(f"Keeping current Fast Graph results: Current total={current_total:.4f} >= Best baseline total={best_total:.4f}")
                output["reverted_to_baseline"] = None
        
        return output
    
    def run(self, initial_state: WorkingMemory):
        """Run the fast graph improvement process with enhanced logging and standardized output"""
        self.start_time = datetime.now()
        output_keys = ['generations_fast_graph', 'improvement_history']
        visited_keys = []
        final_output = None
        
        for output in self.decision_procedure.stream(initial_state, output_keys=output_keys, debug=False):
            final_output = output
            for node_name, state in output.items():
                # Print generations updates
                for k, v in state['generations_fast_graph'].items():
                    if state['generations_fast_graph'][k] and k not in visited_keys:
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
                        content = Text(
                            f"Outcome: {latest_improvement['outcome']}\n"
                            f"Improvements:\n"
                            f"  New Distribution: {latest_improvement['improvements']['new_distribution']:.4f}\n"
                            f"  Old Distribution: {latest_improvement['improvements']['old_distribution']:.4f}"
                        )
                        panel = Panel(content, title=title)
                        print(panel)
                        visited_keys.append('latest_improvement')
        
        # Calculate runtime
        end_time = datetime.now()
        runtime_seconds = (end_time - self.start_time).total_seconds()
        
        # Get the final state
        final_state = final_output[list(final_output.keys())[-1]]
        
        # Extract metrics from the execution output as a verification/backup
        if 'generations_fast_graph' in final_state and 'execution_output' in final_state['generations_fast_graph']:
            console_metrics = self._extract_metrics_from_output(final_state['generations_fast_graph']['execution_output'])
            if console_metrics:
                final_state['generations_fast_graph']['extracted_metrics'] = console_metrics
        
        # Generate the standardized YAML output
        yaml_output = self.export_results_to_yaml(final_state, runtime_seconds)
        
        # Add the YAML output to the state
        final_state['yaml_output'] = yaml_output
        
        return final_state

class EnhancedStateGraph(StateGraph):
    def add_node(self, node_name, function):
        decorated_function = print_function_name(function)
        super().add_node(node_name, decorated_function)