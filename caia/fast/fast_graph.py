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
    prompt_execute_and_fix_retraining_code
)

from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor
from typing import TypedDict, Dict, Optional
from caia.memory import ModelScore, ImprovementEntry, create_improvement_entry

class FastGraph:
    def __init__(self, llm, debug=False):
        self.llm = llm
        self.graph = EnhancedStateGraph(WorkingMemory)
        self.build_plan()
        self.decision_procedure = self.graph.compile(debug=debug)

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

    # def measure_criticality(self, state: WorkingMemory) -> WorkingMemory:
        
    #     # Prepare the monitoring report for analysis
    #     monitoring_report = state['monitoring_report']
        
    #     # Create the prompt for the LLM
    #     prompt = prompt_measure_criticality()
        
    #     # Invoke the LLM
    #     chain = prompt | self.llm
    #     output = chain.invoke({'input': monitoring_report}).content
        
    #     # Update the state with the analysis result
    #     state['generations_fast_graph']['criticality_analysis'] = output
        
        
    #     return state
    

    def generate_retraining_code(self, state: WorkingMemory) -> WorkingMemory:
        try:
            episodic_memory = state['episodic_memory']
            semantic_memory = state['semantic_memory']
            
            if not semantic_memory.model_code:
                raise ValueError("No model code found in semantic memory")
            
            training_code = semantic_memory.model_code
            new_data = (
                'X_train_new = pd.read_csv(f"{dataset_folder}/X_train_new.csv")\n'
                'X_test_new = pd.read_csv(f"{dataset_folder}/X_test_new.csv")\n'
                'y_train_new = pd.read_csv(f"{dataset_folder}/y_train_new.csv").squeeze("columns")\n'
                'y_test_new = pd.read_csv(f"{dataset_folder}/y_test_new.csv").squeeze("columns")\n'
            )

            # Initialize generations dictionary if not present
            if 'generations_fast_graph' not in state:
                state['generations_fast_graph'] = {}

            # Create the prompt and prepare YAML content
            prompt = prompt_generate_retraining_code()
            yaml_content = self._prepare_yaml_content(training_code, new_data)
            
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
            timeout=10,
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

                # Create improvement entry
                improvement_entry = create_improvement_entry(
                    previous_code=state['semantic_memory'].model_code,
                    new_code=current_code,
                    graph_type='fast',
                    strategy_type=None,
                    old_model_score=old_model_score,
                    new_model_score=new_model_score,
                    changes_made={
                        'retrained_on_combined_data': True,
                        'iteration_count': state['generations_fast_graph'].get('iteration_count', 0)
                    }
                )

                # Initialize improvement_history if it doesn't exist
                if 'improvement_history' not in state:
                    state['improvement_history'] = []
                
                # Add the new improvement entry
                state['improvement_history'].append(improvement_entry)
                
                state['generations_fast_graph']['execution_success'] = True
                
            except Exception as e:
                print(f"Error processing metrics and creating improvement entry: {str(e)}")
                state['generations_fast_graph']['execution_success'] = False
        else:
            state['generations_fast_graph']['execution_success'] = False
        
        # Increment the iteration count
        state['generations_fast_graph']['iteration_count'] = state.get('iteration_count', 0) + 1
        
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


    def run(self, initial_state: WorkingMemory):
        """Run the fast graph improvement process with enhanced logging"""
        output_keys = ['generations_fast_graph', 'improvement_history']
        visited_keys = []
        
        for output in self.decision_procedure.stream(initial_state, output_keys=output_keys, debug=False):
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

        return output

class EnhancedStateGraph(StateGraph):
    def add_node(self, node_name, function):
        decorated_function = print_function_name(function)
        super().add_node(node_name, decorated_function)