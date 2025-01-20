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
    prompt_measure_criticality,
    prompt_generate_retraining_code,
    prompt_execute_and_fix_retraining_code
)

from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor


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
        self.graph.add_node('measure_criticality', self.measure_criticality)

        self.graph.add_node('generate_retraining_code', self.generate_retraining_code)
        self.graph.add_node('execute_retraining_code', self.execute_retraining_code)
        self.graph.add_node('fix_retraining_code', self.fix_retraining_code)
        
        # Set entry point
        self.graph.set_entry_point('measure_criticality')
        
        self.graph.add_edge('measure_criticality', 'generate_retraining_code')
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

    def measure_criticality(self, state: WorkingMemory) -> WorkingMemory:
        
        # Prepare the monitoring report for analysis
        monitoring_report = state['monitoring_report']
        
        # Create the prompt for the LLM
        prompt = prompt_measure_criticality()
        
        # Invoke the LLM
        chain = prompt | self.llm
        output = chain.invoke({'input': monitoring_report}).content
        
        # Update the state with the analysis result
        state['generations_fast_graph']['criticality_analysis'] = output
        
        
        return state
    

    def generate_retraining_code(self, state: WorkingMemory) -> WorkingMemory:
        episodic_memory = state['episodic_memory']
        semantic_memory = state['semantic_memory']
        
        training_code = semantic_memory.model_code
        new_data = (
            'X_train_new = pd.read_csv(f"{dataset_folder}/X_train_new.csv")\n'
            'X_test_new = pd.read_csv(f"{dataset_folder}/X_test_new.csv")\n'
            'y_train_new = pd.read_csv(f"{dataset_folder}/y_train_new.csv").squeeze("columns")\n'
            'y_test_new = pd.read_csv(f"{dataset_folder}/y_test_new.csv").squeeze("columns")\n'
        )

        # Create the prompt for the LLM
        prompt = prompt_generate_retraining_code()
        
        # Prepare the input in YAML format
        cleaned_code = textwrap.dedent(training_code).strip()
        indented_code = textwrap.indent(cleaned_code, '  ')
        
        cleaned_new_data = textwrap.dedent(new_data).strip()
        indented_new_data = textwrap.indent(cleaned_new_data, '  ')

        yaml_content = (
            f"reference_training_code: |\n"
            f"{indented_code}\n"
            f"new_data: |\n"
            f"{indented_new_data}\n"
        )

        # Invoke the LLM
        chain = prompt | self.llm
        output = chain.invoke({'input': yaml_content}).content

        # Update the state with the analysis result
        state['generations_fast_graph']['new_training_code'] = output
        
        return state
    

    def execute_retraining_code(self, state: WorkingMemory) -> WorkingMemory:
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
        
        # print(f"Error output: {execution_output}")
        
        # Store the error output in the state
        state['generations_fast_graph']['execution_output'] = execution_output
        
        # Check if execution was successful
        if "succeeded" in execution_output.lower():
            state['generations_fast_graph']['execution_success'] = True
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
        if state['generations_fast_graph']['execution_success']:
            state['episodic_memory'][-1].quick_insight = state['generations_fast_graph']
            return 'end'
        elif state['generations_fast_graph']['iteration_count'] >= 3:
            return 'end'
        else:
            return 'fix'


    def run(self, initial_state: WorkingMemory):
        output_keys = ['generations_fast_graph']
        visited_keys = []
        for output in self.decision_procedure.stream(initial_state, output_keys=output_keys, debug=False):
            for node_name, state in output.items():
                # sleep(1)
                for k, v in state['generations_fast_graph'].items():
                    if state['generations_fast_graph'][k]:
                        # Create the title with bold, red text
                        if k in visited_keys:
                            continue
                        title = Text(k, style="bold green")
                        content = Text(str(v))
                        panel = Panel(content, title=title)
                        print(panel)

                        visited_keys.append(k)
        return output

class EnhancedStateGraph(StateGraph):
    def add_node(self, node_name, function):
        decorated_function = print_function_name(function)
        super().add_node(node_name, decorated_function)