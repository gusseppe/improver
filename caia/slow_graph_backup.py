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
from autogen.coding import LocalCommandLineCodeExecutor

from caia.memory import WorkingMemory, SemanticMemory, EpisodicMemory 
from caia.utils import print_function_name
from caia.prompts import (
    prompt_distill_memories,
    prompt_generate_tiny_change,
    prompt_evaluate_change, prompt_summarize_model_docs,
    prompt_fix_code
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
        # Add nodes for the slow improvement process
        self.graph.add_node('distill_memories', self.distill_memories)
        self.graph.add_node('generate_tiny_change', self.generate_tiny_change)
        self.graph.add_node('apply_change', self.apply_change)
        self.graph.add_node('evaluate_change', self.evaluate_change)
        
        # Set entry point
        self.graph.set_entry_point('distill_memories')
        
        # Add edges for the main flow
        self.graph.add_edge('distill_memories', 'generate_tiny_change')
        self.graph.add_edge('generate_tiny_change', 'apply_change')
        
        # Add conditional edge for apply_change to handle retries
        self.graph.add_conditional_edges(
            'apply_change',
            self.should_evaluate_code,
            {
                'evaluate': 'evaluate_change',
                'retry': 'generate_tiny_change'
            }
        )
        
        # Add conditional edge for evaluate_change to handle improvement loops
        self.graph.add_conditional_edges(
            'evaluate_change',
            self.should_continue_improving,
            {
                'continue': 'generate_tiny_change',
                'end': END
            }
        )

    def distill_memories(self, state: WorkingMemory) -> WorkingMemory:
        """Distill insights from semantic and episodic memories.
        
        Args:
            state: Current working memory state
            
        Returns:
            Updated working memory with distilled insights
        """
        # semantic_memory = state['semantic_memory']
        semantic_memory = state['semantic_memory']

        doc_prompt = prompt_summarize_model_docs()
        doc_chain = doc_prompt | self.llm
        model_params_summary = doc_chain.invoke({'input': semantic_memory.model_object.__doc__}).content
        
        semantic_memory = {
            'old_training_code': semantic_memory.model_code,
            'model_documentation': model_params_summary,
            'dataset_description': semantic_memory.reference_dataset.description
        }

        episodic_memory = state['episodic_memory'][-1].quick_insight
        # improvement_history = state.get('improvement_history', [])
        
        # Create the prompt for the LLM
        prompt = prompt_distill_memories()
        
        # Prepare the input in YAML format
        yaml_content = {
            'semantic_memory': str(semantic_memory),
            'episodic_memory': str(episodic_memory),
            # 'improvement_history': str(improvement_history)
        }
        
        # Invoke the LLM with yaml content
        chain = prompt | self.llm
        output = chain.invoke({'input': yaml.dump(yaml_content)}).content
        
        # Update the state with the distilled insights
        if 'generations' not in state:
            state['generations'] = {}
        state['generations']['distilled_insights'] = output
        
        return state

    def generate_tiny_change(self, state: WorkingMemory) -> WorkingMemory:
        """Generate a small improvement change based on insights.
        
        Args:
            state: Current working memory state
            
        Returns:
            Updated working memory with generated change
        """
        # Extract the latest execution results
        execution_output = state['generations'].get('execution_output', '')
        execution_success = state['generations'].get('execution_success', False)
        iteration_count = state['generations'].get('iteration_count', 0)
        
        # Get insights and analysis
        distilled_insights = state['generations'].get('distilled_insights', '')
        criticality_analysis = state['generations'].get('criticality_analysis', '')
        
        # Get current code - either from previous iteration or initial code
        if 'tiny_change' in state['generations']:
            # Extract the actual code from the YAML content of previous change
            try:
                previous_change = yaml.safe_load(state['generations']['tiny_change'])
                current_code = previous_change.get('new_training_code', '')
            except:
                current_code = state['episodic_memory'][-1].quick_insight.get('new_training_code', '')
        else:
            current_code = state['episodic_memory'][-1].quick_insight.get('new_training_code', '')
        
        # Extract performance metrics for decision making
        try:
            # Parse performance metrics from execution output
            ref_score = float(re.search(r'Model trained and evaluated on the reference distribution: ([\d.]+)', execution_output).group(1))
            new_dist_score = float(re.search(r'Reference model evaluated on the new distribution: ([\d.]+)', execution_output).group(1))
            score_diff = float(re.search(r'Score difference: ([-\d.]+)', execution_output).group(1))
        except:
            ref_score = 0.0
            new_dist_score = 0.0
            score_diff = 0.0
        
        # Create the prompt for the LLM
        prompt = prompt_generate_tiny_change()
        
        # Prepare the input in YAML format
        yaml_content = {
            'current_code': current_code,
            'execution_results': {
                'output': execution_output,
                'reference_score': ref_score,
                'new_distribution_score': new_dist_score,
                'score_difference': score_diff,
                'success': execution_success
            },
            'distilled_insights': distilled_insights,
            'criticality_analysis': criticality_analysis,
            'iteration_count': iteration_count,
            'previous_changes': state.get('improvement_history', [])
        }
        
        # Invoke the LLM
        chain = prompt | self.llm
        output = chain.invoke({'input': yaml.dump(yaml_content)}).content
        
        # Parse and validate the output
        try:
            generated_change = yaml.safe_load(output)
            # Ensure we have the required keys
            if not all(k in generated_change for k in ['new_training_code', 'changes_made', 'rationale']):
                raise ValueError("Missing required keys in generated change")
            
            # Store the changes in improvement history
            if 'improvement_history' not in state:
                state['improvement_history'] = []
                
            state['improvement_history'].append({
                'iteration': iteration_count,
                'changes': generated_change['changes_made'],
                'rationale': generated_change['rationale'],
                'previous_scores': {
                    'reference': ref_score,
                    'new_distribution': new_dist_score,
                    'difference': score_diff
                }
            })
            
        except Exception as e:
            print(f"Error parsing LLM output: {e}")
            # Fallback to raw output
            state['improvement_history'].append({
                'iteration': iteration_count,
                'changes': 'Error parsing changes',
                'rationale': 'Error parsing rationale',
                'previous_scores': {
                    'reference': ref_score,
                    'new_distribution': new_dist_score,
                    'difference': score_diff
                }
            })
        
        # Update the state with the new code change
        state['generations']['tiny_change'] = output
        
        return state


    def apply_change(self, state: WorkingMemory) -> WorkingMemory:
        """Apply the generated change and execute the code.
        
        Args:
            state: Current working memory state
            
        Returns:
            Updated working memory with execution results
        """
        # Get the current code with tiny change
        current_code_yaml = state['generations']['tiny_change']
        parsed_yaml = yaml.safe_load(current_code_yaml)
        current_code = parsed_yaml['new_training_code']
        
        # Wrap the code in Python code block
        wrapped_code = f"```python\n{current_code}\n```"
        
        # Execute code using autogen executor
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
        execution_output = code_executor_agent.generate_reply(
            messages=[{"role": "user", "content": wrapped_code}]
        )
        
        # Store execution results
        print("Execution Output:", "-"*100)
        # print(wrapped_code)
        print(execution_output)
        state['generations']['execution_output'] = execution_output
        
        # Check if execution was successful
        if "succeeded" in execution_output.lower():
            state['generations']['execution_success'] = True
            # Extract accuracy from the output
            try:
                accuracy = float(execution_output.split("New Model Accuracy: ")[-1].split("\n")[0])
                state['generations']['new_accuracy'] = accuracy
            except:
                state['generations']['execution_success'] = False
        else:
            state['generations']['execution_success'] = False
        
        return state

    def evaluate_change(self, state: WorkingMemory) -> WorkingMemory:
        """Evaluate the applied change and update improvement history.
        
        Args:
            state: Current working memory state
            
        Returns:
            Updated working memory with evaluation results
        """
        current_accuracy = state['generations'].get('new_accuracy')
        previous_accuracy = state.get('previous_accuracy', 0)
        current_code = state['generations']['tiny_change']
        previous_code = state.get('previous_code', '')
        
        # Create evaluation prompt
        prompt = prompt_evaluate_change()
        
        # Prepare input YAML
        yaml_content = {
            'current_code': current_code,
            'execution_output': state['generations']['execution_output'],
            'previous_accuracy': previous_accuracy
        }
        
        # Get evaluation from LLM
        chain = prompt | self.llm
        evaluation_output = chain.invoke({'input': yaml.dump(yaml_content)}).content
        evaluation_result = yaml.safe_load(evaluation_output)
        
        # Prepare improvement history entry
        improvement_entry = {
            'previous_code': previous_code,
            'new_code': current_code,
            'accuracy_change': current_accuracy - previous_accuracy if current_accuracy else 0,
            'evaluation': evaluation_result
        }
        
        # Update improvement history
        if 'improvement_history' not in state:
            state['improvement_history'] = []
        state['improvement_history'].append(improvement_entry)
        
        # Update current state
        state['previous_accuracy'] = current_accuracy
        state['previous_code'] = current_code
        state['generations']['evaluation'] = evaluation_result
        
        return state

    def should_evaluate_code(self, state: WorkingMemory) -> str:
        """Determine if code should be evaluated or retried.
        
        Args:
            state: Current working memory state
            
        Returns:
            'evaluate' if code executed successfully, 'retry' otherwise
        """
        return 'evaluate' if state['generations'].get('execution_success', False) else 'retry'

    def should_continue_improving(self, state: WorkingMemory) -> str:
        """Determine if improvement process should continue.
        
        Args:
            state: Current working memory state
            
        Returns:
            'continue' if more improvements needed, 'end' otherwise
        """
        # Get the latest improvement result
        if not state['improvement_history']:
            return 'continue'
            
        latest_improvement = state['improvement_history'][-1]
        iteration_count = len(state['improvement_history'])
        
        # Stop conditions:
        # 1. Reached max iterations (5)
        # 2. Found significant improvement (>5% accuracy gain)
        # 3. Multiple failed attempts in a row (3)
        if (
            iteration_count >= 5 or
            latest_improvement['accuracy_change'] > 0.05 or
            (iteration_count > 2 and all(
                h['evaluation']['recommendation']['action'] == 'reject' 
                for h in state['improvement_history'][-3:]
            ))
        ):
            return 'end'
            
        return 'continue'

    def run(self, initial_state: WorkingMemory):
        """Run the slow improvement process.
        
        Args:
            initial_state: Initial working memory state
            
        Returns:
            Final state after improvements
        """
        output_keys = ['generations', 'improvement_history']
        visited_keys = []
        
        for output in self.decision_procedure.stream(
            initial_state, 
            output_keys=output_keys, 
            debug=False
        ):
            for node_name, state in output.items():
                # Print generations updates
                for k, v in state['generations'].items():
                    if state['generations'][k] and k not in visited_keys:
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