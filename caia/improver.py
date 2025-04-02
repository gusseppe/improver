import textwrap
import yaml
import os
import uuid
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from rich import print
from rich.panel import Panel
from rich.text import Text
from docarray import BaseDoc, DocList

from caia.fast.fast_graph import FastGraph
from caia.slow.slow_graph import SlowGraph
from caia.memory import WorkingMemory, EpisodicMemory
from caia.utils import save_yaml_results
from caia.prompts import prompt_generate_retraining_code
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

def prompt_generate_retraining_with_insights() -> ChatPromptTemplate:
    """Create a prompt for generating retraining code using insights from previous runs."""
    system_prompt = """
    You are an expert machine learning engineer. You have to rewrite the given training code to obtain a retraining code,
    leveraging insights from previous improvement attempts.

    Context: Given old training code, new data loading code, and insights from previous improvements,
    generate a new training code that retrains the model.

    Objective: Create a new training code (new_training_code) that:
    1. First evaluates the old model on both distributions:
       - Trains on old data
       - Tests on old test set 
       - Tests on new (drifted) test set
       - Saves old model metrics to 'old_metrics.yaml' with structure:
         model_old_score:
           on_new_data: [score on new data]
           on_old_data: [score on old data]
    2. Then trains a new model that DIRECTLY implements the insights from previous improvements:
       - Use EXACTLY the same model architecture and parameters from the deep_insights
       - Copy the successful strategy directly from the insights
       - Train on combined dataset (old + new data)
       - Tests on old test set
       - Tests on new (drifted) test set
       - Saves new model metrics to 'fast_graph_metrics.yaml' with structure:
         model_new_score:
           on_new_data: [score on new data]
           on_old_data: [score on old data]
    3. Prints performance metrics at each step
    4. Include proper error handling
    5. Add a final print statement mentioning the insights applied
            
    Style: Provide clear, well-structured Python code that directly applies the insights from previous runs.
    Do not try to "improve" on the insights - implement them exactly as provided.

    Response Format: Format your response as YAML output with the following structure:

    new_training_code: |
      [NEW TRAINING/EVALUATING CODE HERE]

    Only provide the YAML-formatted code output. Do not include any other explanation or commentary.

    IMPORTANT: Make sure to use the exact model and parameters from the deep_insights. Do not improvise or
    make your own improvements - the goal is to replicate the successful strategy from previous runs.
    """

    system_prompt = textwrap.dedent(system_prompt).strip()

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    return final_prompt

# Update FastGraph to use deep insights from episodic memory
def fast_graph_generate_retraining_code(self, state: WorkingMemory) -> WorkingMemory:
    """Generate retraining code, using insights from episodic memory when available."""
    try:
        episodic_memory = state['episodic_memory']
        semantic_memory = state['semantic_memory']
        
        if not semantic_memory.model_code:
            raise ValueError("No model code found in semantic memory")
        
        training_code = semantic_memory.model_code
        dataset_folder = "datasets/financial"
        new_data = (
            f'X_train_new = pd.read_csv(f"{dataset_folder}/X_train_new.csv")\n'
            f'X_test_new = pd.read_csv(f"{dataset_folder}/X_test_new.csv")\n'
            f'y_train_new = pd.read_csv(f"{dataset_folder}/y_train_new.csv").squeeze("columns")\n'
            f'y_test_new = pd.read_csv(f"{dataset_folder}/y_test_new.csv").squeeze("columns")\n'
        )
        
        # Initialize generations dictionary if not present
        if 'generations_fast_graph' not in state:
            state['generations_fast_graph'] = {}
        
        # Enhanced debug information about episodic memory
        print("\n🔍 Checking for insights from previous runs...")
        print(f"Episodic memory exists: {episodic_memory is not None}")
        print(f"Episodic memory entries: {len(episodic_memory) if episodic_memory else 0}")
        
        has_insights = False
        if episodic_memory and len(episodic_memory) > 0:
            print(f"Last entry has deep_insight attribute: {hasattr(episodic_memory[-1], 'deep_insight')}")
            
            if hasattr(episodic_memory[-1], 'deep_insight'):
                insight_exists = bool(episodic_memory[-1].deep_insight)
                print(f"Deep insight is not empty: {insight_exists}")
                
                if insight_exists:
                    deep_insight = episodic_memory[-1].deep_insight
                    strategy = deep_insight.get('strategy', 'unknown')
                    has_code = 'code' in deep_insight and deep_insight['code']
                    print(f"Deep insight strategy: {strategy}")
                    print(f"Deep insight has code: {has_code}")
                    
                    # Set the flag if we have valid insights with code
                    has_insights = has_code
        
        # Prepare the prompt based on whether we have insights
        if has_insights:
            print("\n🔍 Using insights from previous runs to generate retraining code")
            prompt = prompt_generate_retraining_with_insights()
            
            # Use the improved preparation method
            yaml_content = self._prepare_yaml_content_with_insights(
                training_code, 
                new_data,
                episodic_memory[-1].deep_insight
            )
            
            # Track that we're using insights
            state['generations_fast_graph']['using_insights'] = True
            state['generations_fast_graph']['insight_strategy'] = episodic_memory[-1].deep_insight.get('strategy', 'unknown')
        else:
            print("\n🔍 No previous insights found or insights incomplete. Generating retraining code from scratch.")
            prompt = prompt_generate_retraining_code()
            yaml_content = self._prepare_yaml_content(training_code, new_data)
            state['generations_fast_graph']['using_insights'] = False
        
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
        import traceback
        traceback.print_exc()
        state['generations_fast_graph']['error'] = str(e)
        return state

def fast_graph_prepare_yaml_content_with_insights(self, training_code: str, new_data: str, deep_insight: Dict) -> str:
    """Helper method to prepare YAML content with insights from episodic memory."""
    cleaned_code = textwrap.dedent(training_code).strip()
    indented_code = textwrap.indent(cleaned_code, '  ')
    
    cleaned_new_data = textwrap.dedent(new_data).strip()
    indented_new_data = textwrap.indent(cleaned_new_data, '  ')
    
    # Extract only the most relevant deep insights for a cleaner YAML
    insight_dict = {
        'strategy': deep_insight.get('strategy', 'unknown'),
        'code': deep_insight.get('code', ''),
        'changes': deep_insight.get('changes', {}),
        'metrics': deep_insight.get('metrics', {})
    }
    
    # Format the insights in a way that's easier for the model to understand
    insights_yaml = yaml.dump(insight_dict, default_flow_style=False)
    indented_insights = textwrap.indent(insights_yaml, '  ')
    
    return (
        f"old_training_code: |\n"
        f"{indented_code}\n"
        f"new_data: |\n"
        f"{indented_new_data}\n"
        f"deep_insights: |\n"
        f"{indented_insights}\n"
    )

# Update SlowGraph to save insights to episodic memory
def slow_graph_update_episodic_memory(state: WorkingMemory) -> WorkingMemory:
    """Update episodic memory with insights from slow graph execution."""
    # Check if we have improvement history and episodic memory
    if not state.get('improvement_history') or 'episodic_memory' not in state:
        print("\n⚠️ Cannot update episodic memory: Missing improvement history or episodic memory")
        return state
    
    # Get the best improvement
    best_improvement = None
    best_improvement_value = -float('inf')
    
    for entry in state.get('improvement_history', []):
        # Check if this is a successful improvement
        if entry.get('outcome', '') == 'success':
            # Get the improvement value on new distribution
            new_dist_improvement = entry.get('improvements', {}).get('new_distribution', 0)
            
            # Track the best improvement
            if new_dist_improvement > best_improvement_value:
                best_improvement_value = new_dist_improvement
                best_improvement = entry
    
    # If no successful improvements, try to find any improvement with positive metrics
    if not best_improvement:
        for entry in state.get('improvement_history', []):
            # Get the improvement value on new distribution
            new_dist_improvement = entry.get('improvements', {}).get('new_distribution', 0)
            
            # Track the best improvement
            if new_dist_improvement > best_improvement_value:
                best_improvement_value = new_dist_improvement
                best_improvement = entry
    
    # As a last resort, use the last improvement
    if not best_improvement and state.get('improvement_history'):
        best_improvement = state['improvement_history'][-1]
        print("\n⚠️ No positive improvements found. Using last improvement for insights.")
    
    if best_improvement:
        # Create deep insight entry from the best improvement with full details
        # Make sure the code is properly formatted and doesn't have YAML issues
        code = best_improvement.get('new_code', '')
        if code:
            # Strip any YAML formatting that might confuse the parser later
            code = code.replace('|\n', '')
        
        deep_insight = {
            'strategy': best_improvement.get('strategy_type', ''),
            'code': code,
            'changes': best_improvement.get('changes_made', {}),
            'metrics': best_improvement.get('metrics', {}),
            'evaluation': best_improvement.get('evaluation', {})
        }
        
        # Update the last episodic memory entry with deep insight
        if state['episodic_memory'] and len(state['episodic_memory']) > 0:
            try:
                # Store deep insight
                state['episodic_memory'][-1].deep_insight = deep_insight
                
                # Log what was saved
                print("\n✅ Updated episodic memory with insights from SlowGraph")
                print(f"Strategy: {deep_insight['strategy']}")
                print(f"Metrics: New Data {deep_insight['metrics'].get('new_model', {}).get('on_new_data', 0):.4f}, Old Data {deep_insight['metrics'].get('new_model', {}).get('on_old_data', 0):.4f}")
                print(f"Code length: {len(deep_insight['code'])} characters")
                
                # Verify the insight was actually saved
                print(f"Verification - deep_insight exists: {hasattr(state['episodic_memory'][-1], 'deep_insight')}")
                if hasattr(state['episodic_memory'][-1], 'deep_insight'):
                    print(f"Verification - deep_insight has code: {'code' in state['episodic_memory'][-1].deep_insight}")
                    
            except Exception as e:
                print(f"⚠️ Error saving deep insight to episodic memory: {str(e)}")
                import traceback
                traceback.print_exc()
                
    return state


class Improver:
    def __init__(self, llm, max_iterations=1, max_failures=3, debug=False):
        """Initialize the combined Improver agent.
        
        Args:
            llm: The language model to use for generation
            max_iterations: Maximum iterations per graph
            max_failures: Maximum consecutive execution failures allowed
            debug: Whether to run in debug mode
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.max_failures = max_failures
        self.debug = debug
        
        # Initialize both graphs with same parameters
        self.fast_graph = FastGraph(llm, debug=debug)
        self.slow_graph = SlowGraph(llm, max_iterations=max_iterations, max_failures=max_failures, debug=debug)
        
        # Add the new methods to FastGraph
        self.fast_graph.generate_retraining_code = fast_graph_generate_retraining_code.__get__(self.fast_graph)
        self.fast_graph._prepare_yaml_content_with_insights = fast_graph_prepare_yaml_content_with_insights.__get__(self.fast_graph)
        
        # Start time for tracking
        self.start_time = datetime.now()
        
    def run(self, working_memory: WorkingMemory) -> WorkingMemory:
        """Run the combined improvement process with improved memory preservation and debugging."""
        print("\n🚀 Starting Combined Improver Agent")
        self.start_time = datetime.now()
        
        # Print info about input working memory
        print("\n📋 Input Working Memory Stats:")
        print(f"- Episodic memory exists: {working_memory.get('episodic_memory') is not None}")
        print(f"- Episodic memory entries: {len(working_memory.get('episodic_memory', [])) if working_memory.get('episodic_memory') else 0}")
        
        has_prior_insights = False
        if working_memory.get('episodic_memory') and len(working_memory.get('episodic_memory', [])) > 1:
            print(f"- Last entry has deep insight: {hasattr(working_memory['episodic_memory'][-1], 'deep_insight')}")
            if hasattr(working_memory['episodic_memory'][-1], 'deep_insight') and working_memory['episodic_memory'][-1].deep_insight:
                has_prior_insights = True
                print(f"- Deep insight strategy: {working_memory['episodic_memory'][-1].deep_insight.get('strategy', 'unknown')}")
                print("✅ Found prior insights from previous run - will skip slow graph")
        
        # Configure working memory
        working_memory["max_iterations"] = self.max_iterations
        working_memory["max_failures"] = self.max_failures
        
        # Ensure episodic memory exists (create if missing)
        if 'episodic_memory' not in working_memory or working_memory['episodic_memory'] is None:
            print("\n⚠️ Missing episodic memory - initializing empty DocList")
            working_memory['episodic_memory'] = DocList[EpisodicMemory]([])
        
        # Check if episodic memory is empty
        if len(working_memory['episodic_memory']) == 0:
            print("\n⚠️ Empty episodic memory - adding a placeholder")
            # Create a placeholder entry to prevent None reference issues
            placeholder = EpisodicMemory(
                dataset_new=working_memory.get('semantic_memory', {}).dataset_old,
                quick_insight={},
                deep_insight={}
            )
            working_memory['episodic_memory'].append(placeholder)
        
        # Initialize token tracking
        token_counts = {"prompt": 0, "completion": 0, "total": 0}
        
        # Store original episodic memory to preserve it
        original_episodic_memory = working_memory['episodic_memory']
        
        # Run Fast Graph
        print("\n⚡ Running Fast Graph...")
        fast_start_time = datetime.now()
        fast_output = self.fast_graph.run(working_memory)
        fast_end_time = datetime.now()
        fast_runtime = (fast_end_time - fast_start_time).total_seconds()
        print(f"\n⚡ Fast Graph completed in {fast_runtime:.2f} seconds")
        
        # Update token counts if available
        if hasattr(self.fast_graph, 'token_counts'):
            token_counts["prompt"] += self.fast_graph.token_counts.get("prompt", 0)
            token_counts["completion"] += self.fast_graph.token_counts.get("completion", 0)
            token_counts["total"] += self.fast_graph.token_counts.get("total", 0)
        
        # Ensure episodic memory is preserved
        if 'episodic_memory' not in fast_output or fast_output['episodic_memory'] is None:
            print("\n⚠️ Fast Graph lost episodic memory - restoring from original")
            fast_output['episodic_memory'] = original_episodic_memory
        
        # Skip Slow Graph if we already have deep insights from previous runs
        if has_prior_insights:
            print("\n🧠 Skipping Slow Graph - using insights from previous run")
            slow_output = fast_output
            slow_runtime = 0
            slow_success = True
        else:
            # Run Slow Graph only for first execution
            print("\n🧠 Running Slow Graph (first execution)...")
            slow_start_time = datetime.now()
            
            try:
                slow_output = self.slow_graph.run(fast_output)
                slow_success = True
            except Exception as e:
                print(f"\n❌ Error in Slow Graph: {str(e)}")
                import traceback
                traceback.print_exc()
                slow_output = fast_output  # Use fast graph output as fallback
                slow_success = False
            
            slow_end_time = datetime.now()
            slow_runtime = (slow_end_time - slow_start_time).total_seconds()
            print(f"\n🧠 Slow Graph completed in {slow_runtime:.2f} seconds")
            
            # Update token counts
            if hasattr(self.slow_graph, 'token_counts'):
                token_counts["prompt"] += self.slow_graph.token_counts.get("prompt", 0)
                token_counts["completion"] += self.slow_graph.token_counts.get("completion", 0)
                token_counts["total"] += self.slow_graph.token_counts.get("total", 0)
            
            # Update episodic memory only if slow graph completed successfully
            if slow_success:
                slow_output = slow_graph_update_episodic_memory(slow_output)
        
        # Calculate total runtime
        end_time = datetime.now()
        total_runtime = (end_time - self.start_time).total_seconds()
        
        # Print stats about output episodic memory
        print("\n📋 Output Working Memory Stats:")
        print(f"- Episodic memory exists: {slow_output.get('episodic_memory') is not None}")
        print(f"- Episodic memory entries: {len(slow_output.get('episodic_memory', [])) if slow_output.get('episodic_memory') else 0}")
        
        if slow_output.get('episodic_memory') and len(slow_output.get('episodic_memory', [])) > 0:
            print(f"- Last entry has deep insight: {hasattr(slow_output['episodic_memory'][-1], 'deep_insight')}")
            if hasattr(slow_output['episodic_memory'][-1], 'deep_insight'):
                print(f"- Deep insight exists: {bool(slow_output['episodic_memory'][-1].deep_insight)}")
                if slow_output['episodic_memory'][-1].deep_insight:
                    print(f"- Deep insight strategy: {slow_output['episodic_memory'][-1].deep_insight.get('strategy', 'unknown')}")
                    print(f"- Deep insight has code: {'code' in slow_output['episodic_memory'][-1].deep_insight}")
        
        # Create combined YAML output
        final_output = slow_output
        
        # Ensure yaml_output exists
        if 'yaml_output' not in final_output:
            if 'yaml_output' in fast_output:
                final_output['yaml_output'] = fast_output['yaml_output']
            else:
                # Create minimal YAML output
                final_output['yaml_output'] = {
                    "agent_name": "improver",
                    "initial_code": working_memory.get('semantic_memory', {}).model_code if hasattr(working_memory.get('semantic_memory', {}), 'model_code') else "",
                    "initial_metrics": {},
                    "improvement_path": [],
                    "final_code": "",
                    "final_metrics": {},
                    "runtime_statistics": {
                        "total_time_seconds": total_runtime,
                        "fast_graph_time": fast_runtime,
                        "slow_graph_time": slow_runtime,
                        "tokens_used": token_counts["total"],
                        "evaluation_timestamp": datetime.now().isoformat() + "Z"
                    }
                }
        
        if 'yaml_output' in final_output:
            # Update runtime statistics
            final_output['yaml_output']['runtime_statistics']['total_time_seconds'] = total_runtime
            final_output['yaml_output']['runtime_statistics']['fast_graph_time'] = fast_runtime
            final_output['yaml_output']['runtime_statistics']['slow_graph_time'] = slow_runtime
            final_output['yaml_output']['runtime_statistics']['prompt_tokens'] = token_counts["prompt"]
            final_output['yaml_output']['runtime_statistics']['completion_tokens'] = token_counts["completion"]
            final_output['yaml_output']['runtime_statistics']['tokens_used'] = token_counts["total"]
            
            # Update agent name
            final_output['yaml_output']['agent_name'] = "improver"
            
            # Include fast graph metrics if available
            if 'yaml_output' in fast_output:
                fast_metrics = fast_output['yaml_output'].get('final_metrics', {})
                final_output['yaml_output']['fast_graph_metrics'] = fast_metrics
                
            # Add whether insights from previous runs were used
            using_insights = fast_output.get('generations_fast_graph', {}).get('using_insights', False)
            final_output['yaml_output']['used_previous_insights'] = using_insights or has_prior_insights
            
            # Add slow graph success information
            final_output['yaml_output']['slow_graph_success'] = slow_success
            final_output['yaml_output']['slow_graph_skipped'] = has_prior_insights
        
        print(f"\n✅ Improver Agent completed in {total_runtime:.2f} seconds")
        print(f"- Fast Graph: {fast_runtime:.2f} seconds")
        if has_prior_insights:
            print(f"- Slow Graph: Skipped (using previous insights)")
        else:
            print(f"- Slow Graph: {slow_runtime:.2f} seconds")
        print(f"- Token Usage: {token_counts.get('total', 0)} tokens")
        print(f"- Used Previous Insights: {using_insights or has_prior_insights}")
        
        return final_output