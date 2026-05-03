# KC-Agent Architecture

KC-Agent (Kahneman-Clear Agent) is a dual-process cognitive architecture for automatic ML model improvement under distribution drift. It is inspired by Daniel Kahneman's System 1 / System 2 thinking and James Clear's incremental improvement principles.

## Core Problem

When a production ML model encounters new data with a shifted distribution, its accuracy degrades. KC-Agent automatically:
1. Detects the degradation
2. Generates new training code
3. Executes and validates it
4. Stores successful strategies for future runs

## Dual-Process Design

```
Input: WorkingMemory (old model + old dataset + new dataset)
         |
         v
  ┌─────────────┐
  │  Fast Graph  │  System 1: quick, pattern-based retraining
  │   (S1)       │  Generates and executes retraining code
  └──────┬──────┘
         | fast_graph_metrics.yaml written to disk
         v
  ┌─────────────┐
  │  Slow Graph  │  System 2: deliberate, strategy-driven improvement
  │   (S2)       │  Runs only on first iteration (skipped if insights exist)
  └──────┬──────┘
         | slow_graph_metrics.yaml written to disk
         v
  slow_graph_update_episodic_memory()
         | stores deep_insight into EpisodicMemory
         v
Output: WorkingMemory with updated episodic memory
```

On subsequent runs, the Fast Graph reads `deep_insight` from `EpisodicMemory` and directly applies the previously discovered strategy, skipping the Slow Graph entirely.

## Fast Graph (System 1)

**File:** `caia/fast/fast_graph.py`

LangGraph nodes, executed in order:

1. `generate_retraining_code`: calls the LLM to write Python training code. If `deep_insight` exists in episodic memory, uses `prompt_generate_retraining_code_with_insights` to embed the prior strategy directly into the new code. Otherwise uses `prompt_generate_retraining_code`.

2. `execute_retraining_code`: runs the generated code via AutoGen's `LocalCommandLineCodeExecutor`. The code writes metrics to `old_metrics.yaml` (baseline) and `fast_graph_metrics.yaml` (retrained model). Captures stdout/stderr.

3. `fix_retraining_code` (conditional): if execution failed, calls the LLM with the error output to produce a corrected version. Loops back to execute. Bounded by `max_failures`.

**Decision:** after execution, checks if `δ(score) > threshold`. If yes, ends. If no, output is passed to the Slow Graph.

## Slow Graph (System 2)

**File:** `caia/slow/slow_graph.py`

LangGraph nodes:

1. `check_fast_graph_results`: reads the fast graph output to establish the baseline for improvement.

2. `distill_memories`: summarizes the current situation (dataset description, model state, prior improvement history) into a compact insight using `prompt_distill_memories`.

3. `analyze_needs`: decides which of three strategies to pursue using `prompt_analyze_improvement_needs`.

4. Strategy nodes (one selected per iteration):
   - `model_selection`: tries a different model family
   - `hyperparameter_tuning`: adjusts parameters of the current model
   - `ensemble_method`: combines multiple models

5. `apply_change`: generates and executes the "tiny change" code. Writes results to `slow_graph_metrics.yaml`.

6. `evaluate_change`: compares new metrics against old across both distributions. Records result as `ImprovementEntry` in `improvement_history`. Routes back to strategy selection or ends.

The slow graph runs up to `max_iterations` times. Each iteration appends to `improvement_history`.

## Memory Architecture

**File:** `caia/memory.py`

```
WorkingMemory (TypedDict)
├── semantic_memory: SemanticMemory
│     ├── dataset_old: Dataset        # CSV paths for old train/test data
│     ├── model_object: Any           # the trained sklearn model
│     └── model_code: str             # Python source of the training script
│
├── episodic_memory: DocList[EpisodicMemory]
│     └── EpisodicMemory
│           ├── dataset_new: Dataset  # CSV paths for new (drifted) data
│           ├── quick_insight: Dict   # lightweight notes from Fast Graph
│           └── deep_insight: Dict    # full strategy+code from Slow Graph
│                 ├── strategy: str   # 'model_selection' | 'hyperparameter_tuning' | 'ensemble_method'
│                 ├── code: str       # the actual Python code that worked
│                 ├── changes: Dict   # description of what changed
│                 └── metrics: Dict   # scores that validated the improvement
│
├── generations_fast_graph: Dict      # raw LLM outputs from Fast Graph nodes
├── generations_slow_graph: Dict      # raw LLM outputs from Slow Graph nodes
├── improvement_history: List[ImprovementEntry]  # append-only log
├── threshold: float                  # minimum score delta to accept a change
└── max_iterations: int
```

`deep_insight` is the critical cross-run artifact. It is written by `slow_graph_update_episodic_memory()` after the Slow Graph completes, and read by the Fast Graph at the start of the next run to skip redundant reasoning.

## Theoretical Guarantee

Under three conditions — (1) each change is a "tiny" verifiable modification, (2) the LLM knowledge function is consistent, (3) the evaluation environment is stationary — KC-Agent guarantees monotonic improvement with high probability:

```
P(score_final >= score_initial + Σ improvements - ε) >= 1 - δ
```

## LLM Integration

The LLM is injected at construction time and is fully swappable:

```python
from langchain_openai import ChatOpenAI
from caia.improver import Improver

llm = ChatOpenAI(model="gpt-4o", temperature=0.9)
improver = Improver(llm, max_iterations=3, max_failures=3)
result = improver.run(working_memory)
```

`caia/utils.py` also provides `ChatOpenRouter`, a subclass of `ChatOpenAI` that routes through OpenRouter (uses `OPENROUTER_API_KEY`).
