# Codebase Guide

## Repository Layout

```
caia/                    Core Python package
  __init__.py            Empty (namespace only)
  improver.py            Top-level Improver class + episodic memory update logic
  memory.py              All TypedDicts and DocArray schemas
  prompts.py             All LangChain prompt factories (~3300 lines)
  tools.py               Evidently drift, SHAP, TrustScore wrappers
  representation.py      DatasetRepresentation DocArray schema
  insight.py             Insight-related utilities
  utils.py               ChatOpenRouter, save/load helpers, visualization, misc
  fast/
    fast_graph.py        FastGraph class (LangGraph)
  slow/
    slow_graph.py        SlowGraph class (LangGraph)
  benchmark/
    baseline.py          Baseline: single LLM call to improve code
    react.py             ReAct agent
    reflection.py        Reflexion agent
    self_discover.py     Self-Discovery agent
    tot.py               Tree of Thoughts agent
    plan_and_execute.py  Plan-and-Execute agent
    codeact.py           CodeAct agent

datasets/                CSV data per domain
  financial/             X_train_old.csv, X_test_old.csv, y_train_old.csv, ...
  healthcare/            (same structure)
  eligibility/           (same structure)
  nasa/                  (same structure)
  nasa-FD002/            (same structure)

experiments/             Active experiment notebooks
  dev5_financial.ipynb   Latest KC-Agent run on financial domain
  dev5_healthcare.ipynb
  dev5_eligibility.ipynb
  dev5_nasa.ipynb
  dev5_nasa2.ipynb
  generate_dataset.ipynb Creates synthetic datasets with drift
  split_datasets.ipynb   Splits datasets into old/new distributions
  nasa_turbofan_dataset.ipynb   NASA-specific data prep
  nasa_turbofan_dataset2.ipynb

archive/                 Older notebooks + generated artifacts (gitignored)
results/                 Per-run YAML files (one per experiment run)
data/                    Committed result summaries and evaluation scores
paper_figures*/          Publication figures, one dir per domain
enhanced_figures/        Additional figures
imgs/                    Static images (improver_agent.png architecture diagram)
docs/                    This documentation
```

## Key Files in Detail

### `caia/memory.py`

Defines all data structures. The most important ones:

- `WorkingMemory`: the single state dict threaded through both graphs
- `SemanticMemory(BaseDoc)`: holds the old dataset paths and the model's Python source code
- `EpisodicMemory(BaseDoc)`: one entry per improvement session; `deep_insight` is the key field written by the Slow Graph
- `ImprovementEntry(TypedDict)`: one record per attempted change, stored in `improvement_history`
- `create_improvement_entry()`: helper that computes improvement deltas and determines outcome (`success`/`failure`)

### `caia/improver.py`

The orchestrator. Notable design issue (currently): two functions at module level (`fast_graph_generate_retraining_code` and `fast_graph_prepare_yaml_content_with_insights`) are monkey-patched onto the `FastGraph` instance at runtime. This was done to extend FastGraph behavior without subclassing.

`Improver.run()` flow:
1. Validates and initializes `WorkingMemory`
2. Runs `FastGraph.run()`
3. Checks if `deep_insight` already exists in episodic memory — if yes, skips Slow Graph
4. Runs `SlowGraph.run()` (first run only)
5. Calls `slow_graph_update_episodic_memory()` to persist the best strategy
6. Assembles `yaml_output` with runtime statistics and token counts

### `caia/prompts.py`

All LangChain `ChatPromptTemplate` factory functions. Key ones:

| Function | Used by | Purpose |
|---|---|---|
| `prompt_generate_retraining_code` | FastGraph | Generate training code from scratch |
| `prompt_generate_retraining_code_with_insights` | FastGraph | Generate code reusing prior strategy |
| `prompt_execute_and_fix_retraining_code` | FastGraph | Fix code given error output |
| `prompt_distill_memories` | SlowGraph | Summarize current situation |
| `prompt_analyze_improvement_needs` | SlowGraph | Choose strategy |
| `prompt_model_selection_change` | SlowGraph | Generate model-swap code |
| `prompt_hyperparameter_tuning` | SlowGraph | Generate hyperparameter code |
| `prompt_ensemble_method` | SlowGraph | Generate ensemble code |
| `prompt_evaluate_change` | SlowGraph | Assess the improvement |
| `prompt_fix_code` | Both | General code error correction |

### `caia/tools.py`

Three analysis tools used in the diagnostic layer:
- `get_drift_report(X_ref, X_cur)`: Evidently DataDrift report, returns per-feature drift scores
- `get_shap_values(model, X)`: TreeSHAP via alibi, returns ranked feature attributions
- `calculate_trust_score(model, X_train, X_test)`: alibi TrustScore for prediction confidence

These are wrapped in a `Tool(BaseDoc)` schema and registered via `get_tools()`.

### `caia/utils.py`

Mixed-purpose file. Main utilities:
- `ChatOpenRouter`: OpenAI-compatible LLM client pointing to openrouter.ai
- `save_yaml_results(state, path)`: dumps `state['yaml_output']` to YAML
- `save_fast_graph_results() / save_slow_graph_results()`: save structured CSVs and YAMLs to `results/`
- `plot_improvements(dataset_folder, llm_name)`: bar chart comparing baseline → fast → slow scores
- `print_function_name(func)`: decorator that prints a Rich panel when a graph node executes

Also contains report generators that duplicate benchmark functionality (from older code): `ReflectionReportGenerator`, `ReactReportGenerator`, `SelfDiscoverReportGenerator`, `PlanAndExecuteReportGenerator`.

### `caia/benchmark/`

Each file follows the same pattern:
- `StandardState(TypedDict)`: state with `model_code`, `improved_code`, `execution_output`, `metrics`, `improvement_history`, `token_usage`
- A class (e.g., `BaselineImprover`, `ReactImprover`) with a LangGraph workflow
- Code is executed via AutoGen's `LocalCommandLineCodeExecutor`
- Results written to a YAML file named `{agent}_{params}_{dataset}_{hash}.yaml` in `results/`

**CodeAct** (`codeact.py`) is the newest benchmark. It uses a `SimpleCodeExecutor` that maintains persistent globals/locals across turns (stateful execution), and a multi-turn conversation loop where the LLM emits code in `<execute>...</execute>` tags.

## Data Flow for a Single Run

```
1. User creates WorkingMemory:
   - SemanticMemory: old CSV paths + model.py source code
   - EpisodicMemory: new CSV paths, empty insights

2. improver.run(working_memory)

3. FastGraph generates Python code via LLM
   → AutoGen executes it in a subprocess
   → Code writes: old_metrics.yaml, fast_graph_metrics.yaml
   → State updated with ImprovementEntry

4. SlowGraph (if no prior deep_insight):
   → Distills memories
   → Picks strategy (model_selection / hyperparameter_tuning / ensemble)
   → Generates "tiny change" code
   → AutoGen executes it
   → Code writes: slow_graph_metrics.yaml
   → Loops up to max_iterations

5. slow_graph_update_episodic_memory():
   → Finds best ImprovementEntry (highest new_distribution improvement)
   → Saves strategy + code + metrics as deep_insight

6. save_fast/slow_graph_results() (called from notebook):
   → Copies metric YAMLs to results/ with structured names
   → Writes CSV summary rows
```

## Adding a New Benchmark

1. Create `caia/benchmark/my_agent.py` following the pattern in `baseline.py`
2. Define `StandardState` (or reuse it), a class with a LangGraph graph
3. Write results to `results/my_agent_{params}_{dataset}_{hash}.yaml`
4. Add a corresponding cell in the relevant `experiments/dev5_*.ipynb`
5. Add the agent name to the `name_mapping` dict in `results.ipynb`
