# Experiments and Results

## Setup

```bash
git clone <repo>
cd improver
uv venv --python 3.11.5
uv pip install -e .
```

Set your LLM API key before running any experiment notebook:

```bash
export OPENROUTER_API_KEY="sk-..."   # if using OpenRouter
# or
export OPENAI_API_KEY="sk-..."       # if using OpenAI directly
```

## Domains

Five synthetic datasets simulating distribution drift:

| Domain | Task | Drift type |
|---|---|---|
| `financial` | Binary classification (loan default) | Feature distribution shift |
| `healthcare` | Binary classification (patient outcome) | Feature distribution shift |
| `eligibility` | Binary classification (benefit eligibility) | Feature distribution shift |
| `nasa` | Binary classification (turbofan failure) | NASA FD001 sensor drift |
| `nasa-FD002` | Binary classification (turbofan failure) | NASA FD002 sensor drift |

Each domain lives in `datasets/{domain}/` with this structure:

```
X_train_old.csv   X_test_old.csv   y_train_old.csv   y_test_old.csv
X_train_new.csv   X_test_new.csv   y_train_new.csv   y_test_new.csv
```

To regenerate datasets: run `experiments/generate_dataset.ipynb` then `experiments/split_datasets.ipynb`.

## Running an Experiment

Open `experiments/dev5_{domain}.ipynb`. Each notebook:

1. Loads the old dataset and trains a baseline model
2. Saves the model code to a `SemanticMemory` object
3. Saves the new dataset reference to an `EpisodicMemory` object
4. Constructs `WorkingMemory` and calls `improver.run()`
5. Calls `save_fast_graph_results()` and `save_slow_graph_results()` to write to `results/`

### Changing the LLM

At the top of each notebook, the LLM is instantiated. Swap it freely:

```python
from langchain_openai import ChatOpenAI
from caia.utils import ChatOpenRouter

# OpenRouter (supports many models)
llm = ChatOpenRouter(
    model_name="meta-llama/llama-3.1-8b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.9,
)

# Direct OpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0.9)
```

### Changing Max Iterations

```python
improver = Improver(llm, max_iterations=3, max_failures=3)
```

- `max_iterations`: how many strategy attempts the Slow Graph makes
- `max_failures`: how many consecutive code execution errors before giving up

## Benchmarks

All baselines live in `caia/benchmark/`. Each is run from the same experiment notebooks with a shared interface.

| Agent | File | Description |
|---|---|---|
| Baseline | `baseline.py` | Single LLM call: improve this code |
| ReAct | `react.py` | Reason + Act loop with tool access |
| Reflexion | `reflection.py` | Generate → reflect → refine loop |
| Self-Discovery | `self_discover.py` | Select → adapt → structure → reason |
| Tree of Thoughts | `tot.py` | Branch and evaluate multiple reasoning paths |
| Plan-and-Execute | `plan_and_execute.py` | Plan steps then execute sequentially |
| CodeAct | `codeact.py` | Multi-turn code generation with stateful execution |
| KC-Agent (fast only) | `fast/fast_graph.py` | System 1 only |
| KC-Agent (full) | `improver.py` | System 1 + System 2 |

## Results Structure

### Per-run files: `results/`

Every experiment run appends a YAML file:

```
{agent}_temp_{t}_max_iter_{n}_llm_{model}_dataset_{domain}_{hash}.yaml
```

Example: `improver_temp_0.9_max_iter_1_llm_llama-3.1-8b-instruct_dataset_financial_73bc59c4.yaml`

Each file contains:
```yaml
agent_name: improver
initial_metrics: {new_distribution: 0.71, old_distribution: 0.91}
final_metrics: {new_distribution: 0.84, old_distribution: 0.89}
improvement_path: [...]
runtime_statistics:
  total_time_seconds: 142.3
  tokens_used: 18400
```

### Aggregated files: `data/`

These are committed and used directly by `results.ipynb`:

- `summary_metrics_{domain}.yaml`: aggregated per-domain results across all agents and runs. Written by `results.ipynb`.
- `llm_evals_{model}.yaml`: scores from LLM judges (Gemini, GPT, Opus) evaluating agent output quality
- `human_evaluation.yaml`: human evaluation scores
- `metrics_{baseline}.yaml`: snapshot metrics for each baseline agent

### Intermediate files (gitignored)

Written to the working directory during experiment runs, then read by `results.ipynb`:

- `old_metrics.yaml`: baseline model scores on both distributions
- `fast_graph_metrics.yaml`: scores after Fast Graph retraining
- `slow_graph_metrics.yaml`: scores after Slow Graph improvement
- `initial_code_{domain}.yaml`: the starting model code for a domain
- `initial_metrics_{domain}.yaml`: the starting metrics for a domain

These are regenerated on every run and should not be committed.

## Analyzing Results

Open `results.ipynb` from the repo root. Run cells in order:

1. **Load results**: reads all `results/*.yaml` files via glob, parses into a DataFrame
2. **Aggregate**: groups by agent, dataset, LLM; computes mean/std for `new_distribution` and `old_distribution` accuracy
3. **Save**: writes `data/summary_metrics_{domain}.yaml` and `paper_figures/final_aggregated_results.csv`
4. **Plots**: generates bar charts, radar plots, bubble charts, heatmaps into `paper_figures-{domain}/` and `enhanced_figures/`
5. **LLM eval tables**: loads `data/llm_evals_*.yaml` and formats LaTeX tables
6. **Human eval table**: loads `data/human_evaluation.yaml` and formats LaTeX table
7. **Statistical tests**: Wilcoxon, Mann-Whitney U, paired t-tests comparing KC-Agent vs. baselines

The notebook assumes it is run from the repo root (relative paths like `results/`, `data/`, `paper_figures/` are all relative to root).

## Key Metrics

| Metric | Meaning |
|---|---|
| `on_old_data` | Accuracy on the original distribution (stability) |
| `on_new_data` | Accuracy on the drifted distribution (adaptability) |
| `improvement_new` | Delta on new data vs. baseline |
| `improvement_old` | Delta on old data vs. baseline (should not regress) |
| tokens_used | Total prompt + completion tokens consumed |
| total_time_seconds | Wall-clock time for the full agent run |

A good result: `improvement_new > 0` and `improvement_old >= -epsilon` (improved on new data without forgetting old data).
