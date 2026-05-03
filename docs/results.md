# results.ipynb Walkthrough

`results.ipynb` is the single analysis notebook for the paper. It loads all experiment runs, aggregates them, produces the figures and LaTeX tables used in the publication, and runs statistical significance tests.

**Must be run from the repo root** — all file paths are relative to that location.

---

## Inputs

| File / Directory | Required | Description |
|---|---|---|
| `results/*.yaml` | Yes | Per-run experiment files written by each agent |
| `data/llm_evals_gemini3pro.yaml` | Yes | LLM judge scores from Gemini |
| `data/llm_evals_gpt5-2.yaml` | Yes | LLM judge scores from GPT |
| `data/llm_evals_opus4-5.yaml` | Yes | LLM judge scores from Claude Opus |
| `data/human_evaluation.yaml` | Yes | Human annotator scores |
| `paper_figures/final_aggregated_results.csv` | Auto | Written by this notebook, then re-read later |
| `initial_code_{domain}.yaml` | No | Optional; used to embed initial code in summaries. Falls back to empty string if missing |
| `initial_metrics_{domain}.yaml` | No | Optional; used to embed initial metrics. Falls back to empty dict if missing |

### Format of `results/*.yaml`

Each file is named `{agent}_temp_{t}_max_iter_{n}_llm_{model}_dataset_{domain}_{hash}.yaml` and contains:

```yaml
agent_name: improver
initial_metrics:
  new_distribution: 0.71
  old_distribution: 0.91
final_metrics:
  new_distribution: 0.84
  old_distribution: 0.89
improvement_path: [...]
runtime_statistics:
  total_time_seconds: 142.3
  tokens_used: 18400
  prompt_tokens: 14200
  completion_tokens: 4200
```

### Format of `data/llm_evals_*.yaml`

Each LLM judge file contains scores per agent per dataset across three dimensions:

```yaml
financial:
  improver:
    validity: 8.5
    quality: 7.0
    novelty: 6.5
  react:
    validity: 6.0
    ...
healthcare:
  ...
```

### Format of `data/human_evaluation.yaml`

Same structure as LLM eval files — domain → agent → {validity, quality, novelty}.

---

## Outputs

| File / Directory | Description |
|---|---|
| `paper_figures/final_aggregated_results.csv` | Central aggregated DataFrame, used by all downstream cells |
| `data/summary_metrics_{domain}.yaml` | Per-domain summary with initial code, initial metrics, and per-agent results |
| `paper_figures-{domain}/` | Per-dataset bar charts and distribution plots |
| `enhanced_figures/` | Evaluator breakdown faceted bar chart |
| LaTeX strings | Printed to cell output (not saved to disk) — copy-paste into paper |

---

## Section-by-Section Walkthrough

### 1. Imports and Styling (Cell 2)

Sets up the environment: imports numpy, pandas, matplotlib, seaborn, scipy. Configures matplotlib for publication-ready output (font family, sizes, color palette). Defines `COLORS` using a seaborn palette.

No file I/O here.

### 2. Load and Aggregate All Results (Cell 3)

**Core function:** `analyze_results(results_dir, output_dir, dataset)`

For each of the five domains (`healthcare`, `eligibility`, `financial`, `nasa`, `nasa-FD002`):
- Globs all `results/*.yaml` files
- Filters by dataset name using regex on the filename
- Parses each YAML into a row: `agent`, `llm`, `temperature`, `max_iter`, `initial_new`, `initial_old`, `final_new`, `final_old`, `improvement_new`, `improvement_old`, `exec_time`, `tokens_used`, `success`
- Computes `avg_final_combined = (final_new + final_old) / 2`
- Groups by `agent` → computes mean, std, success rate
- Saves per-dataset bar charts into `paper_figures-{domain}/`
- Returns three DataFrames: `df` (raw rows), `agg_df` (aggregated per agent), `sig_df` (significance)

All five domain results are concatenated into **`final_agg_df`** — the central DataFrame used by everything downstream.

### 3. Rename Agents (Cell 4)

Maps internal agent names to paper-friendly display names:

| Internal | Paper name |
|---|---|
| `baseline` / `standard` | Standard |
| `improver` | KC-agent |
| `fast` | KC-fast |
| `slow` | KC-slow |
| `react` | ReAct |
| `reflection` | Reflexion |
| `self_discovery` | Self-Discovery |
| `plan_execute` | Plan-Execute |
| `tot` | ToT |
| `codeact` | CodeAct |

### 4. Save Aggregated CSV (Cell 5)

Saves `final_agg_df` to `paper_figures/final_aggregated_results.csv`. Also renames `nasa` → `nasa-FD001` for the paper.

This CSV is the checkpoint — if you want to skip re-parsing all YAML files, comment out Cells 3–5 and uncomment the `pd.read_csv(...)` line to load from here directly.

### 5. Build Summary YAML Files (Cell 8)

**Function:** `build_summary_metrics(datasets)`

For each domain:
- Reads `initial_code_{domain}.yaml` and `initial_metrics_{domain}.yaml` (optional)
- For each YAML in `results/` matching the domain, reads the full agent result
- Writes `data/summary_metrics_{domain}.yaml` with structure:

```yaml
initial_code: |
  <python source code>
initial_metrics:
  new_distribution: 0.71
  old_distribution: 0.91
improver:
  final_metrics: {new_distribution: 0.84, old_distribution: 0.89}
  improvement_path: [...]
  runtime_statistics: {...}
react:
  ...
```

This is the most information-dense output: it stores the actual code alongside metrics.

### 6. LLM Judge Evaluation Tables (Cells 11–12)

**Function:** `create_averaged_evaluation_table(yaml_files)`

Reads all three `data/llm_evals_*.yaml` files, averages scores across LLM judges and datasets, and prints a LaTeX table with three columns: Validity, Quality, Novelty. Two variants:
- Cell 11: simple averages
- Cell 12: averages with standard deviation (±)

Output: LaTeX `tabular` string printed to cell output.

### 7. Evaluator Breakdown Plot (Cell 13)

**Function:** `create_evaluator_breakdown_plot(output_dir="enhanced_figures")`

Reads all three LLM eval files. Produces a faceted bar chart (one facet per judge: Gemini, GPT, Opus), showing how each judge scored each agent on the three metrics.

Output: saved to `enhanced_figures/evaluator_breakdown.png` (or similar name).

### 8. Reload CSV (Cell 15)

```python
final_agg_df = pd.read_csv("paper_figures/final_aggregated_results.csv")
```

Re-reads from the checkpoint CSV. All subsequent cells work from this DataFrame.

### 9. Main Performance Table (Cells 16–17)

**Function:** `create_latex_table(final_agg_df)`

Aggregates `final_agg_df` by agent (across all datasets), computes mean for:
- `avg_final_combined` — combined accuracy (new + old) / 2
- `avg_exec_time` — wall-clock seconds
- `avg_tokens_used` — total tokens consumed

Sorts descending by accuracy. Best values in each column are bolded (`\textbf{}`). Output: LaTeX `table` environment printed to cell output.

### 10. Ablation Study Table (Cell 19)

**Function:** `create_ablation_table(final_agg_df)`

Same as the main table but filtered to KC-* agents only (`KC-agent`, `KC-slow`, `KC-fast`). Shows the contribution of each component: running both graphs vs. only Slow vs. only Fast.

Output: LaTeX table string.

### 11. Human Evaluation Table (Cell 21)

**Function:** `create_human_evaluation_table(yaml_file="data/human_evaluation.yaml")`

Reads `data/human_evaluation.yaml`, averages Validity / Quality / Novelty scores per agent across all domains, and formats a LaTeX table. Handles missing agent entries gracefully (skips them).

Output: LaTeX table string.

### 12. Statistical Tests (Cells 23–24)

**Functions:** `perform_pairwise_tests()`, `perform_kc_vs_all_tests()`

Runs pairwise significance tests comparing KC-agent against every other agent using:
- Wilcoxon signed-rank test (paired, non-parametric)
- Mann-Whitney U test (unpaired, non-parametric)
- Paired t-test

Metric tested: `avg_final_combined` by default (configurable). Only includes agents with success rate ≥ 0.5.

Output: p-values printed per pair; also can generate a LaTeX significance table.

### 13. Radar Plot (Cell 31)

Reads `agg_df` (single-dataset slice), maps agent names, and draws a radar chart comparing agents across multiple metrics simultaneously. Good for visualizing multi-dimensional trade-offs.

Output: displayed inline; optionally saved.

### 14. Bubble Chart (Cell 34)

**Function:** `plot_bubble_performance(agg_df)`

Creates a scatter plot where:
- X-axis: average execution time (seconds)
- Y-axis: combined accuracy
- Bubble size: tokens used

Useful for the efficiency-accuracy trade-off. Each agent is a bubble with its name annotated.

Output: displayed inline; optionally saved to `output_dir`.

---

## Key DataFrame: `final_agg_df`

The central data structure throughout the notebook. Schema after Cell 5:

| Column | Type | Description |
|---|---|---|
| `agent` | str | Display name (e.g., KC-agent, ReAct) |
| `dataset` | str | Domain name (e.g., financial, nasa-FD001) |
| `avg_final_combined` | float | Mean (new + old accuracy) / 2 across runs |
| `avg_final_new` | float | Mean accuracy on new (drifted) distribution |
| `avg_final_old` | float | Mean accuracy on original distribution |
| `avg_exec_time` | float | Mean wall-clock seconds |
| `avg_tokens_used` | float | Mean total tokens |
| `success_rate` | float | Fraction of runs with non-null final metrics |
| `std_final_combined` | float | Std dev of combined accuracy |

---

## Common Usage Patterns

**Re-run everything from scratch (new experiment results added):**
Run all cells top to bottom.

**Re-generate plots only (results haven't changed):**
Skip to Cell 15 (`pd.read_csv(...)`) and run from there.

**Add a new agent to all tables:**
1. Make sure its result YAMLs are in `results/` with the correct filename pattern
2. Add its name to the `name_mapping` dict in Cells 4 and 11–13
3. Re-run from Cell 3

**Change which LLM judge files are used:**
Edit the `yaml_files` list in Cells 11/12 to point to different `data/llm_evals_*.yaml` files.
