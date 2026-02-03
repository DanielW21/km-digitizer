# Kaplan-Meier Digitization Project

**LLM-powered extraction and evaluation of Kaplan-Meier survival curves**

This repository provides tools for automatically extracting data points from Kaplan-Meier (KM) survival plots using large language models and evaluating the reconstruction quality with KM-GPT style metrics.

## Features

- **LLM-Based Extraction**: Uses Google's Gemini API to extract survival curve data from plot images
- **KM-GPT Metrics**: Comprehensive evaluation metrics including Integrated Absolute Error (IAE) and median survival error
- **Iterative Refinement**: Agentic loop with validation and feedback for improved accuracy
- **Batch Processing**: Evaluate multiple datasets with summary statistics
- **Synthetic Data Generation**: Generator for creating benchmark datasets

## Installation

```bash
git clone <repository-url>
cd km-digitizer
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Configure your environment:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

## Usage

### Extract Data from KM Plots

```bash
python src/demo.py
```

This processes all images in `data/synthetic_data/benchmark/plots/` and `data/real_data/`, saving results to timestamped folders in `logs/`.

### Evaluate Pipeline Performance

```bash
python src/eval.py logs/SESSION_TIMESTAMP --data-dir data --output eval_logs/results.json
```

This compares extracted data against ground truth and computes KM-GPT style metrics.

### Programmatic Usage

**Extract data from a single plot:**
```python
from src.llm_analysis import KMDigitizer

digitizer = KMDigitizer()
data = digitizer.run("path/to/km_plot.png", max_attempts=5)
print(data)  # {"Group1": [[x1, y1], [x2, y2], ...], ...}
```

**Evaluate reconstruction quality:**
```python
from src.metrics import KMMetrics

metrics = KMMetrics()
results = metrics.evaluate_reconstruction(
    truth_durations, truth_events,
    reconstructed_durations, reconstructed_events
)
print(f"IAE: {results['integrated_absolute_error']:.6f}")
```

**Batch evaluation:**
```python
from src.metrics import BatchMetricsEvaluator

evaluator = BatchMetricsEvaluator()
for dataset_id, (truth_data, recon_data) in datasets.items():
    evaluator.evaluate_dataset(dataset_id, truth_data[0], truth_data[1],
                              recon_data[0], recon_data[1])

summary = evaluator.compute_summary_statistics()
print(f"Median IAE: {summary['iae']['median']:.6f}")
```

## Project Structure

```
src/
├── llm_analysis/          # LLM-based extraction
├── metrics/               # Evaluation metrics
├── demo.py               # Main extraction script
└── eval.py               # Evaluation script
data/
├── real_data/            # Real KM plot images
├── synthetic_data/       # Generated benchmark data
└── generators/           # Data generation scripts
logs/                     # Output directory (auto-created)
```

## Metrics

The project implements KM-GPT style evaluation metrics:

- **Integrated Absolute Error (IAE)**: Area between truth and reconstructed curves
- **Median OS Error**: Difference in median overall survival times
- **Batch Statistics**: 95% confidence intervals across multiple datasets

Example output:
```
Median IAE: 0.018432 (95% CI: 0.002134-0.088756)
```

## Environment

Required environment variable:
```bash
GEMINI_API_KEY=your_api_key_here
```
