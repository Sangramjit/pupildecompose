# pupildecompose

<div align="center">

<img src="Assets/PupilDecompose.png" width="300"/>

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![Scientific](https://img.shields.io/badge/domain-pupillometry-purple.svg)

### A Python package for physiological pupil-feature decomposition

</div>

<br>

# ✨ Overview

`pupildecompose` is a Python package designed for extracting physiologically meaningful  features from pupil responses.

The package provides:

- Automated pupil decomposition
- Physiological feature extraction
- Trial-wise and participant-wise analysis
- Multiple constriction-onset detection methods
- Visualization
- CSV export utilities


The package is suitable for:

- Pupillometry
- Cognitive Neuroscience
- Attention Research
- Eye Tracking
- PLR Analysis
- Cognitive Psychophysiology
- Computational Neuroscience

---

# 🧩 Extracted Features

The package decomposes pupil traces into the following features:

| Feature | Description |
|---|---|
| 🔹 Constriction Onset | Estimated PLR onset time |
| 🔹 Pre-PLR Dilation | Early transient dilation before constriction |
| 🔹 Constriction Rate | Slope of constriction phase |
| 🔹 Constriction Amplitude | Peak constriction magnitude |
| 🔹 Constriction Latency | Time to minimum pupil size |
| 🔹 Dilation Rate | Recovery slope after constriction |

---

# 📦 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/pupildecompose.git
cd pupildecompose
```

---

## Install Package

```bash
pip install -e .
```

---

# 📁 Expected DataFrame Structure

The package accepts a `pandas DataFrame`.

---

## Required Structure

| Column | Content |
|---|---|
| Column 1 | Participant ID |
| Column 2 | Condition (optional) |
| Remaining Columns | Pupil time-series samples |

---

## Example DataFrame

| Participant | Condition | T1 | T2 | T3 | T4 | ... |
|---|---|---|---|---|---|---|
| P1 | Happy | 1.02 | 1.03 | 1.01 | 0.99 | ... |
| P1 | Happy | 1.01 | 1.02 | 1.00 | 0.98 | ... |
| P2 | Neutral | 1.00 | 1.01 | 0.99 | 0.97 | ... |

---

## Notes

- Each row represents one trial.
- Pupil samples must be arranged horizontally.
- Different trial lengths are allowed.
- Shorter trials are automatically padded using `NaN`.
- Excessive NaNs trigger warnings during decomposition.

---

# ⚠️ Data Assumptions

The package assumes users already performed:

- Blink reconstruction
- Basic artifact correction
- Initial preprocessing
- Missing-value handling

Optional smoothing can still be applied within the package.

---

# 🚀 Quick Example

```python
import pandas as pd

from pupildecompose import run_decompose


# -------------------------------------------------
# Load dataframe
# -------------------------------------------------

df = pd.read_csv(
    "pupil_dataset.csv"
)


# -------------------------------------------------
# Run decomposition
# -------------------------------------------------

results = run_decompose(

    df,

    participant_col=0,

    condition_col=1,

    pupil_start_col=2,

    mode="trial",

    preprocess=True,

    preprocess_method="gaussian",

    sigma=1,

    sample_interval_ms=8.33,

    baseline_window=(0, 20),

    stim_onset_idx=0,

    dilation_window_ms=300,

    onset_method="wilcoxon",

    plot=True,

    save_csv=True
)
```

---

# ⚙️ Main Parameters

| Parameter | Description |
|---|---|
| `participant_col` | Participant column index |
| `condition_col` | Condition column index |
| `pupil_start_col` | First pupil-data column |
| `mode` | `"trial"` or `"participant"` |
| `sample_interval_ms` | Sampling interval in milliseconds |
| `baseline_window` | Baseline indices |
| `stim_onset_idx` | Stimulus onset index |
| `dilation_window_ms` | Dilation-rate fitting window |
| `onset_method` | `"wilcoxon"` or `"slope"` |

---

# 🧠 Constriction Onset Detection

Two onset-detection methods are currently available.

---

## 1️⃣ Wilcoxon Method

Uses:

- Sliding 100 ms windows
- Physiological latency constraints
- Wilcoxon signed-rank testing

Recommended for cleaner datasets.

```python
onset_method="wilcoxon"
```

---

## 2️⃣ Slope Method

Uses:

- Sustained negative slopes
- Local trend consistency

More robust for noisy pupil traces.

```python
onset_method="slope"
```

---

# 🧪 Analysis Modes

## 🔹 Trial Mode

Each trial is decomposed independently.

Useful for:

- Single-trial analysis
- Computational modeling
- Trial-level statistics

```python
mode="trial"
```

---

## 🔹 Participant Mode

Trials are averaged within participant and condition before decomposition.

Useful for:

- Participant-level analysis
- ERP-style averaging
- Cleaner PLR estimation

```python
mode="participant"
```

---

# 🛠 Optional Preprocessing

Currently supported:

✅ Gaussian smoothing

```python
preprocess=True
```

```python
preprocess_method="gaussian"
```

---

# 📊 Output

The package automatically generates:

- Feature dataframe
- Decomposition summary
- Trial-success report
- CSV output
- Pupil-response plots
- Feature comparison plots

---

# 📈 Visualization

The package generates:

## Pupil Time Course

- Mean pupil traces
- Confidence intervals
- Multi-condition overlays

---

## Feature Subplots

Separate subplots for:

- Constriction Onset
- Pre-PLR Dilation
- Constriction Rate
- Constriction Amplitude
- Constriction Latency
- Dilation Rate

with:

- SEM bars
- Multi-condition comparisons

---

# 🧪 Testing

Run tests using:

```bash
pytest -v
```

The repository includes:

- API tests
- Smoke tests
- Functional pipeline tests

---

# 📂 Repository Structure

```text
pupildecompose/
│
├── pupildecompose/
│   ├── dataframe.py
│   ├── preprocess.py
│   ├── features.py
│   ├── plotting.py
│   ├── pipeline.py
│   └── utils.py
│
├── tests/
├── example/
├── README.md
├── setup.py
└── pyproject.toml
```

---

# 🧠 Scientific Applications

The package can be used for:

- Attention Research
- Pupillometry Studies
- Cognitive Load Analysis
- Eye Tracking Experiments
- Computational Neuroscience

---

# 📌 Version

```text
v1.0
```

---

# 🤝 Contributions

Bug reports, feature requests, and scientific collaborations are welcome.

---

# 👨‍🔬 Author

**Sangramjit Maity**

---

<div align="center">

## ⭐ If this package helps your research, consider starring the repository and cite the author ⭐

</div>