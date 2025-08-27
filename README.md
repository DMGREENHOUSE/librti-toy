# LIBRTI Data Generators (Streamlit + CLI)

Synthetic, physics-inspired data generators for **Purification**, **Corrosion**, and **Neutronics + Activation** models — built to create clean 1D and multi-D datasets for **Gaussian Process (GP)** training and demos in the Amentum **LIBRTI** workshop.

> ⚠️ These are **toy** models for teaching and prototyping. They are **not** validated engineering models.

---

## Features

- **Three models** in one Streamlit app (tabs) + **single-model apps**
- **Input sources:**  
  **Auto sampling** (1D sweep or Multi-D uniform) • **Custom points** (editable table) • **Upload inputs** (CSV)
- **Append mode:** append new rows to a user-uploaded base dataset if columns are compatible
- **Preview** Inputs & Outputs separately; **download a single combined CSV**
- **Auto-collapse** preview after successful download (data persists via `st.session_state`)
- Reproducible randomness via user-set seed

---

## Repo Layout

```
librti-datagenerators/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ requirements.txt
├─ requirements-dev.txt
├─ environment.yml                 # optional (Conda)
├─ .streamlit/
│  └─ config.toml                  # optional theme/port
├─ data/
│  ├─ examples/                    # tiny CSVs you ship for demos
│  └─ generated/                   # gitignored; attendees save downloads here
├─ scripts/
│  ├─ run_combined.sh
│  ├─ run_purification.sh
│  ├─ run_corrosion.sh
│  └─ run_neutroact.sh
├─ src/
│  └─ librti_datagen/
│     ├─ __init__.py
│     ├─ cli.py                    # optional CLI (see examples below)
│     ├─ models/
│     │  ├─ __init__.py
│     │  ├─ purification.py        # compute()
│     │  ├─ corrosion.py           # compute()
│     │  └─ neutroact.py           # compute()
│     └─ ui/
│        ├─ __init__.py
│        ├─ shared_ui.py           # shared Streamlit utilities
│        └─ app/
│           ├─ combined_app.py     # 3-tab Streamlit app
│           ├─ purification_app.py
│           ├─ corrosion_app.py
│           └─ neutroact_app.py
└─ tests/
   ├─ test_models.py
   └─ test_cli.py
```

---

## Installation

### Option A — pip + venv
```bash
python -m venv .venv
source .venv/bin/activate            # Windows (PowerShell): .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option B — Conda (optional)
```bash
conda env create -f environment.yml
conda activate librti-datagenerators
```

---

## Quickstart (Streamlit)

> All commands assume repo root. On Windows PowerShell, set `PYTHONPATH` with  
> `setx PYTHONPATH "$PWD/src"` then restart the shell, or temporary: `$env:PYTHONPATH="src"`.

### Run the **combined** app (3 tabs)
```bash
export PYTHONPATH=src
streamlit run src/librti_datagen/ui/app/combined_app.py
```
Custom port:
```bash
streamlit run src/librti_datagen/ui/app/combined_app.py --server.port 8502
```

### Run a **single-model** app
```bash
# Purification
export PYTHONPATH=src
streamlit run src/librti_datagen/ui/app/purification_app.py

# Corrosion
export PYTHONPATH=src
streamlit run src/librti_datagen/ui/app/corrosion_app.py

# Neutronics + Activation
export PYTHONPATH=src
streamlit run src/librti_datagen/ui/app/neutroact_app.py
```

---

## Usage Examples (Streamlit)

### 1) Auto Sampling — **1D Sweep**
1. Open the Purification tab.
2. At the top, set **Random seed** (e.g., 42).
3. Select **Auto sampling** > global toggle **1D**.
4. Choose the variable to vary (e.g., `temperature_c`) and set:
   - `temperature_c min = 20`, `max = 500`, **Number of points = 100**.
   - Keep other parameters as fixed values in the expander.
5. Click **Generate 1D Purification Data**.
6. Inspect **Preview: Inputs** & **Preview: Outputs**.
7. Click **Download dataset CSV** → file like `purification_dataset.csv`.

> The combined CSV contains **inputs + outputs** column-wise, ready for GP regression.

---

### 2) Auto Sampling — **Multi-D Uniform**
1. In the Corrosion tab, select **Auto sampling** > global toggle **Multi-D**.
2. Set **Number of samples = 500**.
3. Expand **Set parameter ranges** and adjust min/max for each input if desired.
4. Click **Generate Multi-D Corrosion Data** and download.

---

### 3) **Custom Points** (manual rows)
1. In Neutronics + Activation, choose **Custom points**.
2. Use the editable table to add rows with exact values (e.g., a small DoE).
3. Click **Generate Neutronics + Activation Data from Custom Points**.
4. Download the combined CSV.

---

### 4) **Upload Inputs** (CSV → evaluate → combined output)
1. Prepare a CSV with expected input columns (see **Schemas** below).
2. Choose **Upload inputs**, select the CSV, then **Generate [...] from Uploaded Inputs**.
3. Download the combined dataset.

**Example CSV (Purification inputs)**
```csv
temperature_c,flow_rate_mps,impurity_ppm,additive_frac,bed_length_m
100,0.25,100,0.3,0.5
250,0.40,500,0.5,1.0
400,0.10,50,0.1,2.0
```

---

### 5) **Append Mode** (merge with a base dataset)
- Toggle **Append new data to an uploaded base dataset**.
- Upload a base dataset that already contains **both inputs and outputs columns** (same model).
- Generate a new batch; if columns match (order doesn’t matter), the app appends and the download becomes `*_appended.csv`.

**Append compatibility rules**
- Columns must match the **combined** schema (inputs + outputs).
- On mismatch, you’ll see a warning and the app will not append.

---

## Model Schemas

### Purification
- **Inputs**: `temperature_c`, `flow_rate_mps`, `impurity_ppm`, `additive_frac`, `bed_length_m`
- **Outputs**: `purification_efficiency` (0–1)

### Corrosion
- **Inputs**: `temperature_c`, `pH`, `flow_rate_mps`, `dissolved_oxygen_ppm`, `chloride_ppm`
- **Outputs**: `corrosion_rate_mm_per_year` (~0–20)

### Neutronics + Activation
- **Inputs**: `blanket_thickness_m`, `li6_enrichment_pct`, `multiplier_fraction`, `structural_fraction`, `coolant_temp_c`
- **Outputs**: `TBR` (0.5–2.0), `activation_index_1y` (arb., ~0–10)

---

## CLI Usage (Optional)

The CLI runs the same `compute(...)` functions without Streamlit, producing **combined** CSVs.

> Ensure `PYTHONPATH=src` and that your **inputs CSV** matches the model’s input columns.

```bash
export PYTHONPATH=src
python -m librti_datagen.cli --help
```

### Purification (1D example via CSV)
```bash
python -m librti_datagen.cli   --model purification   --inputs data/examples/purification_inputs.csv   --output data/generated/purification_dataset.csv   --seed 42   --noise-std 0.02
```

### Corrosion (multi-D CSV)
```bash
python -m librti_datagen.cli   --model corrosion   --inputs data/examples/corrosion_inputs.csv   --output data/generated/corrosion_dataset.csv   --seed 0   --noise-pct 0.10
```

### Neutronics + Activation
```bash
python -m librti_datagen.cli   --model neutroact   --inputs data/examples/neutroact_inputs.csv   --output data/generated/neutroact_dataset.csv   --seed 0   --noise-tbr 0.01   --noise-act 0.10
```

**Tip:** Use the Streamlit app to generate inputs only (by exporting the combined file and deleting output columns), then feed that to the CLI for reproducible batch runs.

---

## Example Workflows

### A. Teach GP basics with a **1D sweep**
1. Purification → Auto sampling (1D) on `temperature_c`, 100 points.
2. Small noise (e.g., 0.02).
3. Download combined dataset; fit a GP (RBF) live; show mean + 2σ.

### B. Compare GP behavior in **sparse multi-D**
1. Corrosion → Multi-D, 200 points across broad ranges.
2. Fit GP with Matérn 3/2; visualize slices or partial dependence.
3. Discuss hyperparameters & anisotropy.

### C. Build a **feasible set** demo
1. Neutronics + Activation → Custom points near constraints (e.g., TBR > 1.15).
2. Fit a GP classifier (feasible/infeasible) or regress TBR and threshold it.
3. Show uncertainty bands and acquisition logic.

---

## Development

Install dev tools:
```bash
pip install -r requirements-dev.txt
```

Lint/format:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Run tests:
```bash
pytest -q
```

---

## Troubleshooting

- **“Missing columns”** when uploading inputs: ensure your CSV matches the model’s **Inputs** exactly.
- **Append mode warning**: your base CSV must include **inputs + outputs** of the same model (order doesn’t matter).
- **Nothing happens on Generate**: make sure you selected an **Input Source** and completed its step (e.g., uploaded a CSV or edited the table).
- **Port conflict**: run Streamlit with `--server.port 8502` (or any free port).

---

## License

Choose a license (e.g., MIT) and place the text in `LICENSE`.

---

## Acknowledgements

Developed for the **Amentum LIBRTI** workshop to support GP training with 1D and multi-D datasets across purification, corrosion, and neutronics + activation examples.