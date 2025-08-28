import numpy as np
import pandas as pd
from typing import Dict, List

# -----------------------------
# Corrosion → dissolved composition (fractions)
# -----------------------------
# Inputs:  temperature_c (required)
#          + optional geometry: flow_rate_mps, wall_thickness_m, surface_area_m2
# Output:  comp_Fe, comp_Cr, comp_Ni, comp_Mn  (each row sums to 1.0)
#
# Rationale: simple Arrhenius + flow/mass-transfer tilt; normalized to a composition vector.
# noise_pct controls Dirichlet-like dispersion around the mean composition.

ELEMENTS: List[str] = ["Fe", "Cr", "Ni", "Mn"]
BASE_ALLOY_FRAC: Dict[str, float] = {"Fe": 0.70, "Cr": 0.18, "Ni": 0.10, "Mn": 0.02}  # nominal source

# Synthetic Arrhenius params (per element)
A: Dict[str, float]  = {"Fe": 4.0e5, "Cr": 1.8e5, "Ni": 2.5e5, "Mn": 5.0e5}
Ea: Dict[str, float] = {"Fe": 4.5e4, "Cr": 6.0e4, "Ni": 5.5e4, "Mn": 4.0e4}  # J/mol

# Flow sensitivity (dimensionless multiplier on 1 + beta*flow)
BETA_FLOW: Dict[str, float] = {"Fe": 0.8, "Cr": 0.4, "Ni": 0.6, "Mn": 0.9}

# Geometry (optional)
GEOMETRY_COLUMNS = ["flow_rate_mps", "wall_thickness_m", "surface_area_m2"]
GEOMETRY_DEFAULTS: Dict[str, float] = {"flow_rate_mps": 0.2, "wall_thickness_m": 0.010, "surface_area_m2": 10.0}

INPUT_COLUMNS  = ["temperature_c"] + GEOMETRY_COLUMNS
OUTPUT_COLUMNS = [f"comp_{e}" for e in ELEMENTS]

R = 8.314  # J/(mol*K)

def _geom(inputs: pd.DataFrame, col: str) -> np.ndarray:
    if col in inputs.columns:
        return pd.to_numeric(inputs[col], errors="coerce").fillna(GEOMETRY_DEFAULTS[col]).to_numpy()
    return np.full(len(inputs), GEOMETRY_DEFAULTS[col], dtype=float)

def compute(inputs: pd.DataFrame, noise_pct: float, rng: np.random.Generator) -> pd.DataFrame:
    if "temperature_c" not in inputs.columns:
        raise ValueError("Expected 'temperature_c' column in inputs.")

    T_K = pd.to_numeric(inputs["temperature_c"], errors="coerce").fillna(20.0).to_numpy() + 273.15
    flow = np.clip(_geom(inputs, "flow_rate_mps"), 0.0, None)
    thk  = np.clip(_geom(inputs, "wall_thickness_m"), 1e-6, None)
    area = np.clip(_geom(inputs, "surface_area_m2"),  1e-9, None)

    area_thick = area / thk  # simple geometric driver

    # Unnormalized weights
    W = np.zeros((len(T_K), len(ELEMENTS)), dtype=float)
    for j, el in enumerate(ELEMENTS):
        k = A[el] * np.exp(-Ea[el] / (R * T_K))
        f_flow = 1.0 + BETA_FLOW[el] * flow
        w = BASE_ALLOY_FRAC[el] * k * np.power(np.maximum(1e-9, area_thick), 0.2) * f_flow
        W[:, j] = w

    W = np.clip(W, 1e-20, None)
    comp = W / W.sum(axis=1, keepdims=True)

    # Dirichlet-like noise
    if noise_pct and noise_pct > 0:
        s = float(np.clip(1.0 / (noise_pct**2 + 1e-9), 8.0, 200.0))  # larger s -> tighter
        comp_eps = np.clip(comp, 1e-6, None)
        comp_eps = comp_eps / comp_eps.sum(axis=1, keepdims=True)
        noisy = np.empty_like(comp_eps)
        for i in range(len(comp_eps)):
            noisy[i, :] = rng.dirichlet(comp_eps[i, :] * s)
        comp = noisy

    return pd.DataFrame({f"comp_{el}": comp[:, j] for j, el in enumerate(ELEMENTS)}, index=inputs.index)
