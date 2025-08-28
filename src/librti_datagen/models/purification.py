import numpy as np
import pandas as pd
from typing import Dict, List

# -----------------------------
# Purification → inventory updater
# -----------------------------
# Inputs:  any number of radionuclide inventory columns (prefix 'inv_')
#          + optional purifier geometry (bed_length_m, flow_rate_mps, porosity, surface_area_m2, temperature_c, additive_frac)
# Output:  updated inventory (same 'inv_*' columns), after a single pass / time step.
#
# The model computes a removal efficiency shaped by geometry, and applies it per species.
# You can customize selectivity via SELECTIVITY.

GEOMETRY_COLUMNS = ["bed_length_m", "flow_rate_mps", "porosity", "surface_area_m2", "temperature_c", "additive_frac"]
GEOMETRY_DEFAULTS: Dict[str, float] = {
    "bed_length_m": 1.0,
    "flow_rate_mps": 0.2,
    "porosity": 0.40,
    "surface_area_m2": 10.0,
    "temperature_c": 300.0,
    "additive_frac": 0.0,
}

# Species-specific selectivity multipliers (1.0 = neutral). Example:
# SELECTIVITY = {"inv_Mn54": 1.2, "inv_Co60": 0.8}
SELECTIVITY: Dict[str, float] = {}

INPUT_COLUMNS: List[str]  = []  # dynamic (all inv_* present + optional geometry)
OUTPUT_COLUMNS: List[str] = []  # same inv_* columns

def _g(df: pd.DataFrame, name: str) -> np.ndarray:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(GEOMETRY_DEFAULTS[name]).to_numpy()
    return np.full(len(df), GEOMETRY_DEFAULTS[name], dtype=float)

def compute(inputs: pd.DataFrame, noise_std: float, rng: np.random.Generator) -> pd.DataFrame:
    """Apply a single-step purification to any inv_* columns present."""
    df = inputs.copy()
    inv_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("inv_")]
    if not inv_cols:
        raise ValueError("No radionuclide inventory columns found. Expected columns starting with 'inv_'.")

    # Geometry-driven efficiency
    bed_L = np.clip(_g(df, "bed_length_m"), 1e-6, None)
    flow  = np.clip(_g(df, "flow_rate_mps"), 1e-9, None)
    poro  = np.clip(_g(df, "porosity"),     1e-6, 0.999999)
    area  = np.clip(_g(df, "surface_area_m2"), 1e-9, None)
    T_c   = _g(df, "temperature_c")
    addf  = np.clip(_g(df, "additive_frac"), 0.0, 1.0)

    contact_time = bed_L / (1e-6 + flow)
    area_fac = np.log1p(area)
    poro_fac = 1.0 - 0.8 * np.abs(poro - 0.40)
    temp_fac = 1.0 + 0.002 * (T_c - 300.0)
    addi_fac = 1.0 + 0.5 * addf

    k_eff = 0.08 * area_fac * np.maximum(0.1, poro_fac) * np.maximum(0.1, temp_fac) * addi_fac
    eff = 1.0 - np.exp(-k_eff * contact_time)           # 0..~1
    eff = np.clip(eff, 0.0, 0.999)

    out = {}
    for col in inv_cols:
        inv_in = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        sel = float(SELECTIVITY.get(col, 1.0))
        eff_i = np.clip(sel * eff, 0.0, 0.999)
        inv_out = inv_in * (1.0 - eff_i)

        if noise_std and noise_std > 0:
            sigma = np.maximum(1e-12, noise_std * np.abs(inv_out))
            inv_out = inv_out + rng.normal(0.0, sigma, size=inv_out.shape)

        out[col] = np.clip(inv_out, 0.0, None)

    return pd.DataFrame(out, index=df.index)
