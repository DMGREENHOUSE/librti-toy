import numpy as np
import pandas as pd
from typing import Dict, List

# -----------------------------
# Neutronics + Activation (with flux in 1e23 units)
# -----------------------------
# Inputs:
#   - comp_* : composition fractions (comp_Fe, comp_Cr, comp_Ni, comp_Mn)   [0..1, sum~1]
#   - inv_*  : current radionuclide inventories (e.g., inv_Fe59, inv_Cr51, inv_Ni59, inv_Mn54)
#   - blanket_thickness_m : geometry factor (optional)
#   - neutron_flux_e23    : neutron flux in units of (1e23 n·m^-2·s^-1)
#   - exposure_time_s     : irradiation window (s)
#
# Back-compat: if 'neutron_flux' (absolute n·m^-2·s^-1) is present and 'neutron_flux_e23' is not,
#              we will convert internally via flux_e23 = neutron_flux / 1e23.
#
# Output:
#   - Updated inv_* columns with activation increment added.

ELEMENTS: List[str] = ["Fe", "Cr", "Ni", "Mn"]
ACT_TARGET: Dict[str, str] = {"Fe": "inv_Fe59", "Cr": "inv_Cr51", "Ni": "inv_Ni59", "Mn": "inv_Mn54"}

# Synthetic activation "yields" (scales)
YIELD: Dict[str, float] = {"Fe": 2.0e-20, "Cr": 2.5e-20, "Ni": 1.8e-20, "Mn": 3.0e-20}

DEFAULTS = {
    "blanket_thickness_m": 0.5,
    "neutron_flux_e23": 1.0e-7,    # ~1e16 n·m^-2·s^-1
    "exposure_time_s": 3600.0,     # 1 hour
}

INPUT_COLUMNS  = (
    list(ACT_TARGET.values())
    + [f"comp_{e}" for e in ELEMENTS]
    + ["blanket_thickness_m", "neutron_flux_e23", "exposure_time_s"]
)
OUTPUT_COLUMNS = list(ACT_TARGET.values())  # plus any passthrough inv_* present

def _col(inputs: pd.DataFrame, name: str, default: float) -> np.ndarray:
    if name in inputs.columns:
        return pd.to_numeric(inputs[name], errors="coerce").fillna(default).to_numpy()
    return np.full(len(inputs), default, dtype=float)

def _get_flux_e23(df: pd.DataFrame) -> np.ndarray:
    """Return flux in units of 1e23 n·m^-2·s^-1 (dimensionless). Back-compat for 'neutron_flux'."""
    if "neutron_flux_e23" in df.columns:
        return _col(df, "neutron_flux_e23", DEFAULTS["neutron_flux_e23"])
    if "neutron_flux" in df.columns:
        flux_abs = _col(df, "neutron_flux", DEFAULTS["neutron_flux_e23"] * 1e23)  # absolute (n·m^-2·s^-1)
        return flux_abs / 1e23
    return np.full(len(df), DEFAULTS["neutron_flux_e23"], dtype=float)

def compute(inputs: pd.DataFrame, noise_tbr: float, noise_act: float, rng: np.random.Generator) -> pd.DataFrame:
    """
    Update radionuclide inventory using composition and irradiation parameters.
    (noise_tbr is unused; present for UI compatibility.)
    """
    df = inputs.copy()

    # Composition (missing => 0)
    comp = {e: pd.to_numeric(df.get(f"comp_{e}", 0.0)).fillna(0.0).to_numpy() for e in ELEMENTS}
    # Current inventory (include all inv_* columns present)
    inv = {c: pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy()
           for c in df.columns if str(c).startswith("inv_")}

    thickness = _col(df, "blanket_thickness_m", DEFAULTS["blanket_thickness_m"])
    flux_e23  = _get_flux_e23(df)                  # dimensionless (in 1e23 units)
    flux_abs  = flux_e23 * 1e23                    # convert back to n·m^-2·s^-1 for calculation
    t_s       = _col(df, "exposure_time_s", DEFAULTS["exposure_time_s"])

    # Simple attenuation with thickness (0..~1)
    atten = 1.0 - np.exp(-1.5 * np.clip(thickness, 0.0, None))

    # Activation increments per target radionuclide
    delta = {}
    for e in ELEMENTS:
        tgt = ACT_TARGET[e]
        frac = np.clip(comp[e], 0.0, 1.0)
        inc = frac * YIELD[e] * flux_abs * atten * t_s
        if noise_act and noise_act > 0:
            sigma = np.maximum(1e-12, noise_act * inc)
            inc = inc + rng.normal(0.0, sigma, size=inc.shape)
        delta[tgt] = np.clip(inc, 0.0, None)

    # Build outputs: pass-through all inv_* then add increments to targets
    out = {}
    for col in [c for c in df.columns if str(c).startswith("inv_")]:
        out[col] = inv[col].copy()
    for tgt in ACT_TARGET.values():
        if tgt not in out:
            out[tgt] = np.zeros(len(df))
    for tgt, d in delta.items():
        out[tgt] = np.clip(out[tgt] + d, 0.0, None)

    return pd.DataFrame(out, index=df.index)
