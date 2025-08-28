from __future__ import annotations
import streamlit as st
import numpy as np
from librti_datagen.models import purification
from librti_datagen.ui.shared_ui import run_tab

st.set_page_config(page_title="Purification Data Generator", layout="wide")
st.title("Purification → Inventory Update")

seed = st.number_input("Random seed", min_value=0, max_value=2**31-1, value=42, step=1)
rng = np.random.default_rng(seed)
sampling_dim = st.radio("Auto sampling dimensionality", ["1D", "Multi-D"], horizontal=True)

st.header("Purification Model")
st.caption(
    "Inputs → **inv_*** radionuclide inventory (e.g., inv_Fe59, inv_Cr51, inv_Ni59, inv_Mn54; optionally inv_T) "
    "+ optional purifier geometry (bed_length_m, flow_rate_mps, porosity, surface_area_m2, temperature_c, additive_frac). "
    "Output → updated **inv_*** after a single pass/time-step removal."
)

def pur_noise_controls():
    val = st.slider("Relative noise on updated inventory (std fraction)", 0.0, 0.5, 0.02, 0.01, key="pur_noise")
    return {"noise_std": val}

pur_ranges = {
    "inv_Fe59": (0.0, 1e6),
    "inv_Cr51": (0.0, 1e6),
    "inv_Ni59": (0.0, 1e6),
    "inv_Mn54": (0.0, 1e6),
    "inv_T": (0.0, 1e6),

    "bed_length_m": (0.1, 3.0),
    "flow_rate_mps": (0.01, 2.0),
    "porosity": (0.10, 0.70),
    "surface_area_m2": (1.0, 100.0),
    "temperature_c": (200.0, 600.0),
    "additive_frac": (0.0, 1.0),
}

run_tab(
    label="Purification",
    key_prefix="pur",
    ranges=pur_ranges,
    compute_fn=purification.compute,
    noise_controls_fn=pur_noise_controls,
    sampling_dim=sampling_dim,
    rng=rng,
    default_prefix="purification",
)
