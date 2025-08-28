from __future__ import annotations
import streamlit as st
import numpy as np
from librti_datagen.models import corrosion
from librti_datagen.ui.shared_ui import run_tab

st.set_page_config(page_title="Corrosion Data Generator", layout="wide")
st.title("Corrosion → Dissolved Composition")

seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42, step=1)
rng = np.random.default_rng(seed)
sampling_dim = st.radio("Auto sampling dimensionality", ["1D", "Multi-D"], horizontal=True)

st.header("Corrosion Model")
st.caption(
    "Inputs → **temperature_c** (+ geometry: flow_rate_mps, wall_thickness_m, surface_area_m2). "
    "Outputs → **comp_Fe, comp_Cr, comp_Ni, comp_Mn** (fractions sum to 1)."
)

def cor_noise_controls():
    val = st.slider("Composition noise (Dirichlet-like)", 0.0, 0.5, 0.10, 0.01, key="cor_noise")
    return {"noise_pct": val}

cor_ranges = {
    "temperature_c": (20.0, 600.0),
    "flow_rate_mps": (0.0, 2.0),
    "wall_thickness_m": (0.002, 0.050),
    "surface_area_m2": (1.0, 100.0),
}

run_tab(
    label="Corrosion",
    key_prefix="cor",
    ranges=cor_ranges,
    compute_fn=corrosion.compute,
    noise_controls_fn=cor_noise_controls,
    sampling_dim=sampling_dim,
    rng=rng,
    default_prefix="corrosion",
)
