from __future__ import annotations
import streamlit as st
import numpy as np
from librti_datagen.models import purification
from librti_datagen.ui.shared_ui import run_tab

st.set_page_config(page_title="Purification Data Generator", layout="wide")
st.title("Purification Data Generator")

seed = st.number_input("Random seed", min_value=0, max_value=2**31-1, value=42, step=1)
rng = np.random.default_rng(seed)
sampling_dim = st.radio("Auto sampling dimensionality", ["1D", "Multi-D"], horizontal=True)

st.header("Purification Model")
st.caption("Inputs → purification_efficiency (0–1).")

def pur_noise_controls():
    val = st.slider("Output noise (absolute std)", 0.0, 0.2, 0.02, 0.01, key="pur_noise")
    return {"noise_std": val}

pur_ranges = {
    "temperature_c": (20.0, 500.0),
    "flow_rate_mps": (0.01, 2.0),
    "impurity_ppm": (0.0, 5000.0),
    "additive_frac": (0.0, 1.0),
    "bed_length_m": (0.1, 3.0),
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
