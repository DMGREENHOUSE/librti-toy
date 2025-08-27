from __future__ import annotations
import streamlit as st
import numpy as np
from librti_datagen.models import neutroact
from librti_datagen.ui.shared_ui import run_tab

st.set_page_config(page_title="Neutronics + Activation Data Generator", layout="wide")
st.title("Neutronics + Activation Data Generator")

seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42, step=1)
rng = np.random.default_rng(seed)
sampling_dim = st.radio("Auto sampling dimensionality", ["1D", "Multi-D"], horizontal=True)

st.header("Neutronics + Activation")
st.caption("Inputs → [TBR, activation_index_1y].")

def neu_noise_controls():
    c1, c2 = st.columns(2)
    with c1:
        tbr = st.slider("TBR noise (abs std)", 0.0, 0.05, 0.01, 0.005, key="neu_tbr_noise")
    with c2:
        act = st.slider("Activation noise (abs std)", 0.0, 0.50, 0.10, 0.05, key="neu_act_noise")
    return {"noise_tbr": tbr, "noise_act": act}

neu_ranges = {
    "blanket_thickness_m": (0.2, 1.2),
    "li6_enrichment_pct": (5.0, 90.0),
    "multiplier_fraction": (0.0, 0.4),
    "structural_fraction": (0.2, 0.6),
    "coolant_temp_c": (200.0, 600.0),
}

run_tab(
    label="Neutronics + Activation",
    key_prefix="neu",
    ranges=neu_ranges,
    compute_fn=neutroact.compute,
    noise_controls_fn=neu_noise_controls,
    sampling_dim=sampling_dim,
    rng=rng,
    default_prefix="neutroact",
)
