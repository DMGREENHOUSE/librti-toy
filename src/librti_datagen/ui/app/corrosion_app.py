from __future__ import annotations
import streamlit as st
import numpy as np
from librti_datagen.models import corrosion
from librti_datagen.ui.shared_ui import run_tab

st.set_page_config(page_title="Corrosion Data Generator", layout="wide")
st.title("Corrosion Data Generator")

seed = st.number_input("Random seed", min_value=0, max_value=2**31-1, value=42, step=1)
rng = np.random.default_rng(seed)
sampling_dim = st.radio("Auto sampling dimensionality", ["1D", "Multi-D"], horizontal=True)

st.header("Corrosion Model")
st.caption("Inputs → corrosion_rate_mm_per_year.")

def cor_noise_controls():
    val = st.slider("Output noise (fraction of value)", 0.0, 0.5, 0.10, 0.01, key="cor_noise")
    return {"noise_pct": val}

cor_ranges = {
    "temperature_c": (20.0, 600.0),
    "pH": (0.0, 14.0),
    "flow_rate_mps": (0.0, 2.0),
    "dissolved_oxygen_ppm": (0.0, 10.0),
    "chloride_ppm": (0.0, 20000.0),
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
