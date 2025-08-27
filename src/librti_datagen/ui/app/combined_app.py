from __future__ import annotations
import streamlit as st
import numpy as np
from librti_datagen.models import purification, corrosion, neutroact
from librti_datagen.ui.shared_ui import run_tab

st.set_page_config(page_title="LIBRTI Data Generators", layout="wide")

st.title("LIBRTI Synthetic Data Generators")
st.write(
    "Use these generators to create training datasets for Gaussian Process demos. "
    "Each module supports Auto sampling (1D or Multi-D), Custom points, or uploaded inputs; "
    "previews inputs/outputs; and downloads a single combined CSV. "
    "After downloading, the preview collapses automatically — data persists so you can reopen it."
)

seed = st.number_input("Random seed", min_value=0, max_value=2**31-1, value=42, step=1)
rng = np.random.default_rng(seed)

sampling_dim = st.radio(
    "Auto sampling dimensionality (used only when 'Auto sampling' is selected per tab)",
    ["1D", "Multi-D"], horizontal=True
)

pur_tab, cor_tab, neu_tab = st.tabs(["Purification", "Corrosion", "Neutronics + Activation"])

with pur_tab:
    st.header("Purification Model")
    st.caption("Inputs → purification_efficiency (0–1). Toy surrogate with temperature/contact-time effects and impurity load.")
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

with cor_tab:
    st.header("Corrosion Model")
    st.caption("Inputs → corrosion_rate_mm_per_year. Toy Arrhenius-style surrogate with pH/flow/oxygen/chloride effects.")
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

with neu_tab:
    st.header("Neutronics + Activation (Outer Emulator)")
    st.caption("Inputs → [TBR, activation_index_1y]. Toy coupled surrogate capturing broad trends without exposing spectra/cross-sections.")
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

with st.expander("Notes & Tips"):
    st.markdown(
        """
        - **Synthetic only**: These formulas are illustrative for ML training. Adjust coefficients to better match your storytelling.
        - **Input modes**:
          - *Auto sampling*: Use 1D sweeps or uniform Multi-D sampling over ranges.
          - *Custom points*: Manually define exact points via the in-browser table.
          - *Upload inputs*: Provide your own input CSV to evaluate the model.
        - **Append mode**: Upload a base dataset to append generated rows; columns must match (order-insensitive).
        - **Persistence**: Generated datasets are kept in `st.session_state`. After download, the preview collapses; click *Reopen Preview* to expand.
        """
    )
