from __future__ import annotations
import streamlit as st
import numpy as np
from librti_datagen.models import corrosion, neutroact, purification
from librti_datagen.ui.shared_ui import run_tab

st.set_page_config(page_title="LIBRTI Data Generators", layout="wide")

st.title("LIBRTI Synthetic Data Generators")
st.write(
    "Use these generators to create training datasets which can be subsequently downloaded."
)
col1, col2 = st.columns([1,2])  # two equal halves
with col1:
    st.image("assets/Model.png", use_container_width=True)
with col2:
    st.empty()  # leave blank or add text



seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42, step=1)
rng = np.random.default_rng(seed)

sampling_dim = st.radio(
    "Auto sampling dimensionality (used only when 'Auto sampling' is selected per tab)",
    ["1D", "Multi-D"],
    horizontal=True,
)

cor_tab, neu_tab, pur_tab = st.tabs(["Corrosion", "Neutronics + Activation", "Purification"])

# ---------------------------
# Corrosion: temperature (+ geometry) -> composition fractions
# ---------------------------
with cor_tab:
    st.header("Corrosion")
    col1, col2 = st.columns([1,2])  # two equal halves
    with col1:
        st.image("assets/Corrosion.png", use_container_width=True)
    with col2:
        st.empty()  # leave blank or add text

    def cor_noise_controls():
        val = st.slider("Composition noise (Dirichlet-like)", 0.0, 0.5, 0.10, 0.01, key="cor_noise")
        return {"noise_pct": val}

    cor_ranges = {
        "temperature_c": (20.0, 600.0),
        "flow_rate_mps": (0.0, 2.0),
        "wall_thickness_m": (0.002, 0.050),  # 2–50 mm
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

# ---------------------------
# Neutronics + Activation: comp_* + inv_* + geometry + irradiation -> updated inv_*
# ---------------------------
with neu_tab:
    st.header("Neutronics + Activation")
    col1, col2 = st.columns([1,2])  # two equal halves
    with col1:
        st.image("assets/Neutroact.png", use_container_width=True)
    with col2:
        st.empty()  # leave blank or add text

    def neu_noise_controls():
        act = st.slider("Activation noise (fractional std)", 0.0, 0.5, 0.10, 0.01, key="neu_act_noise")
        return {"noise_tbr": 0.0, "noise_act": act}

    neu_ranges = {
        # Composition
        "comp_Fe": (0.0, 1.0),
        "comp_Cr": (0.0, 1.0),
        "comp_Ni": (0.0, 1.0),
        "comp_Mn": (0.0, 1.0),

        # Current inventory
        "inv_Fe59": (0.0, 1e6),
        "inv_Cr51": (0.0, 1e6),
        "inv_Ni59": (0.0, 1e6),
        "inv_Mn54": (0.0, 1e6),

        # Geometry + irradiation (flux in 1e23 units)
        "blanket_thickness_m": (0.1, 1.5),
        "neutron_flux_e23": (1e-8, 5e-6),   # corresponds to ~1e15 .. 5e17 n·m⁻²·s⁻¹
        "exposure_time_s": (1e2, 1e6),
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


# ---------------------------
# Purification: inv_* (+ purifier geometry) -> updated inv_*
# ---------------------------
with pur_tab:
    st.header("Purification")
    col1, col2 = st.columns([1,2])  # two equal halves
    with col1:
        st.image("assets/Purification.png", use_container_width=True)
    with col2:
        st.empty()  # leave blank or add text

    def pur_noise_controls():
        val = st.slider("Relative noise on updated inventory (std fraction)", 0.0, 0.5, 0.02, 0.01, key="pur_noise")
        return {"noise_std": val}

    pur_ranges = {
        # Example inventory columns (extend or edit freely)
        "inv_Fe59": (0.0, 1e6),
        "inv_Cr51": (0.0, 1e6),
        "inv_Ni59": (0.0, 1e6),
        "inv_Mn54": (0.0, 1e6),
        # Optional: keep tritium in the vector if you want to show incidental removal
        "inv_T": (0.0, 1e6),

        # Purifier geometry (optional in model; defaults used if omitted)
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

with st.expander("Parameter Details"):
    NOTES_MD = """

### Corrosion → Dissolved Composition

> Produces **fractions** that sum to 1.0; use them as composition inputs for Neutronics+Activation.

**Inputs**

| Quantity            | Units   | Description |
|---|---|---|
| `temperature_c`     | °C      | Bulk fluid / wall temperature . |
| `flow_rate_mps`     | m·s⁻¹   | Superficial velocity; tilts composition via mass transfer. |
| `wall_thickness_m`  | m       | Sets area/thickness scaling. |
| `surface_area_m2`   | m²      | Effective exposed area. |

**Outputs**

| Quantity                                 | Units | Description |
|---|---|---|
| `comp_Fe`, `comp_Cr`, `comp_Ni`, `comp_Mn` | – (0–1) | Composition fractions of dissolved corrosion products (row sums ≈ 1). |

---

### Neutronics + Activation → Inventory Update

> Flux is entered as **`neutron_flux_e23` in ×10^23 n·m⁻²·s⁻¹** (i.e., Φ/1e23). The model converts back internally.  
> Adds an activation increment over the exposure window to the provided inventories.

**Inputs**

| Quantity                                  | Units                         | Description |
|---|---|---|
| `comp_Fe`, `comp_Cr`, `comp_Ni`, `comp_Mn` | – (0–1)                      | Composition fractions (often from Corrosion output). |
| `inv_Fe59`, `inv_Cr51`, `inv_Ni59`, `inv_Mn54`, ... | user-chosen          | Current radionuclide inventories (any `inv_*` columns present are passed through and updated). |
| `blanket_thickness_m`                     | m                             | Effective path length (attenuation factor). |
| `neutron_flux_e23`                        | ×10^23 n·m⁻²·s⁻¹              | Incident flux scaled by 1e23 (enter Φ/1e23). |
| `exposure_time_s`                         | s                             | Irradiation time window. |

**Outputs**

| Quantity    | Units           | Description |
|---|---|---|
| `inv_*`     | same as input   | Updated radionuclide inventories after activation increment. |

---
### Purification → Inventory Update

> The model is unit-agnostic for radionuclide inventories: whatever unit you use for `inv_*` is preserved in the output.  
> **Recommended units:** mol·m⁻³ (concentration), Bq·m⁻³ (activity), or mol (amount).

**Inputs**

| Quantity           | Units (recommended)         | Description |
|---|---|---|
| `inv_*`            | user-chosen                 | Current radionuclide inventories (e.g., `inv_Fe59`, `inv_Cr51`, `inv_Ni59`, `inv_Mn54`, optionally `inv_T`). |
| `bed_length_m`     | m                           | Effective bed length / contact path. |
| `flow_rate_mps`    | m·s⁻¹                       | Superficial velocity (controls contact time). |
| `porosity`         | – (0–1)                     | Bed porosity; performance peaks near ~0.4 in this toy model. |
| `surface_area_m2`  | m²                          | Effective sorbent/reactive surface area. |
| `temperature_c`    | °C                          | Operating temperature (mild acceleration with T). |
| `additive_frac`    | – (0–1)                     | Additive fraction; boosts removal efficiency. |

**Outputs**

| Quantity    | Units           | Description |
|---|---|---|
| `inv_*`     | same as input   | Updated radionuclide inventories after a single pass/time-step removal. |

---
"""
    st.markdown(NOTES_MD)
