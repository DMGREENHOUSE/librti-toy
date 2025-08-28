from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Callable

# -----------------------------
# Branding palette
# -----------------------------
KEY_LIME   = "#EBF38B"
INDIGO     = "#16425B"
INDIGO_50  = "#8AA0AD"
KEPPEL     = "#16D5C2"
KEPPEL_50  = "#8AEAE1"
BLACK      = "#000000"
GREY_80    = "#333333"

__all__ = [
    "to_csv_bytes",
    "sample_uniform",
    "compatible_append",
    "one_d_controls",
    "multid_controls",
    "custom_points_editor",
    "upload_inputs",
    "append_controls",
    "preview_and_download",
    "run_tab",
]

# Try Altair for branded scatter; fall back gracefully
try:
    import altair as alt
except Exception:  # pragma: no cover
    alt = None

# ---------- Brand theming ----------
def apply_branding():
    """Inject lightweight CSS for brand look & feel (idempotent)."""
    if st.session_state.get("_branding_injected"):
        return
    BRAND_CSS = f"""
    <style>
      :root {{
        --brand-primary: {INDIGO};
        --brand-primary-50: {INDIGO_50};
        --brand-accent: {KEPPEL};
        --brand-accent-50: {KEPPEL_50};
        --brand-accent-2: {KEY_LIME};
        --brand-black: {BLACK};
        --brand-grey-80: {GREY_80};
      }}
      /* Headings */
      h1, h2, h3 {{ color: var(--brand-black); }}
      /* Primary buttons & download buttons */
      div.stButton > button, div.stDownloadButton > button {{
        background-color: var(--brand-primary);
        color: white;
        border: 0;
        border-radius: 12px;
      }}
      div.stButton > button:hover, div.stDownloadButton > button:hover {{
        background-color: var(--brand-primary-50);
      }}
      /* Expander header with accent gradient */
      div.streamlit-expanderHeader {{
        background: linear-gradient(90deg, var(--brand-accent-50) 0%, var(--brand-accent-2) 100%);
        color: var(--brand-black);
        border-radius: 8px;
      }}
      /* Labels a tad darker for contrast */
      label {{ color: var(--brand-grey-80); }}
      /* Table header tint */
      [data-testid="stTable"] thead th {{
        background-color: var(--brand-accent-50);
        color: var(--brand-black);
      }}
    </style>
    """
    st.markdown(BRAND_CSS, unsafe_allow_html=True)
    st.session_state["_branding_injected"] = True

# ---------- Data utilities ----------
@st.cache_data(show_spinner=False)
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def sample_uniform(n: int, ranges: Dict[str, Tuple[float, float]], rng: np.random.Generator) -> pd.DataFrame:
    data = {k: rng.uniform(a, b, size=n) for k, (a, b) in ranges.items()}
    return pd.DataFrame(data)

def compatible_append(base_df: pd.DataFrame, new_df: pd.DataFrame):
    """Check column-compatibility (order-insensitive). Returns (ok, reordered_new_df_or_None, message_or_None)."""
    base_cols = list(base_df.columns)
    new_cols = list(new_df.columns)
    if set(base_cols) == set(new_cols):
        return True, new_df[base_cols], None
    missing = [c for c in base_cols if c not in new_cols]
    extra = [c for c in new_cols if c not in base_cols]
    return False, None, f"Column mismatch. Missing: {missing}; Extra: {extra}"

# ---------- UI controls ----------
def one_d_controls(label: str, ranges: Dict[str, Tuple[float, float]], defaults: Dict[str, float]):
    st.subheader(f"{label} • 1D Sweep")
    var_to_vary = st.selectbox("Select variable to vary (1D sweep)", list(ranges.keys()), key=f"{label}_1d_var")
    n = st.number_input("Number of points", 1, 10_000, 10, 1, key=f"{label}_1d_n")
    with st.expander("Set ranges and fixed values"):
        new_ranges, fixed_vals = {}, {}
        for k, (a, b) in ranges.items():
            c = defaults.get(k, (a + b) / 2)
            if k == var_to_vary:
                new_a = st.number_input(f"{k} min", value=float(a), key=f"{label}_{k}_a")
                new_b = st.number_input(f"{k} max", value=float(b), key=f"{label}_{k}_b")
                if new_b < new_a:
                    st.warning(f"Adjusted {k} max to be ≥ min")
                    new_b = new_a
                new_ranges[k] = (new_a, new_b)
                fixed_vals[k] = None
            else:
                val = st.number_input(f"{k} (fixed)", value=float(c), key=f"{label}_{k}_fixed")
                new_ranges[k] = (a, b)
                fixed_vals[k] = val
    return var_to_vary, int(n), new_ranges, fixed_vals

def multid_controls(label: str, ranges: Dict[str, Tuple[float, float]]):
    st.subheader(f"{label} • Multi-D Sampling")
    n = st.number_input("Number of samples", 10, 50_000, 200, 10, key=f"{label}_md_n")
    with st.expander("Set parameter ranges"):
        new_ranges = {}
        for k, (a, b) in ranges.items():
            new_a = st.number_input(f"{k} min", value=float(a), key=f"{label}_{k}_a_md")
            new_b = st.number_input(f"{k} max", value=float(b), key=f"{label}_{k}_b_md")
            if new_b < new_a:
                st.warning(f"Adjusted {k} max to be ≥ min")
                new_b = new_a
            new_ranges[k] = (new_a, new_b)
    return int(n), new_ranges

def custom_points_editor(label: str, ranges: Dict[str, Tuple[float, float]], defaults: Dict[str, float]) -> pd.DataFrame:
    st.subheader(f"{label} • Custom Points")
    st.caption("Add/remove rows to choose exact input points where the model will be evaluated.")
    init = {k: [defaults.get(k, (a + b) / 2)] for k, (a, b) in ranges.items()}
    df = pd.DataFrame(init)
    return st.data_editor(df, num_rows="dynamic", use_container_width=True, key=f"{label}_custom_editor")

def upload_inputs(label: str, ranges: Dict[str, Tuple[float, float]]):
    st.subheader(f"{label} • Upload Inputs")
    st.caption(f"Expected columns: {list(ranges.keys())}")
    file = st.file_uploader(f"Upload inputs CSV for {label}", type=["csv"], key=f"{label}_inputs_uploader")
    if file is None:
        return None
    try:
        df = pd.read_csv(file)
        missing = [c for c in ranges.keys() if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return None
        return df
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None

def append_controls(label: str):
    append = st.checkbox("Append new data to an uploaded base dataset", key=f"{label}_append_toggle")
    base_df = None
    if append:
        st.caption("Upload base dataset (must share the same columns as the generated combined dataset).")
        up = st.file_uploader(f"Upload base dataset CSV for {label}", type=["csv"], key=f"{label}_append_uploader")
        if up is not None:
            try:
                base_df = pd.read_csv(up)
            except Exception as e:
                st.error(f"Failed to read base dataset: {e}")
                base_df = None
    return append, base_df

def preview_and_download(key_prefix: str, default_prefix: str):
    X = st.session_state.get(f"{key_prefix}_inputs")
    y = st.session_state.get(f"{key_prefix}_outputs")
    combined = st.session_state.get(f"{key_prefix}_combined")
    appended = st.session_state.get(f"{key_prefix}_appended")

    if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
        collapsed = st.session_state.get(f"{key_prefix}_collapsed", False)
        with st.expander("Preview, Plot & Download", expanded=not collapsed):
            # ---- PREVIEW ----
            st.subheader("Preview: Inputs")
            st.dataframe(X, use_container_width=True)
            st.subheader("Preview: Outputs")
            st.dataframe(y, use_container_width=True)

            # ---- PLOT (branded scatter) ----
            st.subheader("Plot (scatter)")
            colx, coly = st.columns(2)
            with colx:
                x_var = st.selectbox("X axis (input variable)", list(X.columns), key=f"{key_prefix}_plot_x")
            with coly:
                y_var = st.selectbox("Y axis (output variable)", list(y.columns), key=f"{key_prefix}_plot_y")

            plot_df = pd.DataFrame({x_var: X[x_var].to_numpy(), y_var: y[y_var].to_numpy()})

            if alt is not None:
                chart = (
                    alt.Chart(plot_df)
                    .mark_circle(size=64, color=INDIGO)
                    .encode(
                        x=alt.X(x_var, title=x_var),
                        y=alt.Y(y_var, title=y_var),
                        tooltip=[x_var, y_var],
                    )
                    .interactive()
                    .configure_axis(labelColor=BLACK, titleColor=BLACK, gridOpacity=0.15)
                    .configure_view(strokeOpacity=0)
                    .configure_title(color=BLACK)
                    .configure(background='white')
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Altair unavailable — using Streamlit fallback plot.")
                st.scatter_chart(plot_df, x=x_var, y=y_var, use_container_width=True)

            # ---- DOWNLOAD (single combined) ----
            st.markdown("### Download")
            target_df = appended if isinstance(appended, pd.DataFrame) else combined
            default_name = f"{default_prefix}_appended.csv" if isinstance(appended, pd.DataFrame) else f"{default_prefix}_dataset.csv"
            file_name = st.text_input("Dataset CSV filename", value=default_name, key=f"{key_prefix}_dl_name")
            clicked = st.download_button(
                label="Download dataset CSV",
                data=to_csv_bytes(target_df),
                file_name=file_name,
                mime="text/csv",
                key=f"{key_prefix}_dl_btn",
            )
            if clicked:
                st.session_state[f"{key_prefix}_collapsed"] = True

        if collapsed:
            st.info("Preview collapsed after download. Use the button below to reopen.")
            if st.button("Reopen Preview", key=f"{key_prefix}_reopen"):
                st.session_state[f"{key_prefix}_collapsed"] = False

# ---------- Orchestrator ----------
def run_tab(
    *,
    label: str,
    key_prefix: str,
    ranges: Dict[str, Tuple[float, float]],
    compute_fn: Callable,
    noise_controls_fn: Callable[[], dict],
    sampling_dim: str,
    rng: np.random.Generator,
    default_prefix: str,
):
    """Render one model tab end-to-end: input source, (optional) append, compute, preview, download."""
    apply_branding()  # ensure brand CSS is injected

    defaults = {k: (a + b) / 2 for k, (a, b) in ranges.items()}
    noise_kwargs = noise_controls_fn()

    st.divider()
    st.markdown("### Input Source")
    input_mode = st.radio(
        "Choose how to provide inputs",
        ["Auto sampling", "Custom points", "Upload inputs"],
        key=f"{label}_input_mode",
        horizontal=True,
    )

    append, base_df = append_controls(label)

    X = None
    if input_mode == "Auto sampling":
        if sampling_dim == "1D":
            var, n, r1, fixed = one_d_controls(label, ranges, defaults)
            if st.button(f"Generate 1D {label} Data", key=f"{label}_go_1d"):
                X = {}
                for k, (a, b) in r1.items():
                    if k == var:
                        X[k] = np.linspace(a, b, n)
                    else:
                        X[k] = np.full(n, fixed[k])
                X = pd.DataFrame(X)
        else:
            n, rmd = multid_controls(label, ranges)
            if st.button(f"Generate Multi-D {label} Data", key=f"{label}_go_md"):
                X = sample_uniform(n, rmd, rng)

    elif input_mode == "Custom points":
        edited = custom_points_editor(label, ranges, defaults)
        if st.button(f"Generate {label} Data from Custom Points", key=f"{label}_go_custom"):
            X = edited.copy()

    else:  # Upload inputs
        up = upload_inputs(label, ranges)
        if st.button(f"Generate {label} Data from Uploaded Inputs", key=f"{label}_go_upload"):
            if up is None:
                st.error("Please upload a valid inputs CSV with the expected columns.")
            else:
                X = up.copy()

    # Compute and (optionally) append
    if isinstance(X, pd.DataFrame):
        y = compute_fn(X, **noise_kwargs, rng=rng)
        combined = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

        st.session_state[f"{key_prefix}_inputs"] = X
        st.session_state[f"{key_prefix}_outputs"] = y
        st.session_state[f"{key_prefix}_combined"] = combined
        st.session_state.pop(f"{key_prefix}_appended", None)

        if append and isinstance(base_df, pd.DataFrame):
            ok, reordered_new, msg = compatible_append(base_df, combined)
            if ok:
                st.session_state[f"{key_prefix}_appended"] = pd.concat([base_df, reordered_new], ignore_index=True)
                st.success("Appended to base dataset.")
            else:
                st.warning(f"Append skipped: {msg}")
        elif append and base_df is None:
            st.warning("Append requested but no base dataset uploaded.")

        st.success(f"Generated {label} dataset.")

    preview_and_download(key_prefix, default_prefix)
