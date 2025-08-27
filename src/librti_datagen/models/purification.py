import numpy as np
import pandas as pd

INPUT_COLUMNS = [
    "temperature_c",
    "flow_rate_mps",
    "impurity_ppm",
    "additive_frac",
    "bed_length_m",
]
OUTPUT_COLUMNS = ["purification_efficiency"]

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def compute(inputs: pd.DataFrame, noise_std: float, rng: np.random.Generator) -> pd.DataFrame:
    """
    Compute purification efficiency in [0, 1).
    Expected input columns: temperature_c, flow_rate_mps, impurity_ppm, additive_frac, bed_length_m
    """
    T_K = inputs["temperature_c"].to_numpy() + 273.15
    flow = inputs["flow_rate_mps"].to_numpy()
    impurity = np.clip(inputs["impurity_ppm"].to_numpy(), 0, None)
    addit = np.clip(inputs["additive_frac"].to_numpy(), 0, 1)
    bed_L = np.clip(inputs["bed_length_m"].to_numpy(), 1e-3, None)

    log_imp = np.log10(1.0 + impurity)
    contact_time = bed_L / (0.1 + flow)  # pseudo-contact time

    z = (
        -0.8 * log_imp
        + 0.015 * (T_K - 300.0)
        + 0.6 * np.log1p(contact_time)
        + 1.2 * addit
        - 0.5 * flow
        - 0.12 * addit * log_imp
        + 0.0005 * (T_K - 300.0) * (addit - 0.5)
    )
    eta = _sigmoid(z)

    if noise_std > 0:
        eta = eta + rng.normal(0.0, noise_std, size=eta.shape)
    eta = np.clip(eta, 0.0, 0.999)
    return pd.DataFrame({"purification_efficiency": eta})
