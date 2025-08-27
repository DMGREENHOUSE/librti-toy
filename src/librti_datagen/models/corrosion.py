import numpy as np
import pandas as pd

INPUT_COLUMNS = [
    "temperature_c",
    "pH",
    "flow_rate_mps",
    "dissolved_oxygen_ppm",
    "chloride_ppm",
]
OUTPUT_COLUMNS = ["corrosion_rate_mm_per_year"]

def compute(inputs: pd.DataFrame, noise_pct: float, rng: np.random.Generator) -> pd.DataFrame:
    """
    Compute synthetic corrosion rate [mm/year].
    Expected input columns: temperature_c, pH, flow_rate_mps, dissolved_oxygen_ppm, chloride_ppm
    """
    R = 8.314   # J/(mol*K)
    A = 2.0e5   # pre-exponential (synthetic)
    Ea = 5.0e4  # J/mol

    T_K = inputs["temperature_c"].to_numpy() + 273.15
    pH = np.clip(inputs["pH"].to_numpy(), 0.0, 14.0)
    flow = np.clip(inputs["flow_rate_mps"].to_numpy(), 0.0, None)
    DO = np.clip(inputs["dissolved_oxygen_ppm"].to_numpy(), 0.0, None)
    Cl = np.clip(inputs["chloride_ppm"].to_numpy(), 0.0, None)

    arr = A * np.exp(-Ea / (R * T_K))  # temperature effect
    f_pH = 1.0 + 2.0 * np.exp(-((pH - 3.0) ** 2) / 4.0)   # acidic boost
    f_DO = 1.0 + 0.05 * DO
    f_flow = 1.0 + 0.8 * flow
    f_cl = 1.0 + 0.001 * np.log1p(Cl)

    rate = 0.05 * arr * f_pH * f_DO * f_flow * f_cl

    if noise_pct > 0:
        sigma = np.maximum(1e-3, noise_pct * rate)
        rate = rate + rng.normal(0.0, sigma, size=rate.shape)
    rate = np.clip(rate, 0.0, 20.0)
    return pd.DataFrame({"corrosion_rate_mm_per_year": rate})
