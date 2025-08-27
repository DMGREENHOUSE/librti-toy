import numpy as np
import pandas as pd

INPUT_COLUMNS = [
    "blanket_thickness_m",
    "li6_enrichment_pct",
    "multiplier_fraction",
    "structural_fraction",
    "coolant_temp_c",
]
OUTPUT_COLUMNS = ["TBR", "activation_index_1y"]

def compute(inputs: pd.DataFrame, noise_tbr: float, noise_act: float, rng: np.random.Generator) -> pd.DataFrame:
    """
    Compute synthetic outputs: TBR and activation index (@1 year, arb.).
    Expected input columns: blanket_thickness_m, li6_enrichment_pct, multiplier_fraction, structural_fraction, coolant_temp_c
    """
    t = np.clip(inputs["blanket_thickness_m"].to_numpy(), 0.05, None)
    li6 = np.clip(inputs["li6_enrichment_pct"].to_numpy(), 0.0, 95.0)
    mult = np.clip(inputs["multiplier_fraction"].to_numpy(), 0.0, 0.8)
    struct = np.clip(inputs["structural_fraction"].to_numpy(), 0.05, 0.9)
    Tc = inputs["coolant_temp_c"].to_numpy()

    # TBR
    k = 1.8
    tbr = (
        0.80 + 0.60 * (1.0 - np.exp(-k * t))
        + 0.0020 * (li6 - 7.0)
        + 0.30 * mult
        - 0.40 * (struct - 0.2)
        - 0.0002 * (Tc - 300.0)
    )
    tbr += 0.0010 * (li6 - 30.0) * (mult - 0.2)

    # Activation (arb.)
    act = (
        2.0 + 5.0 * (struct - 0.2)
        + 2.0 * mult
        + 0.003 * (Tc - 300.0)
        + 0.5 * np.exp(-2.0 * t) * (1.0 + 0.01 * (100.0 - li6))
    )

    if noise_tbr > 0:
        tbr = tbr + rng.normal(0.0, noise_tbr, size=tbr.shape)
    if noise_act > 0:
        act = act + rng.normal(0.0, noise_act, size=act.shape)

    tbr = np.clip(tbr, 0.50, 2.00)
    act = np.clip(act, 0.0, 10.0)
    return pd.DataFrame({"TBR": tbr, "activation_index_1y": act})
