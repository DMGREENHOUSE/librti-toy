st.caption(
    "Inputs → composition (comp_*), current radionuclide inventory (inv_*), and parameters "
    "(blanket_thickness_m, neutron_flux_e23 in ×1e23 n·m⁻²·s⁻¹, exposure_time_s). "
    "Output → updated inv_* after activation."
)

neu_ranges = {
    "comp_Fe": (0.0, 1.0),
    "comp_Cr": (0.0, 1.0),
    "comp_Ni": (0.0, 1.0),
    "comp_Mn": (0.0, 1.0),

    "inv_Fe59": (0.0, 1e6),
    "inv_Cr51": (0.0, 1e6),
    "inv_Ni59": (0.0, 1e6),
    "inv_Mn54": (0.0, 1e6),

    "blanket_thickness_m": (0.1, 1.5),
    "neutron_flux_e23": (1e-8, 5e-6),
    "exposure_time_s": (1e2, 1e6),
}
