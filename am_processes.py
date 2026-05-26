"""Additive manufacturing (AM) process modifiers and helpers."""

AM_PROCESSES = {
    "lpbf_metal": {
        "display_name": "LPBF (Metal)",
        "type": "metal",
        "stiffness_scale": 0.90,
        "yield_scale": 0.85,
        "density_scale": 0.98,
        "anisotropy": {"build": 0.90, "transverse": 1.00},
        "notes": "Typical as-built LPBF with modest anisotropy",
    },
    "ebm_metal": {
        "display_name": "EBM (Metal)",
        "type": "metal",
        "stiffness_scale": 0.92,
        "yield_scale": 0.88,
        "density_scale": 0.99,
        "anisotropy": {"build": 0.92, "transverse": 1.00},
        "notes": "EBM with reduced residual stress, moderate anisotropy",
    },
    "ded_metal": {
        "display_name": "DED / WAAM (Metal)",
        "type": "metal",
        "stiffness_scale": 0.88,
        "yield_scale": 0.82,
        "density_scale": 0.97,
        "anisotropy": {"build": 0.85, "transverse": 1.00},
        "notes": "DED/WAAM with higher anisotropy and porosity",
    },
    "binder_jet_metal": {
        "display_name": "Binder Jet + Sinter (Metal)",
        "type": "metal",
        "stiffness_scale": 0.80,
        "yield_scale": 0.75,
        "density_scale": 0.95,
        "anisotropy": {"build": 0.95, "transverse": 1.00},
        "notes": "Sintered density and strength reduction",
    },
    "sls_polymer": {
        "display_name": "SLS (Polymer)",
        "type": "polymer",
        "stiffness_scale": 0.85,
        "yield_scale": 0.80,
        "density_scale": 0.97,
        "anisotropy": {"build": 0.90, "transverse": 1.00},
        "notes": "Powder bed polymer with modest anisotropy",
    },
    "fdm_polymer": {
        "display_name": "FDM / FFF (Polymer)",
        "type": "polymer",
        "stiffness_scale": 0.70,
        "yield_scale": 0.65,
        "density_scale": 0.92,
        "anisotropy": {"build": 0.60, "transverse": 1.00},
        "notes": "Layer bonding dominates; strong build-direction reduction",
    },
    "sla_dlp_polymer": {
        "display_name": "SLA / DLP (Polymer)",
        "type": "polymer",
        "stiffness_scale": 0.90,
        "yield_scale": 0.85,
        "density_scale": 0.99,
        "anisotropy": {"build": 0.95, "transverse": 1.00},
        "notes": "Photopolymer with relatively low porosity",
    },
}


def list_am_process_keys():
    return list(AM_PROCESSES.keys())


def get_am_process(key):
    return AM_PROCESSES.get(key)


def apply_am_process(base_props, process_key, build_axis="z"):
    """Apply AM modifiers to a base material property dict.

    base_props must include: young_modulus, material_density, yield_strength.
    Returns dict with effective properties and anisotropy factors.
    """
    process = get_am_process(process_key)
    if process is None:
        raise ValueError(f"Unknown AM process: {process_key}")

    E0 = float(base_props.get("young_modulus", 1.0))
    density = float(base_props.get("material_density", 1.0))
    ys = float(base_props.get("yield_strength", 1.0))

    stiffness_scale = float(process.get("stiffness_scale", 1.0))
    yield_scale = float(process.get("yield_scale", 1.0))
    density_scale = float(process.get("density_scale", 1.0))

    E0_eff = E0 * stiffness_scale
    density_eff = density * density_scale
    ys_eff = ys * yield_scale

    anisotropy = process.get("anisotropy", None)
    anisotropy_factors = None
    if isinstance(anisotropy, dict) and "build" in anisotropy and "transverse" in anisotropy:
        build_factor = float(anisotropy.get("build", 1.0))
        transverse_factor = float(anisotropy.get("transverse", 1.0))
        axis = build_axis if build_axis in ("x", "y", "z") else "z"
        if axis == "x":
            anisotropy_factors = {"x": build_factor, "y": transverse_factor, "z": transverse_factor}
        elif axis == "y":
            anisotropy_factors = {"x": transverse_factor, "y": build_factor, "z": transverse_factor}
        else:
            anisotropy_factors = {"x": transverse_factor, "y": transverse_factor, "z": build_factor}

    yield_anisotropy = process.get("yield_anisotropy", None)
    if yield_anisotropy is None:
        yield_anisotropy = anisotropy
    yield_anisotropy_factors = None
    if isinstance(yield_anisotropy, dict) and "build" in yield_anisotropy and "transverse" in yield_anisotropy:
        build_factor = float(yield_anisotropy.get("build", 1.0))
        transverse_factor = float(yield_anisotropy.get("transverse", 1.0))
        axis = build_axis if build_axis in ("x", "y", "z") else "z"
        if axis == "x":
            yield_anisotropy_factors = {"x": build_factor, "y": transverse_factor, "z": transverse_factor}
        elif axis == "y":
            yield_anisotropy_factors = {"x": transverse_factor, "y": build_factor, "z": transverse_factor}
        else:
            yield_anisotropy_factors = {"x": transverse_factor, "y": transverse_factor, "z": build_factor}

    return {
        "effective_young_modulus": E0_eff,
        "effective_density": density_eff,
        "effective_yield_strength": ys_eff,
        "anisotropy_factors": anisotropy_factors,
        "yield_anisotropy_factors": yield_anisotropy_factors,
        "process_name": process.get("display_name", process_key),
        "process_key": process_key,
        "build_axis": build_axis if build_axis in ("x", "y", "z") else "z",
        "process_type": process.get("type", "unknown"),
        "stiffness_scale": stiffness_scale,
        "yield_scale": yield_scale,
        "density_scale": density_scale,
        "notes": process.get("notes", ""),
    }
