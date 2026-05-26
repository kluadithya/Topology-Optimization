"""Material library for topology optimization presets.

Users can add custom materials by editing this file.
Units:
- young_modulus_pa: Pa
- poisson_ratio: dimensionless
- density_kg_m3: kg/m^3
- yield_strength_pa: Pa
- thermal_expansion_1_K: 1/K
- fatigue_limit_pa: Pa (fully reversed, approximate)
- fracture_toughness_mpa_sqrt_m: MPa*sqrt(m)
"""

MATERIAL_LIBRARY = {
    "steel_structural": {
        "display_name": "Steel (Structural)",
        "young_modulus_pa": 210e9,
        "poisson_ratio": 0.30,
        "density_kg_m3": 7850.0,
        "yield_strength_pa": 250e6,
        "thermal_expansion_1_K": 12.0e-6,
        "fatigue_limit_pa": 150e6,
        "fracture_toughness_mpa_sqrt_m": 70.0,
    },
    "aluminum_6061_t6": {
        "display_name": "Aluminum 6061-T6",
        "young_modulus_pa": 68.9e9,
        "poisson_ratio": 0.33,
        "density_kg_m3": 2700.0,
        "yield_strength_pa": 276e6,
        "thermal_expansion_1_K": 23.6e-6,
        "fatigue_limit_pa": 96e6,
        "fracture_toughness_mpa_sqrt_m": 29.0,
    },
    "pla_3d_print": {
        "display_name": "PLA (3D Printing)",
        "young_modulus_pa": 3.5e9,
        "poisson_ratio": 0.36,
        "density_kg_m3": 1240.0,
        "yield_strength_pa": 60e6,
        "thermal_expansion_1_K": 68.0e-6,
        "fatigue_limit_pa": 20e6,
        "fracture_toughness_mpa_sqrt_m": 3.5,
    },
    "abs_3d_print": {
        "display_name": "ABS (3D Printing)",
        "young_modulus_pa": 2.1e9,
        "poisson_ratio": 0.35,
        "density_kg_m3": 1040.0,
        "yield_strength_pa": 40e6,
        "thermal_expansion_1_K": 80.0e-6,
        "fatigue_limit_pa": 14e6,
        "fracture_toughness_mpa_sqrt_m": 2.8,
    },
    "titanium_ti6al4v": {
        "display_name": "Titanium Ti-6Al-4V",
        "young_modulus_pa": 114e9,
        "poisson_ratio": 0.34,
        "density_kg_m3": 4430.0,
        "yield_strength_pa": 830e6,
        "thermal_expansion_1_K": 8.6e-6,
        "fatigue_limit_pa": 510e6,
        "fracture_toughness_mpa_sqrt_m": 55.0,
    },
}


def list_material_keys():
    return list(MATERIAL_LIBRARY.keys())


def get_material(material_key):
    return MATERIAL_LIBRARY.get(material_key)
