"""Material library for topology optimization presets.

Users can add custom materials by editing this file.
Units (N-mm-s consistent):
- young_modulus_mpa: MPa (N/mm^2)
- poisson_ratio: dimensionless
- density_kg_mm3: kg/mm^3
- yield_strength_mpa: MPa (N/mm^2)
- thermal_expansion_1_K: 1/K
- fatigue_limit_mpa: MPa (N/mm^2) (fully reversed, approximate)
- fracture_toughness_mpa_sqrt_mm: MPa*sqrt(mm)
"""

MATERIAL_LIBRARY = {
    "steel_structural": {
        "display_name": "Steel (Structural)",
        "young_modulus_mpa": 210000.0,
        "poisson_ratio": 0.30,
        "density_kg_mm3": 7.85e-6,
        "yield_strength_mpa": 250.0,
        "thermal_expansion_1_K": 12.0e-6,
        "fatigue_limit_mpa": 150.0,
        "fracture_toughness_mpa_sqrt_mm": 2213.6,
    },
    "aluminum_6061_t6": {
        "display_name": "Aluminum 6061-T6",
        "young_modulus_mpa": 68900.0,
        "poisson_ratio": 0.33,
        "density_kg_mm3": 2.70e-6,
        "yield_strength_mpa": 276.0,
        "thermal_expansion_1_K": 23.6e-6,
        "fatigue_limit_mpa": 96.0,
        "fracture_toughness_mpa_sqrt_mm": 917.1,
    },
    "pla_3d_print": {
        "display_name": "PLA (3D Printing)",
        "young_modulus_mpa": 3500.0,
        "poisson_ratio": 0.36,
        "density_kg_mm3": 1.24e-6,
        "yield_strength_mpa": 60.0,
        "thermal_expansion_1_K": 68.0e-6,
        "fatigue_limit_mpa": 20.0,
        "fracture_toughness_mpa_sqrt_mm": 110.7,
    },
    "abs_3d_print": {
        "display_name": "ABS (3D Printing)",
        "young_modulus_mpa": 2100.0,
        "poisson_ratio": 0.35,
        "density_kg_mm3": 1.04e-6,
        "yield_strength_mpa": 40.0,
        "thermal_expansion_1_K": 80.0e-6,
        "fatigue_limit_mpa": 14.0,
        "fracture_toughness_mpa_sqrt_mm": 88.5,
    },
    "titanium_ti6al4v": {
        "display_name": "Titanium Ti-6Al-4V",
        "young_modulus_mpa": 114000.0,
        "poisson_ratio": 0.34,
        "density_kg_mm3": 4.43e-6,
        "yield_strength_mpa": 830.0,
        "thermal_expansion_1_K": 8.6e-6,
        "fatigue_limit_mpa": 510.0,
        "fracture_toughness_mpa_sqrt_mm": 1739.3,
    },
}


def list_material_keys():
    return list(MATERIAL_LIBRARY.keys())


def get_material(material_key):
    return MATERIAL_LIBRARY.get(material_key)
