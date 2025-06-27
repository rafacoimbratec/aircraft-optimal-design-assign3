#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aircraft Optimal Design – Final Project
Instituto Superior Técnico – Aerospace Engineering
Academic Year: 2024/2025
Course: Aircraft Optimal Design (AOD)

Project: Conceptual Aerostructural Optimization of the DA40NG 
Objective: To perform a conceptual design study using OpenAeroStruct, including cruise and coordinated turn conditions, with emphasis on structural sizing and aerodynamic efficiency.

Team:
- Rafael Azeiteiro (ist1102478)
- João Pedro Gonçalves da Cruz Coelho de Almeida (ist1103026)
- Eduardo de Almeida Helena (ist1102793)
"""
from openaerostruct.meshing.mesh_generator import generate_mesh
import numpy as np
from openaerostruct.geometry.geometry_mesh import GeometryMesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.utils.plot_wing import Display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessário para 3D


# Diamond DA40NG
# -*- coding: utf-8 -*-
"""
Wing and tail configuration for the DA40NG with Wortmann FX 63-137 airfoil.
"""

import numpy as np
from openaerostruct.geometry.utils import generate_mesh

# ========================================================
# Airfoil section for the wingbox structural model
# 
# These coordinates correspond to the Wortmann FX 63-137 airfoil.
# They are scaled so the chord = 1.
#
# IMPORTANT:
# Only the 10% to 60% chord segment is used because:
# - This region approximately covers the structural wingbox
# - Leading edge and trailing edge are excluded to avoid numerical issues in thickness
# - This subset ensures compatibility with OpenAeroStruct's wingbox definition
#
# The arrays define the x and y coordinates of the airfoil
# upper and lower surfaces within this region.
#
# The dtype="complex128" allows complex-step derivative approximation.
# ========================================================
upper_x = np.array([
    0.103320, 0.124080, 0.146450, 0.170330, 0.195620, 0.222210,
    0.250000, 0.278860, 0.308660, 0.339280, 0.370590, 0.402450,
    0.434740, 0.467300, 0.500000, 0.532700, 0.565260, 0.597550
], dtype="complex128")

upper_y = np.array([
    0.083130, 0.089610, 0.096220, 0.101650, 0.107040, 0.111220,
    0.115220, 0.117920, 0.120240, 0.121280, 0.121910, 0.121370,
    0.120420, 0.118330, 0.115780, 0.112210, 0.108230, 0.103310
], dtype="complex128")

# Lower surface (10% to 60%)
lower_x = np.array([
    0.103320, 0.124080, 0.146450, 0.170330, 0.195620, 0.222210,
    0.250000, 0.278860, 0.308660, 0.339280, 0.370590, 0.402450,
    0.434740, 0.467300, 0.500000, 0.532700, 0.565260, 0.597550
], dtype="complex128")

lower_y = np.array([
   -0.021800, -0.022560, -0.022630, -0.022770, -0.022200, -0.021610,
   -0.020340, -0.018950, -0.016880, -0.014600, -0.011670, -0.008480,
   -0.004860, -0.001030,  0.003070,  0.007160,  0.011120,  0.014750
], dtype="complex128")

# ==========================================
# WING
# ==========================================

# Create mesh dictionary for the wing
mesh_dict_wing = {
    "num_y": 11,                # spanwise panels
    "num_x": 3,                 # chordwise panels
    "wing_type": "rect",        # rectangular planform for simplicity
    "symmetry": True,           # model half-wing
    "chord_cos_spacing": 0,     # uniform chordwise spacing
    "span_cos_spacing": 1,      # cosine spacing along span
    "root_chord": 1.71          # estimated root chord (m)
}

# Generate the wing mesh
mesh_wing = generate_mesh(mesh_dict_wing)

# Create surface dictionary for the wing
surf_dict_wing = {
    "name": "wing",
    "symmetry": True,
    "S_ref_type": "wetted",
    "mesh": mesh_wing,
    "fem_model_type": "wingbox",

    # Airfoil data
    "data_x_upper": upper_x,
    "data_y_upper": upper_y,
    "data_x_lower": lower_x,
    "data_y_lower": lower_y,

    # Geometric design variables
    "twist_cp": np.zeros(3),
    "chord_cp": np.full(3, 1.71),
    "taper": 0.5,
    "sweep": 0.0,
    "span": 11.63,
    "t_over_c_cp": np.full(3, 0.137),
    "original_wingbox_airfoil_t_over_c": 0.137,

    # Structural design variables
    "spar_thickness_cp": np.array([0.0035, 0.0035, 0.0035]),
    "skin_thickness_cp": np.array([0.0035, 0.0035, 0.0035]),

    # Aerodynamic properties
    "CL0": 0.9255,
    "CD0": 0.00737,
    "with_viscous": True, # Enable viscous drag calculations
    "with_wave": True, # Enable wave drag for flight
    "k_lam": 0.05,
    "c_max_t": 0.309,

    # Material properties
    "E": 39e9,
    "G": 3.8e9,
    "yield": (2e9 / 1.5),  # Yield strength divided by safety factor
    "mrho": 2100.0,
    "strength_factor_for_upper_skin": 1.0,
    "wing_weight_ratio": 1.0,
    "exact_failure_constraint": True,
    "struct_weight_relief": True,
    "distributed_fuel_weight": True,
    "fuel_density": 800.0,
    "Wf_reserve": 15000.0,
}

# ==========================================
# TAIL
# ==========================================

# Create mesh dictionary for the tail
mesh_dict_tail = {
    "num_y": 7,
    "num_x": 2,
    "wing_type": "rect",
    "symmetry": True,
    "chord_cos_spacing": 0,
    "span_cos_spacing": 1,
    "root_chord": 1.2,
    "offset": np.array([7.0, 0.0, 0.3])
}

# Generate the tail mesh
mesh_tail = generate_mesh(mesh_dict_tail)

# Create surface dictionary for the tail
surf_dict_tail = {
    "name": "tail",
    "symmetry": True,
    "S_ref_type": "projected",
    "mesh": mesh_tail,
    "fem_model_type": "wingbox",

    # Airfoil data (same placeholder, can be changed)
    "data_x_upper": upper_x,
    "data_y_upper": upper_y,
    "data_x_lower": lower_x,
    "data_y_lower": lower_y,

    # Geometric design variables
    "twist_cp": np.zeros(2),
    "chord_cp": np.full(2, 1.2),
    "taper": 0.5,
    "sweep": 0.0,
    "span": 3.6,
    "t_over_c_cp": np.full(2, 0.12),
    "original_wingbox_airfoil_t_over_c": 0.12,

    # Structural design variables
    "spar_thickness_cp": np.array([0.003, 0.003]),
    "skin_thickness_cp": np.array([0.003, 0.003]),

    # Aerodynamic properties
    "CL0": 0.0,
    "CD0": 0.01,
    "with_viscous": True,
    "with_wave": False,
    "k_lam": 0.05,
    "c_max_t": 0.3,

    # Material properties
    "E": 39e9,
    "G": 3.8e9,
    "yield": (58.5e6 / 1.5),
    "mrho": 2100.0,
    "strength_factor_for_upper_skin": 1.0,
    "wing_weight_ratio": 1.0,
    "exact_failure_constraint": True,
    "struct_weight_relief": True,
    "distributed_fuel_weight": False,
    "n_point_masses": 1,
    "fuel_density": 800.0,
    "Wf_reserve": 0.0,
}

# ==========================================
# List of surfaces for OpenAeroStruct
# ==========================================
surfaces = [surf_dict_wing, surf_dict_tail]

