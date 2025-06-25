#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aircraft Optimal Design – Final Project
Instituto Superior Técnico – Aerospace Engineering
Academic Year: 2024/2025
Course: Aircraft Optimal Design (AOD)

Project: Conceptual Aerostructural Optimization of the MQ-9 Reaper UAV
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


# Diamond DA40NG - Geometria da asa
span = 11.63                  # m
S = 13.5                      # m^2
AR = span**2 / S              # ≈ 10
root_chord = 1.4              # m
taper_ratio = 0.5             # estimado
t_over_c = 0.12               # espessura relativa típica

# Malha da asa
mesh_dict_wing = {
    "num_y": 9,
    "num_x": 3,
    "wing_type": "rect",
    "symmetry": True,
    "chord_cos_spacing": 0,
    "span_cos_spacing": 1,
    "root_chord": root_chord,
}
mesh_wing = generate_mesh(mesh_dict_wing)

surf_dict_wing = {
    "name": "wing",
    "symmetry": True,
    "S_ref_type": "wetted",
    "mesh": mesh_wing,
    "fem_model_type": "wingbox",
    "twist_cp": np.zeros(3),
    "spar_thickness_cp": np.array([0.004, 0.004, 0.004]),
    "skin_thickness_cp": np.array([0.004, 0.004, 0.004]),
    "t_over_c_cp": np.full(3, t_over_c),
    "original_wingbox_airfoil_t_over_c": t_over_c,
    "taper": taper_ratio,
    "chord_cp": np.full(3, root_chord),
    "span": span,
    "sweep": 0.0,
    "CL0": 0.3,
    "CD0": 0.02,
    "with_viscous": True,
    "with_wave": False,
    "k_lam": 0.05,
    "c_max_t": 0.3,
    "E": 70e9,
    "G": 30e9,
    "yield": 500e6 / 1.5,
    "mrho": 2700.0,
    "strength_factor_for_upper_skin": 1.0,
    "wing_weight_ratio": 1.0,
    "exact_failure_constraint": True,
    "struct_weight_relief": True,
    "distributed_fuel_weight": True,
    "fuel_density": 800.0,
    "Wf_reserve": 50.0,
}

# Geometria estimada da cauda
span_tail = 3.6
AR_tail = 6.0
S_tail = span_tail**2 / AR_tail
root_chord_tail = 1.1
taper_tail = 0.5
t_over_c_tail = 0.1

mesh_dict_tail = {
    "num_y": 7,
    "num_x": 2,
    "wing_type": "rect",
    "symmetry": True,
    "chord_cos_spacing": 0,
    "span_cos_spacing": 1,
    "root_chord": root_chord_tail,
    "offset": np.array([7.0, 0.0, 0.3])
}
mesh_tail = generate_mesh(mesh_dict_tail)

surf_dict_tail = {
    "name": "tail",
    "symmetry": True,
    "S_ref_type": "projected",
    "mesh": mesh_tail,
    "fem_model_type": "wingbox",
    "twist_cp": np.zeros(2),
    "spar_thickness_cp": np.array([0.0035, 0.0035]),
    "skin_thickness_cp": np.array([0.0035, 0.0035]),
    "t_over_c_cp": np.full(2, t_over_c_tail),
    "original_wingbox_airfoil_t_over_c": t_over_c_tail,
    "taper": taper_tail,
    "chord_cp": np.full(2, root_chord_tail),
    "span": span_tail,
    "sweep": 0.0,
    "CL0": 0.0,
    "CD0": 0.01,
    "with_viscous": True,
    "with_wave": False,
    "k_lam": 0.05,
    "c_max_t": 0.3,
    "E": 70e9,
    "G": 30e9,
    "yield": 500e6 / 1.5,
    "mrho": 2700.0,
    "strength_factor_for_upper_skin": 1.0,
    "wing_weight_ratio": 1.0,
    "exact_failure_constraint": True,
    "struct_weight_relief": True,
    "distributed_fuel_weight": False,
    "n_point_masses": 1,
    "fuel_density": 800.0,
    "Wf_reserve": 0.0,
}

surfaces = [surf_dict_wing, surf_dict_tail]
