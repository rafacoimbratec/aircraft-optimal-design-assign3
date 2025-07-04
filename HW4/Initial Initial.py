#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openaerostruct.structures.wingbox_fuel_vol_delta import WingboxFuelVolDelta
import openmdao.api as om
import matplotlib.pyplot as plt

# Provide coordinates for a portion of an airfoil for the wingbox cross-section as an nparray with dtype=complex (to work with the complex-step approximation for derivatives).
# These should be for an airfoil with the chord scaled to 1.
# We use the 10% to 60% portion of the NASA SC2-0612 airfoil for this case
# We use the coordinates available from airfoiltools.com. Using such a large number of coordinates is not necessary.
# The first and last x-coordinates of the upper and lower surfaces must be the same

# fmt: off 
upper_x = np.array([0.00000, 0.00107, 0.00428, 0.00961, 0.01704, 0.02653, 0.03806, 0.05156, 0.06699, 0.08427, 0.10332, 0.12408, 0.14645, 0.17033, 0.19562, 0.22221, 0.25000, 0.27886, 0.30866, 0.33928, 0.37059, 0.40245, 0.43474, 0.46730, 0.50000, 0.53270, 0.56526, 0.59755, 0.62941, 0.66072, 0.69134, 0.72114, 0.75000, 0.77779, 0.80438, 0.82967, 0.85355, 0.87592, 0.89668, 0.91573, 0.93301, 0.94844, 0.96194, 0.97347, 0.98296, 0.99039, 0.99572, 0.99893, 1.00000], dtype="complex128")
upper_y = np.array([0.00000, 0.00900, 0.01750, 0.02740, 0.03625, 0.04480, 0.05248, 0.06005, 0.06836, 0.07555, 0.08313, 0.08961, 0.09622, 0.10165, 0.10704, 0.11122, 0.11522, 0.11792, 0.12024, 0.12128, 0.12191, 0.12137, 0.12042, 0.11833, 0.11578, 0.11221, 0.10823, 0.10331, 0.09804, 0.09204, 0.08590, 0.07927, 0.07273, 0.06605, 0.05962, 0.05323, 0.04711, 0.04114, 0.03553, 0.03018, 0.02516, 0.02043, 0.01601, 0.01189, 0.00818, 0.00501, 0.00249, 0.00082, 0.00000], dtype="complex128")
lower_x = np.array([0.00000, 0.00107, 0.00428, 0.00961, 0.01704, 0.02653, 0.03806, 0.05156, 0.06699, 0.08427, 0.10332, 0.12408, 0.14645, 0.17033, 0.19562, 0.22221, 0.25000, 0.27886, 0.30866, 0.33928, 0.37059, 0.40245, 0.43474, 0.46730, 0.50000, 0.53270, 0.56526, 0.59755, 0.62941, 0.66072, 0.69134, 0.72114, 0.75000, 0.77779, 0.80438, 0.82967, 0.85355, 0.87592, 0.89668, 0.91573, 0.93301, 0.94844, 0.96194, 0.97347, 0.98296, 0.99039, 0.99572, 0.99893, 1.00000], dtype="complex128")
lower_y = np.array([0.00000, -0.00232, -0.00566, -0.00995, -0.01254, -0.01537, -0.01698, -0.01887, -0.01992, -0.02122,-0.02180, -0.02256, -0.02263, -0.02277, -0.02220, -0.02161, -0.02034, -0.01895, -0.01688, -0.01460,-0.01167, -0.00848, -0.00486, -0.00103,  0.00307,  0.00716,  0.01112,  0.01475,  0.01813,  0.02098, 0.02345,  0.02530,  0.02668,  0.02745,  0.02768,  0.02729,  0.02631,  0.02479,  0.02284,  0.02052, 0.01794,  0.01514,  0.01219,  0.00921,  0.00630,  0.00373,  0.00169,  0.00040,  0.00000], dtype="complex128")
# fmt: on

# Keep only the section 0.1 to 0.6 chord, 18 points
ix_start, ix_end = 10, 34  # or adjust as needed

upper_x = upper_x[ix_start:ix_end]
upper_y = upper_y[ix_start:ix_end]
lower_x = lower_x[ix_start:ix_end]
lower_y = lower_y[ix_start:ix_end]

# Wing and tail have he same airfoil
upper_x2 = upper_x
lower_x2 = lower_x
upper_y2 = upper_y
lower_y2 = lower_y

# docs checkpoint 2
b= 11.63
AR= 10.223
S= 13.244
c= 1.171 

# Create a dictionary to store options about the surface
mesh_dict = { 
    "num_y": 21,
    "num_x": 5,
    "wing_type": "rect",
    "symmetry": True,
    "chord_cos_spacing": 0,
    "span_cos_spacing": 0,
    "root_chord": 1.54
}

mesh = generate_mesh(mesh_dict)
surf_dict1 = {
    # WING definition
    "name": "wing",  # give the surface some name
    "symmetry": True,  # if True, model only one half of the lifting surface
    "S_ref_type": "wetted",  # how we compute the wing area,
    # can be 'wetted' or 'projected'
    "mesh": mesh,
    "fem_model_type": "wingbox",  # 'wingbox' or 'tube'
    "data_x_upper": upper_x,
    "data_x_lower": lower_x,
    "data_y_upper": upper_y,
    "data_y_lower": lower_y,
    
    "twist_cp": np.array([0, 0, 0, 0]),  # [deg]
    "spar_thickness_cp": np.array([0.0035, 0.0035, 0.0035, 0.0035]),  # [m]
    "skin_thickness_cp": np.array([0.0035, 0.0035, 0.0035, 0.0035]),  # [m]
    "t_over_c_cp": np.array([0.12, 0.12, 0.12, 0.12]),  #local thickness-to-chord ratio control point
    "original_wingbox_airfoil_t_over_c": 0.137,
    "taper": 0.6, 
    "chord_cp":  np.array([1, 1, 1, 1]),
    "span": 11.63, 
    "dihedral": 5.0,
    
    # Aerodynamic deltas.
    # These CL0 and CD0 values are added to the CL and CD
    # obtained from aerodynamic analysis of the surface to get the total CL and CD.
    # These CL0 and CD0 values do not vary wrt alpha. 
    # They can be used to account for things that are not included, such as contributions from the fuselage, camber, etc.
    "CL0": 0.9255,  # CL delta 
    "CD0": 0.007739,  # CD delta 
    "with_viscous": True,  # if true, compute viscous drag
    "with_wave": True,  # if true, compute wave drag
    # Airfoil properties for viscous drag calculation
    "k_lam": 0.05,  # fraction of chord with laminar
    # flow, used for viscous drag
    "c_max_t": 0.309,  # chordwise location of maximum thickness
    "sweep": 1,
   
    # Structural values are based on GFRP
    "E": 39e9,  # [Pa] Young's modulus
    "G": 3.8e9,  # [Pa] shear modulus 
    "yield": (2e9 / 1.25),  # [Pa] allowable yield stress
    "mrho": 2.1e3,  # [kg/m^3] material density
    "strength_factor_for_upper_skin": 1.0,  # the yield stress is multiplied by this factor for the upper skin
    "wing_weight_ratio": 1.1,
    "exact_failure_constraint": True,  # if false, use KS function
   
    "struct_weight_relief": True,
    "distributed_fuel_weight": True,
    #"n_point_masses": 1,  # number of point masses in the system; in this case, the engine (omit option if no point masses)
 
    "fuel_density": 800.0,  # [kg/m^3] fuel density (only needed if the fuel-in-wing volume constraint is used)
    "Wf_reserve": 0,  # [kg] reserve fuel mass
}

mesh_dict2 = {
    "num_y": 21,
    "num_x": 5,
    "wing_type": "rect",
    "symmetry": True,
    "chord_cos_spacing": 0,
    "span_cos_spacing": 0,
    "root_chord": 1,
    "offset": np.array([4.776, 0.000, 1.642])
    
}

mesh2 = generate_mesh(mesh_dict2)
surf_dict2 = {
    # TAIL definition
    "name": "tail",  # give the surface some name
    "symmetry": True,  # if True, model only one half of the lifting surface
    "S_ref_type": "projected",  # how we compute the wing area,
    # can be 'wetted' or 'projected'
    "mesh": mesh2,
    "fem_model_type": "wingbox",  # 'wingbox' or 'tube'
    "data_x_upper": upper_x2,
    "data_x_lower": lower_x2,
    "data_y_upper": upper_y2,
    "data_y_lower": lower_y2,

    "twist_cp": np.array([0]),  # [deg]
    "spar_thickness_cp": np.array([0.0035]),  # [m]
    "skin_thickness_cp": np.array([0.0035]),  # [m]
    "t_over_c_cp": np.array([0.12]), 
    "original_wingbox_airfoil_t_over_c": 0.137,
    "taper": 0.54, 
    "chord_cp":  np.array([1, 1, 1, 1]),
    "span":3.44,
    "sweep":14,
    
    # Aerodynamic deltas.
    # These CL0 and CD0 values are added to the CL and CD
    # obtained from aerodynamic analysis of the surface to get
    # the total CL and CD.
    # These CL0 and CD0 values do not vary wrt alpha.
    # They can be used to account for things that are not included, such as contributions from the fuselage, camber, etc.
    "CL0": 0.0,  # CL delta
    "CD0": 0.006,  # CD delta 
    "with_viscous": True,  # if true, compute viscous drag
    "with_wave": True,  # if true, compute wave drag
    # Airfoil properties for viscous drag calculation
    "k_lam": 0.05,  # fraction of chord with laminar
    # flow, used for viscous drag
    "c_max_t": 0.309,  # chordwise location of maximum thickness
    # docs checkpoint 6
    # Structural values are based on aluminum 7075
    "E": 39e9,  # [Pa] Young's modulus
    "G": 3.8e9,  # [Pa] shear modulus (calculated using E and the Poisson's ratio here)
    "yield": (2e9 / 1.25),  # [Pa] allowable yield stress
    "mrho": 2.1e3,  # [kg/m^3] material density
    "strength_factor_for_upper_skin": 1.0,  # the yield stress is multiplied by this factor for the upper skin
    "wing_weight_ratio": 1.1,
    "exact_failure_constraint": True,  # if false, use KS function

    "struct_weight_relief": True,
    "distributed_fuel_weight": False,  # number of point masses in the system; in this case, the engine (omit option if no point masses)
    #"n_point_masses": 1,  # number of point masses in the system; in this case, the engine (no engine on the tail of our aircraft)
    "fuel_density": 800.0,  # [kg/m^3] fuel density (only needed if the fuel-in-wing volume constraint is used)
    "Wf_reserve": 0.0,  # [kg] reserve fuel mass

}

surfaces = [surf_dict1,surf_dict2]
#surfaces = [surf_dict1]

# Create the problem and assign the model group
prob = om.Problem()

# Add problem information as an independent variables component
indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("Mach_number", val=np.array([0.197, 0.197]))
indep_var_comp.add_output("v", val=np.array([63.89, 63.89]), units="m/s")
indep_var_comp.add_output("re",val=np.array([3e7,3e7]),units="1/m")
indep_var_comp.add_output("rho", val=np.array([0.79786, 0.79786]), units="kg/m**3")
indep_var_comp.add_output("speed_of_sound", val=np.array([324.3247, 324.3247]), units="m/s")

# everything before this point has been updated to use the new airfoil coordinates and mesh generation

indep_var_comp.add_output("CT", val=0.0000569, units="1/s") 
indep_var_comp.add_output("R", val=1730e3, units="m") 
indep_var_comp.add_output("W0", val=1310, units="kg") 

indep_var_comp.add_output("load_factor", val=np.array([1.0, 2]))
indep_var_comp.add_output("alpha", val=0.0, units="deg")
indep_var_comp.add_output("alpha_maneuver", val=4.0, units="deg")

indep_var_comp.add_output("empty_cg", val=np.zeros((3)), units="m")

indep_var_comp.add_output("fuel_mass", val=124, units="kg") 

prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

# Loop over each surface in the surfaces list
for surface in surfaces:
    # Get the surface name and create a group to contain components
    # only for this surface
    name = surface["name"]

    aerostruct_group = AerostructGeometry(surface=surface)

    # Add groups to the problem with the name of the surface.
    prob.model.add_subsystem(name, aerostruct_group)

# Loop through and add a certain number of aerostruct points
for i in range(2):
    point_name = "AS_point_{}".format(i)
    # Connect the parameters within the model for each aero point

    # Create the aerostruct point group and add it to the model
    AS_point = AerostructPoint(surfaces=surfaces, internally_connect_fuelburn=False)

    prob.model.add_subsystem(point_name, AS_point)

    # Connect flow properties to the analysis point
    prob.model.connect("v", point_name + ".v", src_indices=[i])
    prob.model.connect("Mach_number", point_name + ".Mach_number", src_indices=[i])
    prob.model.connect("re", point_name + ".re", src_indices=[i])
    prob.model.connect("rho", point_name + ".rho", src_indices=[i])
    prob.model.connect("CT", point_name + ".CT")
    prob.model.connect("R", point_name + ".R")
    prob.model.connect("W0", point_name + ".W0")
    prob.model.connect("speed_of_sound", point_name + ".speed_of_sound", src_indices=[i])
    prob.model.connect("empty_cg", point_name + ".empty_cg")
    prob.model.connect("load_factor", point_name + ".load_factor", src_indices=[i])
    prob.model.connect("fuel_mass", point_name + ".total_perf.L_equals_W.fuelburn")
    prob.model.connect("fuel_mass", point_name + ".total_perf.CG.fuelburn")

    for surface in surfaces:
        name = surface["name"]

        if name=="wing":
            if surf_dict1["distributed_fuel_weight"]:
                prob.model.connect("load_factor", point_name + ".coupled.load_factor", src_indices=[i])     
        
        com_name = point_name + "." + name + "_perf."
        prob.model.connect(
            name + ".local_stiff_transformed", point_name + ".coupled." + name + ".local_stiff_transformed"
        )
        prob.model.connect(name + ".nodes", point_name + ".coupled." + name + ".nodes")

        # Connect aerodyamic mesh to coupled group mesh
        prob.model.connect(name + ".mesh", point_name + ".coupled." + name + ".mesh")
        
        if name=="wing":
            if surf_dict1["struct_weight_relief"]:
                prob.model.connect(name + ".element_mass", point_name + ".coupled." + name + ".element_mass")
  
        # Connect performance calculation variables
        prob.model.connect(name + ".nodes", com_name + "nodes")
        prob.model.connect(name + ".cg_location", point_name + "." + "total_perf." + name + "_cg_location")
        prob.model.connect(name + ".structural_mass", point_name + "." + "total_perf." + name + "_structural_mass")

        # Connect wingbox properties to von Mises stress calcs
        prob.model.connect(name + ".Qz", com_name + "Qz")
        prob.model.connect(name + ".J", com_name + "J")
        prob.model.connect(name + ".A_enc", com_name + "A_enc")
        prob.model.connect(name + ".htop", com_name + "htop")
        prob.model.connect(name + ".hbottom", com_name + "hbottom")
        prob.model.connect(name + ".hfront", com_name + "hfront")
        prob.model.connect(name + ".hrear", com_name + "hrear")

        prob.model.connect(name + ".spar_thickness", com_name + "spar_thickness")
        prob.model.connect(name + ".t_over_c", com_name + "t_over_c")

prob.model.connect("alpha", "AS_point_0" + ".alpha")
prob.model.connect("alpha_maneuver", "AS_point_1" + ".alpha")

# Here we add the fuel volume constraint componenet to the model
prob.model.add_subsystem("fuel_vol_delta", WingboxFuelVolDelta(surface=surface))
prob.model.connect("wing.struct_setup.fuel_vols", "fuel_vol_delta.fuel_vols")
prob.model.connect("AS_point_0.fuelburn", "fuel_vol_delta.fuelburn")

if surf_dict1["distributed_fuel_weight"]:
    prob.model.connect("wing.struct_setup.fuel_vols", "AS_point_0.coupled.wing.struct_states.fuel_vols")
    prob.model.connect("fuel_mass", "AS_point_0.coupled.wing.struct_states.fuel_mass")

    prob.model.connect("wing.struct_setup.fuel_vols", "AS_point_1.coupled.wing.struct_states.fuel_vols")
    prob.model.connect("fuel_mass", "AS_point_1.coupled.wing.struct_states.fuel_mass")

comp = om.ExecComp("fuel_diff = (fuel_mass - fuelburn) / fuelburn", units="kg")
prob.model.add_subsystem("fuel_diff", comp, promotes_inputs=["fuel_mass"], promotes_outputs=["fuel_diff"])
prob.model.connect("AS_point_0.fuelburn", "fuel_diff.fuelburn")

prob.model.add_objective("AS_point_0.fuelburn", scaler=1e-5)

#prob.model.add_design_var("alpha", lower=-15.0, upper=15.0) 
prob.model.add_design_var("wing.twist_cp", lower=-15.0, upper=15.0, scaler=0.1)   
#prob.model.add_design_var("wing.sweep", lower=0.0, upper=60.0, scaler=0.1)                    #1
#prob.model.add_design_var("wing.geometry.span", upper=30.0, scaler=0.1)            #1
#prob.model.add_design_var("wing.geometry.chord_cp", lower=0.01, upper=6.0)       #1
prob.model.add_design_var("wing.spar_thickness_cp", lower=0.003, upper=0.1, scaler=1e2)  #1
prob.model.add_design_var("wing.skin_thickness_cp", lower=0.003, upper=0.1, scaler=1e2)  #1
#prob.model.add_design_var("wing.geometry.t_over_c_cp", lower=0.07, upper=0.2, scaler=10.0)

prob.model.add_design_var("alpha_maneuver", lower=-10.0, upper=15, scaler=0.1)                #1,2

prob.model.add_design_var("tail.twist_cp", lower=-15.0, upper=15.0, scaler=0.1)  
#prob.model.add_design_var("tail.geometry.span", lower=5, upper=11)
#prob.model.add_design_var("tail.geometry.chord_cp", lower=0.001, upper=2)
#prob.model.add_constraint("AS_point_0.tail_perf.S_ref", equals=14)
#prob.model.add_constraint("AS_point_1.tail_perf.S_ref", equals=14)
#prob.model.add_design_var("tail.spar_thickness_cp", lower=0.003, upper=0.1, scaler=1e2)   #1
#prob.model.add_design_var("tail.skin_thickness_cp", lower=0.003, upper=0.1, scaler=1e2)   #1

#prob.model.add_constraint("wing.twist_cp", indices=[3], lower=0, upper=0)     
prob.model.add_constraint("AS_point_0.wing_perf.Cl", upper=1.8407)            
prob.model.add_constraint("AS_point_1.wing_perf.Cl", upper=1.8407)            #1,2
prob.model.add_constraint("AS_point_0.L_equals_W", equals=0.0)           
prob.model.add_constraint("AS_point_1.L_equals_W", equals=0.0)           #1,2

prob.model.add_constraint("AS_point_0.wing_perf.failure", upper=0.0)     #1,3    tem de estar na segunda otimização
prob.model.add_constraint("AS_point_1.wing_perf.failure", upper=0.0)     #1,2
#prob.model.add_constraint("AS_point_0.tail_perf.failure", upper=0.0)     #1,3    tem de estar na segunda otimização
#prob.model.add_constraint("AS_point_1.tail_perf.failure", upper=0.0)     #1,2
#prob.model.add_constraint("AS_point_0.wing_perf.S_ref", lower=10, upper=14)  #1
#prob.model.add_constraint("AS_point_1.wing_perf.S_ref", lower=10, upper=14)  #1

prob.model.add_constraint("fuel_vol_delta.fuel_vol_delta", lower=0.0)

prob.model.add_design_var("fuel_mass", lower=0.0, upper=2e3, scaler=1e-5)
prob.model.add_constraint("fuel_diff", equals=0.0)

prob.model.add_constraint("AS_point_0.CM", lower=-0.0001, upper=0.0001)
#prob.model.add_constraint("AS_point_1.CM", lower=-0.0001, upper=0.0001)

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"
prob.driver.options["tol"] = 1e-6
prob.driver.options["invalid_desvar_behavior"] = "ignore"
prob.driver.options['maxiter'] = 1000

recorder = om.SqliteRecorder("aerostrf2.db")
prob.driver.add_recorder(recorder)

# We could also just use prob.driver.recording_options['includes']=['*'] here, but for large meshes the database file becomes extremely large. So we just select the variables we need.
prob.driver.recording_options["includes"] = ['*']

prob.driver.recording_options["record_objectives"] = True
prob.driver.recording_options["record_constraints"] = True
prob.driver.recording_options["record_desvars"] = True
prob.driver.recording_options["record_inputs"] = True

# Set up the problem
prob.setup()

# change linear solver for aerostructural coupled adjoint
prob.model.AS_point_0.coupled.linear_solver = om.LinearBlockGS(iprint=0, maxiter=100, use_aitken=True)
prob.model.AS_point_1.coupled.linear_solver = om.LinearBlockGS(iprint=0, maxiter=100, use_aitken=True)

#prob.model.AS_point_0.coupled.nonlinear_solver = om.NonlinearBlockGS(iprint=0, maxiter=500, use_aitken=True)
#prob.model.AS_point_1.coupled.nonlinear_solver = om.NonlinearBlockGS(iprint=0, maxiter=500, use_aitken=True)

# om.view_model(prob)

# prob.check_partials(form='central', compact_print=True)

prob.run_driver()

# === Export optimized wing surface to .pkl ===
optimized_surface = surf_dict1.copy()
optimized_surface["mesh"] = prob.get_val("wing.mesh")
optimized_surface["twist_cp"] = prob.get_val("wing.twist_cp")
optimized_surface["chord_cp"] = prob.get_val("wing.geometry.chord_cp")
optimized_surface["t_over_c_cp"] = prob.get_val("wing.geometry.t_over_c_cp")
optimized_surface["spar_thickness_cp"] = prob.get_val("wing.spar_thickness_cp")
optimized_surface["skin_thickness_cp"] = prob.get_val("wing.skin_thickness_cp")
import pickle

with open("optimized_wing.pkl", "wb") as f:
    pickle.dump(optimized_surface, f)

print("The fuel burn value is", prob["AS_point_0.fuelburn"][0], "[kg]")
print(
    "The wingbox mass wing (excluding the wing_weight_ratio) is",
    prob["wing.structural_mass"][0] / surf_dict1["wing_weight_ratio"],
    "[kg]",
)
print(
    "The wingbox mass tail (excluding the wing_weight_ratio) is",
    prob["tail.structural_mass"][0] / surf_dict2["wing_weight_ratio"],
    "[kg]",
)
print("The total mass is", prob["AS_point_0.total_perf.total_weight"]/9.81, "[kg]")
print("Wing area is",prob["AS_point_0.wing_perf.S_ref"], "[m^2]")
print("Cruise CD is",prob["AS_point_0.wing_perf.CD"])
print("Cruise CL is",prob["AS_point_0.wing_perf.Cl"])
print("Cruise CD tail is",prob["AS_point_0.tail_perf.CD"])
print("Cruise CL tail is",prob["AS_point_0.tail_perf.CL"])
print("Maneuver CL is",prob["AS_point_1.wing_perf.CL"])
print("Wing span is",prob["wing.geometry.span"],"[m]")
print("Wing chord is",prob["wing.geometry.chord_cp"],"[m]")
print("Wing twist is",prob["wing.twist_cp"],"[º]")
print("Tail sweep is",prob["tail.sweep"],"[º]")
print("Tail span is",prob["tail.geometry.span"],"[m]")
print("Tail chord is",prob["tail.geometry.chord_cp"],"[m]")
print("Tail twist is",prob["tail.twist_cp"],"[º]")
print("Alpha cruise is",prob["alpha"],"[º]")
print("Alpha maneuver is",prob["alpha_maneuver"],"[º]")
print("CM_0 is " ,prob["AS_point_0.CM"][1])
print("CM_1 is ",prob["AS_point_1.CM"][1])
print("Cg is ",prob["AS_point_0.cg"])

print("Cruise D is",prob["AS_point_0.total_perf.D"])
print("Cruise L is",prob["AS_point_0.total_perf.L"])

# Instantiate your CaseReader
cr = om.CaseReader("./Initial Final_out/aerostrf2.db")

# Get driver cases (do not recurse to system/solver cases)
driver_cases = cr.get_cases('driver', recurse=False)

# Plot the path the design variables took to convergence
# Note that there are five lines in the left plot because "wing.twist_cp"
# contains five variables that are being optimized
wingtwist_values = []
tailtwist_values = []
cl_values = []
lw_values = []
cl1_values = []
lw1_values = []
alpha_values = []
alpham_values = []
CM_values = []
drag_values = []
fuelburn_values = []
CM1_values = []
drag1_values = []
fuelburn1_values = []
skin_thic_values = []
spar_thic_values = []
span_values = []
sweep_values = []
chord_values = []
#warea_values = []

for case in driver_cases:
    wingtwist_values.append(case['wing.twist_cp'])
    tailtwist_values.append(case['tail.twist_cp'])
    cl_values.append(case['AS_point_0.wing_perf.Cl'])
    cl1_values.append(case['AS_point_1.wing_perf.Cl'])
    lw_values.append(case['AS_point_0.L_equals_W'])
    lw1_values.append(case['AS_point_1.L_equals_W'])
    alpha_values.append(case['alpha'])
    alpham_values.append(case['alpha_maneuver'])
    CM_values.append(case['AS_point_0.CM'])
    CM1_values.append(case['AS_point_1.CM'])
    drag_values.append(case['AS_point_0.total_perf.D'])
    drag1_values.append(case['AS_point_1.total_perf.D']) 
    fuelburn_values.append(case['AS_point_0.fuelburn'])
    fuelburn1_values.append(case['AS_point_1.fuelburn'])
    skin_thic_values.append(case['wing.skin_thickness_cp'])
    spar_thic_values.append(case['wing.spar_thickness_cp'])
    span_values.append(case['wing.geometry.span'])
    sweep_values.append(case['wing.sweep'])
    chord_values.append(case['wing.geometry.chord_cp'])
    #warea_values.append(case["AS_point_0.wing_perf.S_ref"])
    
###################################    
    
fig, (ax1) = plt.subplots(1)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

fig.suptitle('Function optimization history visualization', fontsize=16)

ax1.plot(np.arange(len(wingtwist_values)), np.array(wingtwist_values))
ax1.set(xlabel='Iterations', ylabel='Wing Twist cp', title='Optimization History')
ax1.legend(['cp1', 'cp2','cp3','cp4'])
ax1.grid()

####################################
   
fig, (ax2) = plt.subplots(1)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

fig.suptitle('Function optimization history visualization', fontsize=16)

ax2.plot(np.arange(len(tailtwist_values)), np.array(tailtwist_values))
ax2.set(xlabel='Iterations', ylabel='Tail Twist cp', title='Optimization History')
ax2.legend(['cp1'])
ax2.grid()

####################################

fig, (ax3) = plt.subplots(1)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

fig.suptitle('Function optimization history visualization', fontsize=16)

ax3.plot(np.arange(len(lw_values)), np.array(lw_values))
ax3.plot(np.arange(len(lw1_values)), np.array(lw1_values))
ax3.set(xlabel='Iterations', ylabel='L=W', title='Optimization History')
ax3.legend(['cruise', 'maneuver'])
ax3.grid()

####################################

fig, (ax4) = plt.subplots(1)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

fig.suptitle('Function optimization history visualization', fontsize=16)

ax4.plot(np.arange(len(alpha_values)), np.array(alpha_values))
ax4.plot(np.arange(len(alpham_values)), np.array(alpham_values))
ax4.set(xlabel='Iterations', ylabel='alpha', title='Optimization History')
ax4.legend(['alpha cruise', 'alpha maneuver'])
ax4.grid()

#####################################

fig, (ax5) = plt.subplots(1)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

fig.suptitle('Function optimization history visualization', fontsize=16)

ax5.plot(np.arange(len(CM_values)), np.array(CM_values))
#ax5.plot(np.arange(len(CM1_values)), np.array(CM1_values))
ax5.set(xlabel='Iterations', ylabel='Cm', title='Optimization History')
#ax5.legend(['cruise', 'maneuver'])
ax5.grid()

####################################

fig, (ax6) = plt.subplots(1)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

fig.suptitle('Function optimization history visualization', fontsize=16)

ax6.plot(np.arange(len(drag_values)), np.array(drag_values))
#ax6.plot(np.arange(len(drag1_values)), np.array(drag1_values))
ax6.set(xlabel='Iterations', ylabel='D', title='Optimization History')
#ax6.legend(['cruise', 'maneuver'])
ax6.grid()

###################################

fig, (ax7) = plt.subplots(1)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

fig.suptitle('Function optimization history visualization', fontsize=16)

ax7.plot(np.arange(len(fuelburn_values)), np.array(fuelburn_values))
#ax7.plot(np.arange(len(fuelburn1_values)), np.array(fuelburn1_values))
ax7.set(xlabel='Iterations', ylabel='Objetive Funtion - Fuelburn', title='Optimization History')
#ax7.legend(['cruise', 'maneuver'])
ax7.grid()

###################################

fig, (ax8) = plt.subplots(1)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

fig.suptitle('Function optimization history visualization', fontsize=16)

ax8.plot(np.arange(len(cl_values)), np.array(cl_values))
ax8.set(xlabel='Iterations', ylabel='cl cruise', title='Optimization History')
ax8.legend(['panel 1', 'panel 2', 'panel 3', 'panel 4', 'panel 5', 'panel 6', 'panel 7'])
ax8.grid()

###################################

fig, (ax9) = plt.subplots(1)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

fig.suptitle('Function optimization history visualization', fontsize=16)

ax9.plot(np.arange(len(cl1_values)), np.array(cl1_values))
ax9.set(xlabel='Iterations', ylabel='cl maneuver', title='Optimization History')
ax9.legend(['panel 1', 'panel 2', 'panel 3', 'panel 4', 'panel 5', 'panel 6', 'panel 7'])
ax9.grid()

###################################

fig, (ax10) = plt.subplots(1)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

fig.suptitle('Function optimization history visualization', fontsize=16)

ax10.plot(np.arange(len(spar_thic_values)), np.array(spar_thic_values))
ax10.set(xlabel='Iterations', ylabel='spar_thickness', title='Optimization History')
ax10.legend(['cp1', 'cp2','cp3','cp4'])
ax10.grid()

###################################

fig, (ax11) = plt.subplots(1)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

fig.suptitle('Function optimization history visualization', fontsize=16)

ax11.plot(np.arange(len(skin_thic_values)), np.array(skin_thic_values))
ax11.set(xlabel='Iterations', ylabel='skin_thickness', title='Optimization History')
ax11.legend(['cp1', 'cp2','cp3','cp4'])
ax11.grid()

###################################

#fig, (ax12) = plt.subplots(1)
#fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

#fig.suptitle('Function optimization history visualization', fontsize=16)

#ax12.plot(np.arange(len(warea_values)), np.array(warea_values))
#ax12.set(xlabel='Iterations', ylabel='wing area', title='Optimization History')
#ax12.grid()

###################################

fig, (ax13) = plt.subplots(1)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

fig.suptitle('Function optimization history visualization', fontsize=16)

ax13.plot(np.arange(len(span_values)), np.array(span_values))
ax13.set(xlabel='Iterations', ylabel='span', title='Optimization History')
ax13.grid()

###################################

fig, (ax14) = plt.subplots(1)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

fig.suptitle('Function optimization history visualization', fontsize=16)

ax14.plot(np.arange(len(sweep_values)), np.array(sweep_values))
ax14.set(xlabel='Iterations', ylabel='sweep', title='Optimization History')
ax14.legend(['cp1', 'cp2','cp3','cp4'])
ax14.grid()

###################################

fig, (ax15) = plt.subplots(1)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

fig.suptitle('Function optimization history visualization', fontsize=16)

ax15.plot(np.arange(len(chord_values)), np.array(chord_values))
ax15.set(xlabel='Iterations', ylabel='chord', title='Optimization History')
ax15.legend(['cp1', 'cp2','cp3','cp4'])
ax15.grid()

plt.show()
