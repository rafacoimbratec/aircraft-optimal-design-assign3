
import numpy as np
import openmdao.api as om
import pandas as pd
import matplotlib.pyplot as plt
import time
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

def calculate_flight_conditions():
    cruise_speed = 63
    cruise_altitude = 2000
    T0 = 288.15; p0 = 101325; L = 0.00649; g = 9.80665; R = 287.053
    T = T0 - L * cruise_altitude
    p = p0 * (T / T0) ** (g / (R * L))
    rho = p / (R * T)
    a = np.sqrt(1.4 * R * T)
    mach = cruise_speed / a
    mu = 1.789e-5 * (T / T0) ** 0.7
    re_per_m = rho * cruise_speed / mu
    return {'velocity': cruise_speed, 'altitude': cruise_altitude,
            'density': rho, 'temperature': T, 'pressure': p,
            'mach_number': mach, 'reynolds_per_m': re_per_m}

def create_rectangular_wing_mesh(wing_span, wing_area, num_y=7, num_x=2):
    chord = wing_area / wing_span
    y_coords = np.linspace(0, wing_span / 2, num_y)
    x_coords = np.linspace(0, chord, num_x)
    mesh = np.zeros((num_x, num_y, 3))
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            mesh[i, j, 0] = x
            mesh[i, j, 1] = y
    return mesh, np.zeros(5), chord

def setup_base_problem():
    fc = calculate_flight_conditions()
    mesh, twist_cp, _ = create_rectangular_wing_mesh(11.0, 16.2)
    surface = {
        'name':'wing','symmetry':True,'S_ref_type':'projected','fem_model_type':'tube',
        'twist_cp':twist_cp,'mesh':mesh,'S_ref':16.2,'CL0':0.0,'CD0':0.008,
        'k_lam':0.05,'t_over_c_cp':np.array([0.12]),'c_max_t':0.30,
        'with_viscous':True,'with_wave':False
    }
    return surface, fc

def create_problem(surface, fc, design_vars):
    prob = om.Problem()
    m = prob.model
    indep = om.IndepVarComp()
    indep.add_output('v', val=fc['velocity'], units='m/s')
    indep.add_output('alpha', val=5.0, units='deg')
    indep.add_output('Mach_number', val=fc['mach_number'])
    indep.add_output('re', val=fc['reynolds_per_m'], units='1/m')
    indep.add_output('rho', val=fc['density'], units='kg/m**3')
    indep.add_output('cg', val=np.zeros(3), units='m')
    m.add_subsystem('prob_vars', indep, promotes=['*'])
    m.add_subsystem('wing', Geometry(surface=surface))
    m.add_subsystem('aero_point_0', AeroPoint(surfaces=[surface]),
                    promotes_inputs=['v','alpha','Mach_number','re','rho','cg'])
    m.connect('wing.mesh', 'aero_point_0.wing.def_mesh')
    m.connect('wing.mesh', 'aero_point_0.aero_states.wing_def_mesh')
    m.connect('wing.t_over_c', 'aero_point_0.wing_perf.t_over_c')
    prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, maxiter=500)
    if 'alpha' in design_vars: m.add_design_var('alpha', lower=-15, upper=15)
    if 'twist' in design_vars: m.add_design_var('wing.twist_cp', lower=-10, upper=15)
    if 'chord' in design_vars:
        surface['chord_cp'] = np.ones(5)
        m.add_design_var('wing.chord_cp', lower=0.5, upper=2.0)
    m.add_constraint('aero_point_0.wing_perf.CL', equals=0.5)
    m.add_objective('aero_point_0.wing_perf.CD', scaler=1e4)
    return prob, 'aero_point_0'

def run_optimization_case(surface, fc, design_vars):
    prob, point = create_problem(surface.copy(), fc, design_vars)
    prob.setup()
    t0 = time.time(); prob.run_driver(); dt = time.time()-t0
    CD = prob.get_val(f'{point}.wing_perf.CD')[0]
    return CD, dt, prob.get_val('wing.twist_cp'), prob.get_val('wing.chord_cp')

def main_case4_mesh_study():
    surface0, fc = setup_base_problem()
    design_vars = ['alpha','twist','chord']
    MESHES = [(2,5),(3,9),(4,13),(5,17),(6,21)]
    results = []
    for nx, ny in MESHES:
        print(f"\n=== Mesh {nx}x{ny} ===")
        surface = surface0.copy()
        mesh, twist_cp, _ = create_rectangular_wing_mesh(11.0,16.2,ny,nx)
        surface['mesh'] = mesh; surface['twist_cp'] = twist_cp
        CD, dt, twist, chord = run_optimization_case(surface, fc, design_vars)
        results.append({'mesh':f"{nx}x{ny}", 'CD':CD, 'time':dt})
        span = np.linspace(0, 11.0/2, len(twist))
        plt.figure(); plt.plot(span, twist, '-o'); plt.title(f"Twist {nx}x{ny}"); plt.savefig(f"twist_{nx}x{ny}.png"); plt.close()
        plt.figure(); plt.plot(span, chord, '-o'); plt.title(f"Chord {nx}x{ny}"); plt.savefig(f"chord_{nx}x{ny}.png"); plt.close()
    df = pd.DataFrame(results)
    df.to_csv("mesh_convergence_case4.csv", index=False)
    plt.figure(); plt.plot(df['mesh'], df['CD'], '-o'); plt.title("CD vs Mesh Resolution (Case 4)"); plt.savefig("CD_mesh_case4.png")
    print("Convergence study complete.")

if __name__ == '__main__':
    main_case4_mesh_study()
