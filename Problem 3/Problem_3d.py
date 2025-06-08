import numpy as np
import openmdao.api as om
import pandas as pd
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint
from problem_3c import LiftDragRatio, InducedDragFactor, SpanEfficiency

def calculate_flight_conditions():
    """Calculate flight conditions from cruise speed and altitude"""
    # Given parameters from table
    cruise_speed = 63  # m/s
    cruise_altitude = 2000  # m
    
    # Calculate atmospheric conditions at 2000m altitude
    # Using standard atmosphere model
    T0 = 288.15  # K (sea level temperature)
    p0 = 101325  # Pa (sea level pressure)
    rho0 = 1.225  # kg/m³ (sea level density)
    L = 0.0065  # K/m (temperature lapse rate)
    g = 9.80665  # m/s² (gravitational acceleration)
    R = 287.053  # J/(kg·K) (specific gas constant for air)
    
    # Temperature at altitude
    T = T0 - L * cruise_altitude
    
    # Pressure at altitude
    p = p0 * (T / T0) ** (g / (R * L))
    
    # Density at altitude
    rho = p / (R * T)
    
    # Speed of sound
    gamma = 1.4  # specific heat ratio for air
    a = np.sqrt(gamma * R * T)
    
    # Mach number
    mach = cruise_speed / a
    
    # Reynolds number per meter (approximate)
    mu = 1.789e-5 * (T / T0) ** 0.7  # dynamic viscosity
    re_per_m = rho * cruise_speed / mu
    
    return {
        'velocity': cruise_speed,
        'altitude': cruise_altitude,
        'density': rho,
        'temperature': T,
        'pressure': p,
        'mach_number': mach,
        'reynolds_per_m': re_per_m
    }

def create_rectangular_wing_mesh(wing_span, wing_area, num_y=7, num_x=2):
    """Create mesh for rectangular wing based on given parameters"""
    # Given parameters
    b = wing_span  # 11.0 m
    S = wing_area  # 16.2 m²
    
    # Calculate chord (for rectangular wing, chord is constant)
    chord = S / b
    
    # Create mesh points for half wing (due to symmetry)
    half_span = b / 2
    
    # Create spanwise distribution
    y_coords = np.linspace(0, half_span, num_y)
    
    # Create chordwise distribution  
    x_coords = np.linspace(0, chord, num_x)
    
    # Create mesh grid
    mesh = np.zeros((num_x, num_y, 3))
    
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            mesh[i, j, 0] = x  # x-coordinate (chordwise)
            mesh[i, j, 1] = y  # y-coordinate (spanwise)
            mesh[i, j, 2] = 0  # z-coordinate (no dihedral)
    
    # Create twist control points (no initial twist)
    twist_cp = np.zeros(5)
    
    return mesh, twist_cp, chord

def setup_base_problem():
    """Setup the base problem configuration with table parameters"""
    # Calculate flight conditions
    flight_conditions = calculate_flight_conditions()
    
    # Wing parameters from table
    wing_span = 11.0  # m
    wing_area = 16.2  # m²
    
    # Create rectangular wing mesh
    mesh, twist_cp, chord = create_rectangular_wing_mesh(wing_span, wing_area)
    
    # Create a dictionary with info and options about the aerodynamic lifting surface
    surface = {
        # Wing definition
        "name": "wing",  # name of the surface
        "symmetry": True,  # if true, model one half of wing reflected across y = 0
        "S_ref_type": "projected",  # use projected area (more appropriate for rectangular wing)
        "fem_model_type": "tube",
        "twist_cp": twist_cp,
        "mesh": mesh,
        
        # Aerodynamic performance at alpha=0
        "CL0": 0.0,  # CL of the surface at alpha=0
        "CD0": 0.008,  # CD of the surface at alpha=0 (reduced for cleaner rectangular wing)
        
        # Airfoil properties for NACA 2412
        "k_lam": 0.05,  # percentage of chord with laminar flow
        "t_over_c_cp": np.array([0.12]),  # thickness over chord ratio for NACA 2412
        "c_max_t": 0.30,  # chordwise location of maximum thickness for NACA 2412
        
        # Viscous and wave drag settings
        "with_viscous": True,  # compute viscous drag
        "with_wave": False,  # no wave drag at low Mach numbers
    }
    
    return surface, flight_conditions

def create_problem(surface, flight_conditions, design_vars):
    """Create and setup OpenMDAO problem with specified design variables"""
    prob = om.Problem()
    
    # Create independent variable component with calculated flight conditions
    indep_var_comp = om.IndepVarComp()
    indep_var_comp.add_output("v", val=flight_conditions['velocity'], units="m/s")
    indep_var_comp.add_output("alpha", val=5.0, units="deg")  # initial guess
    indep_var_comp.add_output("Mach_number", val=flight_conditions['mach_number'])
    indep_var_comp.add_output("re", val=flight_conditions['reynolds_per_m'], units="1/m")
    indep_var_comp.add_output("rho", val=flight_conditions['density'], units="kg/m**3")
    indep_var_comp.add_output("cg", val=np.zeros((3)), units="m")
    
    prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])
    
    # Create and add geometry group
    geom_group = Geometry(surface=surface)
    prob.model.add_subsystem(surface["name"], geom_group)
    
    # Create aero point group
    aero_group = AeroPoint(surfaces=[surface])
    point_name = "aero_point_0"
    prob.model.add_subsystem(
        point_name, aero_group, promotes_inputs=["v", "alpha", "Mach_number", "re", "rho", "cg"]
    )
    
    name = surface["name"]
    
    # Connect components
    prob.model.connect(name + ".mesh", point_name + "." + name + ".def_mesh")
    prob.model.connect(name + ".mesh", point_name + ".aero_states." + name + "_def_mesh")
    prob.model.connect(name + ".t_over_c", point_name + "." + name + "_perf." + "t_over_c")
    
    
    # Add Lift-to-Drag Ratio component
    ld_comp = LiftDragRatio(surface=surface)
    prob.model.add_subsystem("ld_ratio_comp", ld_comp,
                             promotes_inputs=["alpha", "beta"],
                             promotes_outputs=["L_over_D", "L", "D"])
    prob.model.connect(point_name + ".aero_states.wing_sec_forces", "ld_ratio_comp.sec_forces")

    # Add Induced Drag Factor component
    drag_factor_comp = InducedDragFactor(surface=surface)
    prob.model.add_subsystem("drag_factor_comp", drag_factor_comp, promotes=["*"])

    # Add Span Efficiency component
    span_eff_comp = SpanEfficiency(surface=surface)
    prob.model.add_subsystem("span_efficiency_comp", span_eff_comp, promotes=["*"])
    prob.model.connect(point_name + ".wing_perf.CL", "CL")
    prob.model.connect(point_name + ".wing_perf.CDi", "CDi")


    # Setup driver
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["tol"] = 1e-9
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["maxiter"] = 500

    # Add design variables based on the case
    if "alpha" in design_vars:
        prob.model.add_design_var("alpha", lower=-15.0, upper=15.0)
    
    if "twist" in design_vars:
        prob.model.add_design_var("wing.twist_cp", lower=-10.0, upper=15.0)
    
    if "chord" in design_vars:
        # For chord distribution, add chord control points to the surface
        num_chord_cp = 5
        chord_cp = np.ones(num_chord_cp)  
        surface["chord_cp"] = chord_cp
        prob.model.add_design_var("wing.chord_cp", lower=0.5, upper=2.0)
    
    # Add constraint and objective
    prob.model.add_constraint(point_name + ".wing_perf.CL", equals=0.5)
    prob.model.add_objective(point_name + ".wing_perf.CD", scaler=1e4)

    # Additional aerodynamic efficiency constraints
    prob.model.add_constraint("L_over_D", lower=8.0)
    prob.model.add_constraint("drag_factor", upper=0.045)
    prob.model.add_constraint("span_efficiency", lower=0.6)

    
    return prob, point_name

def run_optimization_case(case_name, design_vars, surface, flight_conditions):
    """Run optimization for a specific case and return results"""
    print(f"\n--- Running Case: {case_name} ---")
    print(f"Design variables: {design_vars}")
    print(f"Flight conditions: v={flight_conditions['velocity']:.1f} m/s, "
          f"h={flight_conditions['altitude']:.0f} m, M={flight_conditions['mach_number']:.3f}")
    
    try:
        # Create problem
        prob, point_name = create_problem(surface.copy(), flight_conditions, design_vars)
        
        # Setup and run
        prob.setup()
        
        # Record initial values
        initial_alpha = prob.get_val("alpha")[0] if "alpha" in design_vars else "N/A"
        
        # Run optimization
        prob.run_driver()
        
        # Extract results
        final_alpha = prob.get_val("alpha")[0] if "alpha" in design_vars else "N/A"
        final_CD = prob.get_val(point_name + ".wing_perf.CD")[0]
        final_CL = prob.get_val(point_name + ".wing_perf.CL")[0]
        
        # Get twist results if applicable
        final_twist = "N/A"
        if "twist" in design_vars:
            twist_vals = prob.get_val("wing.twist_cp")
            final_twist = f"[{', '.join([f'{t:.2f}' for t in twist_vals])}]"
        
        # Get chord results if applicable  
        final_chord = "N/A"
        if "chord" in design_vars:
            chord_vals = prob.get_val("wing.chord_cp")
            final_chord = f"[{', '.join([f'{c:.2f}' for c in chord_vals])}]"
        
        # Get optimization statistics
        driver = prob.driver
        success = driver.iter_count is not None
        iterations = getattr(driver, 'iter_count', 'N/A')
        
        func_evals = 'N/A'
        grad_evals = 'N/A'
        
        if hasattr(driver, 'result') and driver.result is not None:
            func_evals = getattr(driver.result, 'nfev', 'N/A')
            grad_evals = getattr(driver.result, 'njev', 'N/A')
        
        result = {
            'Case': case_name,
            'Design Variables': ', '.join(design_vars),
            'Success': success,
            'Final CD': final_CD,
            'Final CL': final_CL,
            'Final Alpha (deg)': final_alpha,
            'Final Twist': final_twist,
            'Final Chord': final_chord,
            'Iterations': iterations,
            'Function Evaluations': func_evals,
            'Gradient Evaluations': grad_evals
        }
        
        print(f"Success: {success}")
        print(f"Final CD: {final_CD:.6f}")
        print(f"Final CL: {final_CL:.6f}")
        print(f"Final L/D Ratio: {prob.get_val('L_over_D')[0]:.2f}")
        print(f"Final Drag Factor: {prob.get_val('drag_factor')[0]:.4f}")
        print(f"Final Span Efficiency: {prob.get_val('span_efficiency')[0]:.4f}")
        print(f"Final Alpha: {final_alpha:.3f} deg" if final_alpha != "N/A" else "Final Alpha: N/A")
        if final_twist != "N/A":
            print(f"Final Twist: {final_twist}")
        if final_chord != "N/A":
            print(f"Final Chord: {final_chord}")
        
        return result
        
    except Exception as e:
        print(f"Error in case {case_name}: {str(e)}")
        return {
            'Case': case_name,
            'Design Variables': ', '.join(design_vars),
            'Success': False,
            'Final CD': 'Error',
            'Final CL': 'Error',
            'Final Alpha (deg)': 'Error',
            'Final Twist': 'Error',
            'Final Chord': 'Error',
            'Iterations': 'Error',
            'Function Evaluations': 'Error',
            'Gradient Evaluations': 'Error',
            'Error': str(e)
        }

def main():
    """Main function to run all optimization cases"""
    print("Starting Multi-Case Aerodynamic Optimization Study")
    print("Aircraft Parameters:")
    print("- Cruise speed: 63 m/s")
    print("- Cruise altitude: 2000 m") 
    print("- Wing span: 11.0 m")
    print("- Wing area: 16.2 m²")
    print("- Wing section: NACA 2412")
    print("- Wing type: rectangular")
    print("=" * 60)
    
    # Setup base surface configuration and flight conditions
    surface, flight_conditions = setup_base_problem()
    
    # Define the four cases
    cases = [
        ("Case 1: Alpha Only", ["alpha"]),
        ("Case 2: Alpha + Twist", ["alpha", "twist"]),
        ("Case 3: Alpha + Chord", ["alpha", "chord"]),
        ("Case 4: Alpha + Twist + Chord", ["alpha", "twist", "chord"])
    ]
    
    # Run all cases
    results = []
    for case_name, design_vars in cases:
        result = run_optimization_case(case_name, design_vars, surface, flight_conditions)
        results.append(result)
    
    # Create results table
    df = pd.DataFrame(results)
    
    # Sounds good, looks messy
    #print("\n" + "=" * 100)
    #print("OPTIMIZATION RESULTS SUMMARY")
    #print("=" * 100)
    #print(df.to_string(index=False))
    
    return df

if __name__ == "__main__":
    results_df = main()