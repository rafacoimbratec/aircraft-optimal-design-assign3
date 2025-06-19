import numpy as np
import openmdao.api as om
import pandas as pd
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

def setup_base_problem():
    """Setup the base problem configuration that's common to all cases"""
    # Create a dictionary to store options about the mesh
    mesh_dict = {"num_y": 7, "num_x": 2, "wing_type": "CRM", "symmetry": True, "num_twist_cp": 5}
    
    # Generate the aerodynamic mesh based on the previous dictionary
    mesh, twist_cp = generate_mesh(mesh_dict)
    
    # Create a dictionary with info and options about the aerodynamic lifting surface
    surface = {
        "name": "wing",
        "symmetry": True,
        "S_ref_type": "wetted",
        "fem_model_type": "tube",
        "twist_cp": twist_cp,
        "mesh": mesh,
        "CL0": 0.0,
        "CD0": 0.015,
        "k_lam": 0.05,
        "t_over_c_cp": np.array([0.15]),
        "c_max_t": 0.303,
        "with_viscous": True,
        "with_wave": False,
    }
    
    return surface

def create_problem(surface, design_vars):
    """Create and setup OpenMDAO problem with specified design variables"""
    prob = om.Problem()
    
    # Create independent variable component
    indep_var_comp = om.IndepVarComp()
    indep_var_comp.add_output("v", val=248.136, units="m/s")
    indep_var_comp.add_output("alpha", val=5.0, units="deg")
    indep_var_comp.add_output("Mach_number", val=0.84)
    indep_var_comp.add_output("re", val=1.0e6, units="1/m")
    indep_var_comp.add_output("rho", val=0.38, units="kg/m**3")
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
    
    # Setup driver
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["tol"] = 1e-9
    
    # Add design variables based on the case
    if "alpha" in design_vars:
        prob.model.add_design_var("alpha", lower=-10.0, upper=10.0)
    
    if "twist" in design_vars:
        prob.model.add_design_var("wing.twist_cp", lower=-10.0, upper=15.0)
    
    if "chord" in design_vars:
        # For chord distribution, add chord control points to the surface
        # Mdifying the surface dictionary to include chord_cp
        num_chord_cp = 5  # adjustable, maybe lets play with this
        chord_cp = np.ones(num_chord_cp)  
        surface["chord_cp"] = chord_cp
        prob.model.add_design_var("wing.chord_cp", lower=0.5, upper=2.0)
    
    # Add constraint and objective
    prob.model.add_constraint(point_name + ".wing_perf.CL", equals=0.5)
    prob.model.add_objective(point_name + ".wing_perf.CD", scaler=1e4)
    
    return prob, point_name

def run_optimization_case(case_name, design_vars, surface):
    """Run optimization for a specific case and return results"""
    print(f"\n--- Running Case: {case_name} ---")
    print(f"Design variables: {design_vars}")
    
    try:
        # Create problem
        prob, point_name = create_problem(surface.copy(), design_vars)
        
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
        
        # Get optimization statistics
        driver = prob.driver

        success = driver.iter_count is not None
        iterations = getattr(driver, 'iter_count', 'N/A')
        
        # For function evaluations, we need to access the optimizer result so for now N/A
        # Worst case scenario we fill the table manually
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
            'Iterations': iterations,
            'Function Evaluations': func_evals,
            'Gradient Evaluations': grad_evals
        }
        
        print(f"Success: {success}")
        print(f"Final CD: {final_CD:.6f}")
        print(f"Final CL: {final_CL:.6f}")
        print(f"Final Alpha: {final_alpha:.3f} deg" if final_alpha != "N/A" else "Final Alpha: N/A")
        
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
            'Iterations': 'Error',
            'Function Evaluations': 'Error',
            'Gradient Evaluations': 'Error',
            'Error': str(e)
        }

def main():
    """Main function to run all optimization cases"""
    print("Starting Multi-Case Aerodynamic Optimization Study")
    print("=" * 60)
    
    # Setup base surface configuration
    surface = setup_base_problem()
    
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
        result = run_optimization_case(case_name, design_vars, surface)
        results.append(result)
    
    # Create results table
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    
    return df

if __name__ == "__main__":
    results_df = main()