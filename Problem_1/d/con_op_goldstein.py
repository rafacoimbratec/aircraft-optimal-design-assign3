import openmdao.api as om
import numpy as np
from openmdao.devtools import iprofile as tool

# Start profiling
tool.start()

class GoldsteinPriceComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x1', val=0.0)
        self.add_input('x2', val=0.0)
        self.add_output('f', val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd', form='backward',
                              step=1e-6, step_calc='abs', minimum_step=1e-6)

    def compute(self, inputs, outputs):
        x1 = inputs['x1']
        x2 = inputs['x2']
        A = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
        B = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
        outputs['f'] = A * B

def run_optimization(x1_init, x2_init, label):
    prob = om.Problem()
    model = prob.model

    # Set input defaults
    model.set_input_defaults('x1', val=0.0)
    model.set_input_defaults('x2', val=0.0)

    # Add the Goldstein-Price component
    model.add_subsystem('goldstein', GoldsteinPriceComp(), promotes=['*'])

    # Add constraint: -x1 - x2 <= 0
    model.add_subsystem('sum_constraint_comp', 
                        om.ExecComp('sum_constraint = -x1 - x2'), 
                        promotes=['x1', 'x2', 'sum_constraint'])
    model.add_constraint('sum_constraint', upper=0.0)

    # Design variables and objective
    model.add_design_var('x1')  # Optionally add lower/upper bounds
    model.add_design_var('x2')
    model.add_objective('f')

    # Approximate total derivatives with FD
    model.approx_totals(method='fd', form='backward', step=1e-6)

    # Configure optimizer
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['disp'] = True
    prob.driver.options['debug_print'] = ['totals']

    # Setup and run
    prob.setup()
    prob.set_val('x1', x1_init)
    prob.set_val('x2', x2_init)
    prob.run_driver()

    # Output results
    print(f"\n=== Optimization Result ({label}) ===")
    print(f"x1* = {prob.get_val('x1')[0]:.6f}")
    print(f"x2* = {prob.get_val('x2')[0]:.6f}")
    print(f"f*  = {prob.get_val('f')[0]:.6f}")

# Run optimization for multiple initial guesses
initial_guesses = [(4.0, 2.0), (1.8, 0.2), (1, 2)]
for x1_init, x2_init in initial_guesses:
    run_optimization(x1_init, x2_init, f'initial_guess_({x1_init},{x2_init})')

# Stop profiling
tool.stop()
