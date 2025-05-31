import openmdao.api as om
import numpy as np

# This script sets up an optimization problem using OpenMDAO to minimize the

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

# Set up the optimization problem
prob = om.Problem()
model = prob.model

model.add_subsystem('goldstein', GoldsteinPriceComp(), promotes=['*'])

# Design variables and objective
model.add_design_var('x1')
model.add_design_var('x2')
model.add_objective('f')

# Approximate total derivatives
model.approx_totals(method='fd', form='backward', step=1e-6)

# Use SLSQP optimizer
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['disp'] = True

# Initialize and run
prob.setup()
prob.set_val('x1', 4.0)
prob.set_val('x2', 2.0)

prob.run_driver()

# Output results
print("\n=== Optimization Result (SLSQP) ===")
print(f"x1* = {prob.get_val('x1')[0]:.6f}")
print(f"x2* = {prob.get_val('x2')[0]:.6f}")
print(f"f*  = {prob.get_val('f')[0]:.6f}")
