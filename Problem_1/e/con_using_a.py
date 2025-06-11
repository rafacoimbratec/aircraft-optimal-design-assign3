import openmdao.api as om
import numpy as np
import sympy as sp
import time

# Symbolic setup for analytic derivatives
x1_sym, x2_sym = sp.symbols('x1 x2')
A = 1 + (x1_sym + x2_sym + 1)**2 * (19 - 14*x1_sym + 3*x1_sym**2 - 14*x2_sym + 6*x1_sym*x2_sym + 3*x2_sym**2)
B = 30 + (2*x1_sym - 3*x2_sym)**2 * (18 - 32*x1_sym + 12*x1_sym**2 + 48*x2_sym - 36*x1_sym*x2_sym + 27*x2_sym**2)
f_expr = A * B

df_dx1_expr = sp.diff(f_expr, x1_sym)
df_dx2_expr = sp.diff(f_expr, x2_sym)

f_func = sp.lambdify((x1_sym, x2_sym), f_expr, modules='numpy')
df_dx1_func = sp.lambdify((x1_sym, x2_sym), df_dx1_expr, modules='numpy')
df_dx2_func = sp.lambdify((x1_sym, x2_sym), df_dx2_expr, modules='numpy')

# Goldstein-Price component with analytic derivatives
class GoldsteinPriceComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x1', val=0.0)
        self.add_input('x2', val=0.0)
        self.add_output('f', val=0.0)

        self.declare_partials('f', ['x1', 'x2'])

    def compute(self, inputs, outputs):
        x1 = inputs['x1']
        x2 = inputs['x2']
        outputs['f'] = f_func(x1, x2)

    def compute_partials(self, inputs, partials):
        x1 = inputs['x1']
        x2 = inputs['x2']
        partials['f', 'x1'] = df_dx1_func(x1, x2)
        partials['f', 'x2'] = df_dx2_func(x1, x2)

# Constraint component (same as before)
class ConstraintComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x1', val=0.0)
        self.add_input('x2', val=0.0)
        self.add_output('constraint', val=0.0)
        self.declare_partials('constraint', ['x1', 'x2'])  # Use analytic
        self.eval_count = 0  # Counter for function evaluations

    def compute(self, inputs, outputs):
        outputs['constraint'] = -inputs['x1'] - inputs['x2']
        self.eval_count += 1

    def compute_partials(self, inputs, partials):
        partials['constraint', 'x1'] = -1.0
        partials['constraint', 'x2'] = -1.0

    def get_eval_count(self):
        return self.eval_count

# Initial conditions
initial_conditions = [
    (4.0, 2.0),
    (1.799960, 0.199973)
]

for idx, (x1_init, x2_init) in enumerate(initial_conditions, 1):
    print(f"\n=== Run {idx}: Initial condition (x1, x2) = ({x1_init}, {x2_init}) ===")
    prob = om.Problem()
    model = prob.model

    model.add_subsystem('goldstein', GoldsteinPriceComp(), promotes=['*'])
    model.add_subsystem('constraint_comp', ConstraintComp(), promotes=['*'])

    model.add_design_var('x1')
    model.add_design_var('x2')
    model.add_objective('f')
    model.add_constraint('constraint', upper=0.0)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['disp'] = True

    prob.setup()
    prob.set_val('x1', x1_init)
    prob.set_val('x2', x2_init)

    start_time = time.process_time()
    prob.run_driver()
    end_time = time.process_time()

    # Get constraint function evaluation count
    constraint_comp = prob.model.constraint_comp
    print("\n=== Optimization Result with Constraint x1 + x2 ≥ 0 ===")
    print(f"x1* = {prob.get_val('x1')[0]:.6f}")
    print(f"x2* = {prob.get_val('x2')[0]:.6f}")
    print(f"f*  = {prob.get_val('f')[0]:.6f}")
    print(f"Constraint (should be ≤ 0): {-prob.get_val('x1')[0] - prob.get_val('x2')[0]:.6f}")
    print(f"CPU time: {end_time - start_time:.6f} seconds")
    print(f"Constraint function evaluations: {constraint_comp.get_eval_count()}")

