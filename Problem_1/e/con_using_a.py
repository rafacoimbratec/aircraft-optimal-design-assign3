import openmdao.api as om
import numpy as np
from openmdao.devtools import iprofile as tool

tool.start()

class GoldsteinPriceComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x1', val=0.0)
        self.add_input('x2', val=0.0)
        self.add_output('f', val=0.0)
        self.declare_partials('f', ['x1', 'x2'])

    def compute(self, inputs, outputs):
        x1 = inputs['x1']
        x2 = inputs['x2']

        t1 = x1 + x2 + 1
        t2 = 2*x1 - 3*x2

        A = 1 + t1**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
        B = 30 + t2**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)

        outputs['f'] = A * B

    def compute_partials(self, inputs, partials):
        x1 = inputs['x1']
        x2 = inputs['x2']

        t1 = x1 + x2 + 1
        t2 = 2*x1 - 3*x2

        P = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
        dP_dx1 = -14 + 6*x1 + 6*x2
        dP_dx2 = -14 + 6*x1 + 6*x2

        Q = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
        dQ_dx1 = -32 + 24*x1 - 36*x2
        dQ_dx2 = 48 - 36*x1 + 54*x2

        A = 1 + t1**2 * P
        B = 30 + t2**2 * Q

        dA_dx1 = 2*t1 * P + t1**2 * dP_dx1
        dA_dx2 = 2*t1 * P + t1**2 * dP_dx2

        dB_dx1 = 4*t2 * Q + t2**2 * dQ_dx1
        dB_dx2 = -6*t2 * Q + t2**2 * dQ_dx2

        partials['f', 'x1'] = dA_dx1 * B + A * dB_dx1
        partials['f', 'x2'] = dA_dx2 * B + A * dB_dx2


def run_optimization(x1_init, x2_init, label):
    prob = om.Problem()
    model = prob.model

    model.set_input_defaults('x1', val=0.0)
    model.set_input_defaults('x2', val=0.0)

    model.add_subsystem('goldstein', GoldsteinPriceComp(), promotes=['*'])

    model.add_subsystem('sum_constraint_comp', 
                        om.ExecComp('sum_constraint = -x1 - x2'), 
                        promotes=['x1', 'x2', 'sum_constraint'])
    model.add_constraint('sum_constraint', upper=0.0)

    model.add_design_var('x1')
    model.add_design_var('x2')
    model.add_objective('f')

    # No need for approx_totals when using analytic partials

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['disp'] = True
    prob.driver.options['debug_print'] = ['totals']

    prob.setup()
    prob.set_val('x1', x1_init)
    prob.set_val('x2', x2_init)

    prob.run_driver()

    print(f"\n=== Optimization Result ({label}) ===")
    print(f"x1* = {prob.get_val('x1')[0]:.6f}")
    print(f"x2* = {prob.get_val('x2')[0]:.6f}")
    print(f"f*  = {prob.get_val('f')[0]:.6f}")


# Run with various initial guesses
initial_guesses = [(4.0, 2.0), (1.8, 0.2), (1.0, 2.0)]
for x1_init, x2_init in initial_guesses:
    run_optimization(x1_init, x2_init, f'initial_guess_({x1_init},{x2_init})')

tool.stop()

