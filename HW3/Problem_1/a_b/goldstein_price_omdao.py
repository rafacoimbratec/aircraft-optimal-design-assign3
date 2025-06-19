import openmdao.api as om
import numpy as np

class GoldsteinPriceComp(om.ExplicitComponent):
    def setup(self):
        # Inputs
        self.add_input('x1', val=0.0)
        self.add_input('x2', val=0.0)

        # Output
        self.add_output('f', val=0.0)
        
    def setup_partials(self):
        # Finite difference all partials
        self.declare_partials('*', '*', method='fd', form='backward',
                              step=1e-6, step_calc='abs', minimum_step=1e-6)

    def compute(self, inputs, outputs):
        x1 = inputs['x1']
        x2 = inputs['x2']

        A = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
        B = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)

        outputs['f'] = A * B


# Build the problem
prob = om.Problem()
model = prob.model

model.add_subsystem('goldstein', GoldsteinPriceComp(), promotes=['*'])

# Setup the problem
prob.setup()

# Set input values
prob.set_val('x1', 0.0)
prob.set_val('x2', -1.0)

# Run the model
prob.run_model()

# Print result
print(f"f(x1=0, x2=-1) = {prob.get_val('f')}")

# Generate N2 diagram
om.n2(prob, show_browser=True, outfile='n2_goldstein.html')
