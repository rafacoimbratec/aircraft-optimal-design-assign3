import openmdao.api as om
import numpy as np
import time

# Função objetivo: Goldstein-Price
class GoldsteinPriceComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x1', val=0.0)
        self.add_input('x2', val=0.0)
        self.add_output('f', val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        x1 = inputs['x1']
        x2 = inputs['x2']
        A = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
        B = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
        outputs['f'] = A * B

# Componente de restrição: x1 + x2 >= 0 <=> -(x1 + x2) <= 0
class ConstraintComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x1', val=0.0)
        self.add_input('x2', val=0.0)
        self.add_output('constraint', val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        outputs['constraint'] = -inputs['x1'] - inputs['x2']

# Setup do problema
prob = om.Problem()
model = prob.model

model.add_subsystem('goldstein', GoldsteinPriceComp(), promotes=['*'])
model.add_subsystem('constraint_comp', ConstraintComp(), promotes=['*'])

# Variáveis de design, objetivo e restrição
model.add_design_var('x1')
model.add_design_var('x2')
model.add_objective('f')
model.add_constraint('constraint', upper=0.0)  # -x1 - x2 <= 0

# Derivadas aproximadas
model.approx_totals()

# Driver default
prob.driver = om.ScipyOptimizeDriver()

# Inicialização
prob.setup()
prob.set_val('x1', 1.2)
prob.set_val('x2', -1.2)

# Tempo de CPU
start_time = time.process_time()
prob.run_driver()
end_time = time.process_time()

# Resultados
print("\n=== Optimization Result with Constraint x1 + x2 ≥ 0 ===")
print(f"x1* = {prob.get_val('x1')[0]:.6f}")
print(f"x2* = {prob.get_val('x2')[0]:.6f}")
print(f"f*  = {prob.get_val('f')[0]:.6f}")
print(f"Constraint (should be ≤ 0): {-prob.get_val('x1')[0] - prob.get_val('x2')[0]:.6f}")
print(f"CPU time: {end_time - start_time:.6f} seconds")

