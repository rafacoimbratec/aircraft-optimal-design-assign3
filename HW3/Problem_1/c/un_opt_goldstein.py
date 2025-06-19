import openmdao.api as om
import numpy as np
import time  # Para medir o tempo de execução

# Componente da função Goldstein-Price
class GoldsteinPriceComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x1', val=0.0)
        self.add_input('x2', val=0.0)
        self.add_output('f', val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')  # Usa diferenças finitas padrão

    def compute(self, inputs, outputs):
        x1 = inputs['x1']
        x2 = inputs['x2']
        A = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
        B = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
        outputs['f'] = A * B

# Criação do problema
prob = om.Problem()
model = prob.model

model.add_subsystem('goldstein', GoldsteinPriceComp(), promotes=['*'])

# Variáveis de design e função objetivo
model.add_design_var('x1')
model.add_design_var('x2')
model.add_objective('f')

# Aproxima derivadas totais com diferenças finitas (default backward, step padrão)
model.approx_totals()

# Otimizador com opções padrão
prob.driver = om.ScipyOptimizeDriver()

# Print default options for the optimizer
print("\nScipyOptimizeDriver method:", prob.driver.options['optimizer'])
print("\nDefault ScipyOptimizeDriver options:")
for key, value in prob.driver.options.items():
    print(f"{key}: {value}")

# Inicializa e corre
prob.setup()
prob.set_val('x1', 4.0)
prob.set_val('x2', 2.0)

# Medição do tempo de execução
start_time = time.process_time()

prob.run_driver()

end_time = time.process_time()
cpu_time = end_time - start_time

# Mostra os resultados
print(f"x1* = {prob.get_val('x1')[0]:.6f}")
print(f"x2* = {prob.get_val('x2')[0]:.6f}")
print(f"f*  = {prob.get_val('f')[0]:.6f}")
print(f"CPU time: {cpu_time:.6f} seconds")
