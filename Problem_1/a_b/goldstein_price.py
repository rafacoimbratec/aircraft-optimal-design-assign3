import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define symbolic variables
x1, x2 = sp.symbols('x1 x2')

# Define the components of the Goldstein-Price function
A = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
B = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)

# Goldstein-Price function
f = A * B

# Compute partial derivatives
df_dx1 = sp.diff(f, x1)
df_dx2 = sp.diff(f, x2)

# Display symbolic results
print("Goldstein-Price function:")
sp.pprint(f, use_unicode=True)

print("\nPartial derivative with respect to x1:")
sp.pprint(df_dx1, use_unicode=True)

print("\nPartial derivative with respect to x2:")
sp.pprint(df_dx2, use_unicode=True)

# Optionally evaluate at a point, e.g., (x1, x2) = (0, -1)
val = f.subs({x1: 0, x2: -1})
grad_val = (df_dx1.subs({x1: 0, x2: -1}), df_dx2.subs({x1: 0, x2: -1}))

print(f"\nFunction value at (0, -1): {val}")
print(f"Gradient at (0, -1): {grad_val}")


# Lambdify for numerical evaluation
f_func = sp.lambdify((x1, x2), f, modules='numpy')

# Create grid over [-2, 2] for both x1 and x2
X1_vals = np.linspace(-2, 2, 400)
X2_vals = np.linspace(-2, 2, 400)
X1, X2 = np.meshgrid(X1_vals, X2_vals)

# Evaluate function on the grid
Z = f_func(X1, X2)

# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none')
ax.set_title('Goldstein-Price Function')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1, x_2)$')
plt.tight_layout()
plt.show()