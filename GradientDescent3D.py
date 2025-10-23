import sympy as sp
import numpy as np

x, y = sp.symbols('x y')
z_form = np.sin(5 * x) * np.cos(5 * y) / 5
z_derv_form = sp.diff(z_form, x), sp.diff(z_form, y)


x_range = np.arange(-10, 10, 0.01)
y_range = np.arange(-10, 10, 0.01)

X, Y = np.meshgrid(x_range, y_range)
print(X, Y)