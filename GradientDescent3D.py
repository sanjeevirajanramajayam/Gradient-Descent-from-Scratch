import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

x, y = sp.symbols('x y')

z_form = sp.sin(5 * x) * sp.cos(5 * y) / 5
dz_dx_function = sp.lambdify((x, y), sp.diff(z_form, x), 'numpy')
dz_dy_function = sp.lambdify((x, y), sp.diff(z_form, y), 'numpy')
z_function = sp.lambdify((x, y), z_form, 'numpy')

x_range = np.arange(-1, 1, 0.05)
y_range = np.arange(-1, 1, 0.05)
X, Y = np.meshgrid(x_range, y_range)
Z = z_function(X, Y)

current_pos = [-1.4, 0.6, z_function(-1.4, 0.6)]
learning_rate = 0.01

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

ball = ax.scatter(current_pos[0], current_pos[1], current_pos[2], color='magenta', s=50)

for _ in range(100):
    dz_dx = dz_dx_function(current_pos[0], current_pos[1])
    dz_dy = dz_dy_function(current_pos[0], current_pos[1])
    
    current_pos[0] -= learning_rate * dz_dx
    current_pos[1] -= learning_rate * dz_dy
    current_pos[2] = z_function(current_pos[0], current_pos[1])
    
    ball._offsets3d = ([current_pos[0]], [current_pos[1]], [current_pos[2]])
    plt.pause(0.05)  

plt.show()
