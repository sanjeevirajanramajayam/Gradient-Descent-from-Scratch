import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# Define symbol
x = sp.Symbol('x')

# Define function and derivative symbolically
y_expr = x**2
y_deriv_expr = sp.diff(y_expr, x)

# Convert SymPy expressions to NumPy functions
y_function = sp.lambdify(x, y_expr, 'numpy')
y_derivative = sp.lambdify(x, y_deriv_expr, 'numpy')

learning_rate = 0.01

starting_pos = 8, y_function(8)

x_range = np.arange(-10, 10, 0.1)
y_range = y_function(x_range)
plt.ion()
fig, ax = plt.subplots()
ax.plot(x_range, y_range, label="y = x^2")
point, = ax.plot([starting_pos[0]], [starting_pos[1]], 'ro', label='Current Position')
ax.legend()

while True:
    old_x = starting_pos[0]
    new_x = starting_pos[0] - learning_rate * y_derivative(starting_pos[0])

    if abs(old_x - new_x) < 1e-3:
        break
    new_y = y_function(new_x)

    ax.plot([old_x, new_x], [starting_pos[1], new_y], 'm-')

    starting_pos = new_x, new_y
    point.set_data([starting_pos[0]], [starting_pos[1]])
    plt.pause(0.01)
plt.ioff()
plt.show()