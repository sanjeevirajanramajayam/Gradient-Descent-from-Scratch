import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.widgets import Slider

x = sp.Symbol('x')
y_expr = x**2
y_deriv_expr = sp.diff(y_expr, x)

y_function = sp.lambdify(x, y_expr, 'numpy')
y_derivative = sp.lambdify(x, y_deriv_expr, 'numpy')

learning_rate = 0.01
starting_pos = [8, y_function(8)]

x_range = np.arange(-10, 10, 0.1)
y_range = y_function(x_range)

plt.ion()
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25) 

ax.set_title("Gradient Descent on y = x^2")
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")

ax.plot(x_range, y_range, label="y = x^2")
point, = ax.plot([starting_pos[0]], [starting_pos[1]], 'ro', label='Current Position')
descent_line, = ax.plot([], [], 'm-', lw=2, label='Descent Step')

coord_text = ax.text(starting_pos[0], starting_pos[1]+1, 
                     f"({starting_pos[0]:.2f}, {starting_pos[1]:.2f})",
                     color='black', fontsize=10, ha='center')

ax.legend()

ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'Start x', -10, 10, valinit=8)

def update(val):
    global starting_pos
    starting_pos = [slider.val, y_function(slider.val)]
    point.set_data([starting_pos[0]], [starting_pos[1]])
    descent_line.set_data([], [])  # Clear descent line
    coord_text.set_position((starting_pos[0], starting_pos[1]+1))
    coord_text.set_text(f"({starting_pos[0]:.2f}, {starting_pos[1]:.2f})")
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.pause(0.5)  

while True:
    old_x = starting_pos[0]
    new_x = old_x - learning_rate * y_derivative(old_x)

    if abs(old_x - new_x) < 1e-3:
        break

    new_y = y_function(new_x)

    descent_line.set_data([old_x, new_x], [starting_pos[1], new_y])

    starting_pos = [new_x, new_y]
    point.set_data([starting_pos[0]], [starting_pos[1]])
    coord_text.set_position((starting_pos[0], starting_pos[1]+1))
    coord_text.set_text(f"({starting_pos[0]:.2f}, {starting_pos[1]:.2f})")

    plt.pause(0.05)

plt.ioff()
plt.show()
