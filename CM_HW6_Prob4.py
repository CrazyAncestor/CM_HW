import numpy as np
import matplotlib.pyplot as plt

r0 = 1
alpha = r0
eps = 0.8

def old_orbit(theta):
    return alpha

def new_orbit1(theta):
    return alpha / (1 + eps * np.cos(theta + np.pi / 2))

def new_orbit2(theta):
    return alpha / (1 + eps * np.cos(theta - np.pi / 2))

def plot_xy(theta, orbit, label, color):
    x = orbit * np.cos(theta)
    y = orbit * np.sin(theta)
    plt.plot(x, y, label=label, color=color)
    return color  # Return color for arrow use

theta = np.linspace(0, 2 * np.pi, 1000)
old = old_orbit(theta)
new1 = new_orbit1(theta)
new2 = new_orbit2(theta)

plt.figure(figsize=(8, 8))
old_color = plot_xy(theta, old, r'Old Orbit ($\epsilon = 0$)', 'blue')
new1_color = plot_xy(theta, new1, r'New Orbit, impulse outward ($\epsilon = 0.8$)', 'orange')
new2_color = plot_xy(theta, new2, r'New Orbit, impulse inward ($\epsilon = 0.8$)', 'green')

# Position for arrows (theta = pi/2)
theta_arrow = np.pi / 2
r_old_arrow = old_orbit(theta_arrow)
r_new1_arrow = new_orbit1(theta_arrow)
r_new2_arrow = new_orbit2(theta_arrow)

# Calculate positions for arrows
x_old_arrow = r_old_arrow * np.cos(theta_arrow)
y_old_arrow = r_old_arrow * np.sin(theta_arrow)

x_new1_arrow = r_new1_arrow * np.cos(theta_arrow)
y_new1_arrow = r_new1_arrow * np.sin(theta_arrow)

x_new2_arrow = r_new2_arrow * np.cos(theta_arrow)
y_new2_arrow = r_new2_arrow * np.sin(theta_arrow)

# Add a black dot at theta=0
theta_zero = 0
r_zero = old_orbit(theta_zero)  # or use new1 or new2 if preferred
x_zero = r_zero * np.cos(theta_zero)
y_zero = r_zero * np.sin(theta_zero)

plt.scatter(x_zero, y_zero, color='black', s=100)  # Size of the dot

# Annotate the dot
plt.annotate('t=0 position', xy=(x_zero, y_zero), xytext=(x_zero + 0.1, y_zero + 0.1),
             arrowprops=dict(arrowstyle='->', color='black'), fontsize=12)

# Add arrows to indicate counterclockwise direction at theta = pi/2, pointing left
arrow_length = 0.2  # Length of the arrow (twice as long)
# Old Orbit Arrow
plt.arrow(x_old_arrow, y_old_arrow, -arrow_length, 0, head_width=0.1, head_length=0.1, fc=old_color, ec=old_color)
# New Orbit 1 Arrow
plt.arrow(x_new1_arrow, y_new1_arrow, -arrow_length, 0, head_width=0.1, head_length=0.1, fc=new1_color, ec=new1_color)
# New Orbit 2 Arrow
plt.arrow(x_new2_arrow, y_new2_arrow, -arrow_length, 0, head_width=0.1, head_length=0.1, fc=new2_color, ec=new2_color)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Orbit Plots')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.savefig('orbits_plot_with_colored_arrows.png')  # Save the plot as a PNG file
plt.show()
