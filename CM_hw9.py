import numpy as np
import matplotlib.pyplot as plt

# Set up the parameters for the first case (omega1 = omega2 = 1)
omega1_case1 = 1.
omega2_case1 = 1.
omega1_case2 = 1.
omega2_case2 = 0.3
radius = 10  # radius of the circular orbit
A = 2
phi_0 = np.pi / 4
time = np.linspace(0, 20 * np.pi, 10000)  # angles from 0 to 20π, 10000 points

# Parametric equations for the first case (omega1 = omega2 = 1)
x_case1 = radius * np.cos(omega1_case1 * time) + A * np.sin(omega2_case1 * time + phi_0)
y_case1 = radius * np.sin(omega1_case1 * time) + A * np.cos(omega2_case1 * time + phi_0)

# Parametric equations for the second case (omega1 = 1, omega2 = 0.3)
x_case2 = radius * np.cos(omega1_case2 * time) + A * np.sin(omega2_case2 * time + phi_0)
y_case2 = radius * np.sin(omega1_case2 * time) + A * np.cos(omega2_case2 * time + phi_0)

# Create the plot
plt.figure(figsize=(6, 6))

# Plot both cases
plt.plot(x_case1, y_case1, label=r'2-dim S.H.O. $\omega_1 = \omega_2 = 1$', lw=2)
plt.plot(x_case2, y_case2, label=r'general oscillator $\omega_1 = 1, \omega_2 = 0.3$', lw=2, linestyle='--')

# Mark the center of the orbit
plt.scatter(0, 0, color='red', label='Center', zorder=5)

# Keep the aspect ratio square
plt.gca().set_aspect('equal', adjustable='box')

# Titles and labels
plt.title("Circular Orbits of 2-dim S.H.O. (b) and the general oscillator with magnetic field (d)")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)

# Move the legend to the lower right corner outside the plot
plt.legend(loc='lower right', bbox_to_anchor=(1.05, 0))

# Show the plot
plt.show()

# Constants (you can modify these values)
l = 1.0          # Angular momentum
m = 1.0          # Mass
Omega = 1.0      # Frequency Omega

# Define the effective potential function
def V_eff(rho, l, m, omega, Omega):
    term1 = l**2 / (2 * m * rho**2)
    term2 = (m / 4) * (omega**2 - 2 * Omega**2) * rho**2
    return term1 + term2

# Generate rho values (avoid division by zero by excluding rho=0)
rho_values = np.linspace(0.1, 10, 400)

# Case 1: omega < sqrt(2) * Omega
omega_case1 = np.sqrt(2) * Omega / 2  # omega < sqrt(2) * Omega
V_values_case1 = V_eff(rho_values, l, m, omega_case1, Omega)

# Case 2: omega > sqrt(2) * Omega
omega_case2 = np.sqrt(2) * Omega * 2  # omega > sqrt(2) * Omega
V_values_case2 = V_eff(rho_values, l, m, omega_case2, Omega)

# Plot the effective potential for both cases
plt.figure(figsize=(10, 6))

# Plot Case 1 (omega < sqrt(2) * Omega)
plt.plot(rho_values, V_values_case1, label=r'$\omega < \sqrt{2} \Omega$', color='b')

# Plot Case 2 (omega > sqrt(2) * Omega)
plt.plot(rho_values, V_values_case2, label=r'$\omega > \sqrt{2} \Omega$', color='r')

# Add labels and title
plt.title(r'Effective Potential $V_{\text{eff}}(\rho)$ for Two Cases')
plt.xlabel(r'$\rho$')
plt.ylabel(r'$V_{\text{eff}}(\rho)$')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
