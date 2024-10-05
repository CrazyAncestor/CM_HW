import numpy as np
import matplotlib.pyplot as plt

# Constants
mass = 1.0  # Mass of the particle
dt = 0.01  # Time step
num_steps = 10000  # Number of simulation steps
a = 1
k = 1.

# Initial conditions
r0_values = [0.75, 0.95, 1.2]  # Different initial conditions
v0_values = [np.sqrt(k/mass/r0) * np.exp(-r0/2/a) for r0 in r0_values]

# Lists to store polar coordinates for plotting
plt.figure(figsize=(12, 6))

# Loop over each initial condition
for i, r0 in enumerate(r0_values):
    # Initial position and velocity
    position = np.array([r0, 0.0])  # Initial position (x, y)
    velocity = np.array([0.0, v0_values[i]])  # Initial velocity (vx, vy)

    # Lists to store results for the current initial condition
    r_values = []
    theta_values = []
    times = []

    # Time loop for simulation
    for step in range(num_steps):
        t = step * dt
        
        # Calculate the radial and angular positions
        r = np.linalg.norm(position)
        theta = np.arctan2(position[1], position[0])
        
        # Store polar coordinates and time
        r_values.append(r)
        theta_values.append(theta)
        times.append(t)
        
        # Calculate the force
        f = -k * position / r**3 * np.exp(-r/a) if r != 0 else np.array([0.0, 0.0])
        
        # Update acceleration (a = F/m)
        acceleration = f / mass
        
        # Update velocity and position using Euler's method
        velocity += acceleration * dt
        position += velocity * dt

    # Convert to numpy arrays for easier indexing
    r_values = np.array(r_values)
    theta_values = np.array(theta_values)

    # Polar coordinates plot for the current initial condition
    plt.subplot(1, 1, 1, polar=True)
    plt.plot(theta_values, r_values, label=f'r0 = {r0}')

# Adding a bold line for r = 1
theta_line = np.linspace(0, 2 * np.pi, 100)
plt.plot(theta_line, np.ones_like(theta_line), color='red', linewidth=2, label='r = 1', linestyle='--')

# Plot settings
plt.title('Particle Trajectories in Polar Coordinates')
plt.legend()
plt.tight_layout()
plt.show()
