import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tool

# Parameters (now passed as input)
c1 = 15.6
c2 = 1.
m0 = -8./7.
m1 = -5./7.

# Define the function g(x)
def g(x):
    return m1 * x + (m0 - m1) / 2. * (np.abs(x + 1) - np.abs(x - 1))

# Define the right-hand side functions for the differential equations
def f1(x, y, z, t):
    return c1 * (y - x - g(x))

def f2(x, y, z, t):
    return c2 * (x - y + z)

def f3(x, y, z, t, c3):  # Pass c3 as a parameter
    return -c3 * y

# RK4 method definition
def rk4_step(f1, f2, f3, state, t, dt, c3):
    x, y, z = state
    
    # Compute the four intermediate steps
    k1x = dt * f1(x, y, z, t)
    k1y = dt * f2(x, y, z, t)
    k1z = dt * f3(x, y, z, t, c3)

    k2x = dt * f1(x + 0.5 * k1x, y + 0.5 * k1y, z + 0.5 * k1z, t + 0.5 * dt)
    k2y = dt * f2(x + 0.5 * k1x, y + 0.5 * k1y, z + 0.5 * k1z, t + 0.5 * dt)
    k2z = dt * f3(x + 0.5 * k1x, y + 0.5 * k1y, z + 0.5 * k1z, t + 0.5 * dt, c3)

    k3x = dt * f1(x + 0.5 * k2x, y + 0.5 * k2y, z + 0.5 * k2z, t + 0.5 * dt)
    k3y = dt * f2(x + 0.5 * k2x, y + 0.5 * k2y, z + 0.5 * k2z, t + 0.5 * dt)
    k3z = dt * f3(x + 0.5 * k2x, y + 0.5 * k2y, z + 0.5 * k2z, t + 0.5 * dt, c3)

    k4x = dt * f1(x + k3x, y + k3y, z + k3z, t + dt)
    k4y = dt * f2(x + k3x, y + k3y, z + k3z, t + dt)
    k4z = dt * f3(x + k3x, y + k3y, z + k3z, t + dt, c3)

    # Update the state using the RK4 weighted average
    new_x = x + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
    new_y = y + (k1y + 2 * k2y + 2 * k3y + k4y) / 6
    new_z = z + (k1z + 2 * k2z + 2 * k3z + k4z) / 6

    return new_x, new_y, new_z

def coupled_rk4_step(f1, f2, f3, state1, state2, t, dt, c3, c4):
    x1, y1, z1 = state1
    x2, y2, z2 = state2
    
    # Compute the four intermediate steps for the first set (x1, y1, z1)
    k1x1 = dt * (f1(x1, y1, z1, t) + c4 * (x2 - x1))
    k1y1 = dt * (f2(x1, y1, z1, t) + c4 * (y2 - y1))
    k1z1 = dt * (f3(x1, y1, z1, t, c3) + c4 * (z2 - z1))

    k1x2 = dt * (f1(x2, y2, z2, t) + c4 * (x1 - x2))
    k1y2 = dt * (f2(x2, y2, z2, t) + c4 * (y1 - y2))
    k1z2 = dt * (f3(x2, y2, z2, t, c3) + c4 * (z1 - z2))

    # Compute the intermediate steps for the second set (x1, y1, z1)
    k2x1 = dt * (f1(x1 + 0.5 * k1x1, y1 + 0.5 * k1y1, z1 + 0.5 * k1z1, t + 0.5 * dt) + c4 * (x2 - (x1 + 0.5 * k1x1)))
    k2y1 = dt * (f2(x1 + 0.5 * k1x1, y1 + 0.5 * k1y1, z1 + 0.5 * k1z1, t + 0.5 * dt) + c4 * (y2 - (y1 + 0.5 * k1y1)))
    k2z1 = dt * (f3(x1 + 0.5 * k1x1, y1 + 0.5 * k1y1, z1 + 0.5 * k1z1, t + 0.5 * dt, c3) + c4 * (z2 - (z1 + 0.5 * k1z1)))

    k2x2 = dt * (f1(x2 + 0.5 * k1x2, y2 + 0.5 * k1y2, z2 + 0.5 * k1z2, t + 0.5 * dt) + c4 * (x1 - (x2 + 0.5 * k1x2)))
    k2y2 = dt * (f2(x2 + 0.5 * k1x2, y2 + 0.5 * k1y2, z2 + 0.5 * k1z2, t + 0.5 * dt) + c4 * (y1 - (y2 + 0.5 * k1y2)))
    k2z2 = dt * (f3(x2 + 0.5 * k1x2, y2 + 0.5 * k1y2, z2 + 0.5 * k1z2, t + 0.5 * dt, c3) + c4 * (z1 - (z2 + 0.5 * k1z2)))

    # Compute the intermediate steps for the third set (k3)
    k3x1 = dt * (f1(x1 + 0.5 * k2x1, y1 + 0.5 * k2y1, z1 + 0.5 * k2z1, t + 0.5 * dt) + c4 * (x2 - (x1 + 0.5 * k2x1)))
    k3y1 = dt * (f2(x1 + 0.5 * k2x1, y1 + 0.5 * k2y1, z1 + 0.5 * k2z1, t + 0.5 * dt) + c4 * (y2 - (y1 + 0.5 * k2y1)))
    k3z1 = dt * (f3(x1 + 0.5 * k2x1, y1 + 0.5 * k2y1, z1 + 0.5 * k2z1, t + 0.5 * dt, c3) + c4 * (z2 - (z1 + 0.5 * k2z1)))

    k3x2 = dt * (f1(x2 + 0.5 * k2x2, y2 + 0.5 * k2y2, z2 + 0.5 * k2z2, t + 0.5 * dt) + c4 * (x1 - (x2 + 0.5 * k2x2)))
    k3y2 = dt * (f2(x2 + 0.5 * k2x2, y2 + 0.5 * k2y2, z2 + 0.5 * k2z2, t + 0.5 * dt) + c4 * (y1 - (y2 + 0.5 * k2y2)))
    k3z2 = dt * (f3(x2 + 0.5 * k2x2, y2 + 0.5 * k2y2, z2 + 0.5 * k2z2, t + 0.5 * dt, c3) + c4 * (z1 - (z2 + 0.5 * k2z2)))

    # Compute the intermediate steps for the fourth set (k4)
    k4x1 = dt * (f1(x1 + k3x1, y1 + k3y1, z1 + k3z1, t + dt) + c4 * (x2 - (x1 + k3x1)))
    k4y1 = dt * (f2(x1 + k3x1, y1 + k3y1, z1 + k3z1, t + dt) + c4 * (y2 - (y1 + k3y1)))
    k4z1 = dt * (f3(x1 + k3x1, y1 + k3y1, z1 + k3z1, t + dt, c3) + c4 * (z2 - (z1 + k3z1)))

    k4x2 = dt * (f1(x2 + k3x2, y2 + k3y2, z2 + k3z2, t + dt) + c4 * (x1 - (x2 + k3x2)))
    k4y2 = dt * (f2(x2 + k3x2, y2 + k3y2, z2 + k3z2, t + dt) + c4 * (y1 - (y2 + k3y2)))
    k4z2 = dt * (f3(x2 + k3x2, y2 + k3y2, z2 + k3z2, t + dt, c3) + c4 * (z1 - (z2 + k3z2)))

    # Update the state using the RK4 weighted average for both sets
    new_x1 = x1 + (k1x1 + 2 * k2x1 + 2 * k3x1 + k4x1) / 6
    new_y1 = y1 + (k1y1 + 2 * k2y1 + 2 * k3y1 + k4y1) / 6
    new_z1 = z1 + (k1z1 + 2 * k2z1 + 2 * k3z1 + k4z1) / 6

    new_x2 = x2 + (k1x2 + 2 * k2x2 + 2 * k3x2 + k4x2) / 6
    new_y2 = y2 + (k1y2 + 2 * k2y2 + 2 * k3y2 + k4y2) / 6
    new_z2 = z2 + (k1z2 + 2 * k2z2 + 2 * k3z2 + k4z2) / 6

    # Return the updated states
    return (new_x1, new_y1, new_z1), (new_x2, new_y2, new_z2)




# Define the main function
def attractor(x10, y10, z10, c3):
    # Parameters
    c1 = 15.6
    c2 = 1.
    m0 = -8./7.
    m1 = -5./7.

    # Time parameters
    t0 = 0        # Initial time
    t_final = 100   # Final time
    dt = 0.01      # Time step
    steps = int((t_final - t0) / dt)  # Total number of steps

    # Create the figure for the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set the initial state
    state1 = (x10, y10, z10)  # Set the initial state

    # Record the time and solutions
    time = np.linspace(t0, t_final, steps)
    x_vals1, y_vals1, z_vals1 = [], [], []

    # Solve the system using RK4
    for t in time:
        x_vals1.append(state1[0])
        y_vals1.append(state1[1])
        z_vals1.append(state1[2])

        # Update the state using RK4
        state1 = rk4_step(f1, f2, f3, state1, t, dt, c3)


    # Select the last N steps (e.g., last 100 steps)
    N_last_steps = steps // 2
    x_vals1 = x_vals1[-N_last_steps:]
    y_vals1 = y_vals1[-N_last_steps:]
    z_vals1 = z_vals1[-N_last_steps:]

    time = time[-N_last_steps:]

    # Plot the 3D trajectory for this initial condition (using continuous lines)
    ax.plot(x_vals1, y_vals1, z_vals1, label=f'x0={x10:.2f}, y0={y10:.2f}, z0={z10:.2f}')

    # Label the axes
    ax.set_xlabel('X(t)')
    ax.set_ylabel('Y(t)')
    ax.set_zlabel('Z(t)')

    # Set the title
    ax.set_title('Attractors , c3 = '+str(c3))

    # Display the legend
    ax.legend()

    # Show the plot
    plt.show()


def coupled_attractors(x10, y10, z10, x20, y20, z20, c3,c4):
    # Parameters
    c1 = 15.6
    c2 = 1.
    m0 = -8./7.
    m1 = -5./7.

    # Time parameters
    t0 = 0        # Initial time
    t_final = 100   # Final time
    dt = 0.01      # Time step
    steps = int((t_final - t0) / dt)  # Total number of steps

    # Create the figure for the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set the initial state
    state1 = (x10, y10, z10)  # Set the initial state
    state2 = (x20, y20, z20)  # Set the initial state

    # Record the time and solutions
    time = np.linspace(t0, t_final, steps)
    x_vals1, y_vals1, z_vals1 = [], [], []
    x_vals2, y_vals2, z_vals2 = [], [], []

    # Solve the system using RK4
    for t in time:
        x_vals1.append(state1[0])
        y_vals1.append(state1[1])
        z_vals1.append(state1[2])

        x_vals2.append(state2[0])
        y_vals2.append(state2[1])
        z_vals2.append(state2[2])

        # Update the state using RK4
        state1, state2 = coupled_rk4_step(f1, f2, f3, state1, state2, t, dt, c3, c4)

    # Select the last N steps (e.g., last 100 steps)
    N_last_steps = steps // 2
    x_vals1 = x_vals1[-N_last_steps:]
    y_vals1 = y_vals1[-N_last_steps:]
    z_vals1 = z_vals1[-N_last_steps:]

    x_vals2 = x_vals2[-N_last_steps:]
    y_vals2 = y_vals2[-N_last_steps:]
    z_vals2 = z_vals2[-N_last_steps:]
    time = time[-N_last_steps:]

    # Plot the 3D trajectory for this initial condition (using continuous lines)
    ax.plot(x_vals1, y_vals1, z_vals1, label=f'x0={x10:.2f}, y0={y10:.2f}, z0={z10:.2f}')
    ax.plot(x_vals2, y_vals2, z_vals2, label=f'x0={x20:.2f}, y0={y20:.2f}, z0={z20:.2f}')

    # Label the axes
    ax.set_xlabel('X(t)')
    ax.set_ylabel('Y(t)')
    ax.set_zlabel('Z(t)')

    # Set the title
    ax.set_title('Coupled attractors , c = '+str(c4)+ ' , c3 = '+str(c3))

    # Display the legend
    ax.legend()

    # Show the plot
    plt.show()

# Example usage
attractor(0.1, 0.1, -0.1, 35)
#$coupled_attractors(0.1, 0.1, -0.1, 0.02, 0.04, -0.07, 25.58,0.15)
#coupled_attractors(0.1, 0.1, -0.1, 0.02, 0.04, -0.07, 25.58,0.3)
