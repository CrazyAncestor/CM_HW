import numpy as np
import matplotlib.pyplot as plt

# Define the function and the Fourier series
def f(t):
    return np.abs(np.sin(t))

def fourier_series(t, n_terms):
    a0 = 2 / np.pi
    series = a0  # Start with a0
    for n in range(1, n_terms + 1):
        an = (-4/np.pi)/(4*n**2-1)
        series += an * np.cos(2* n * t)
    return series

# Time array
t = np.linspace(0, np.pi, 1000)

# Original function
original_function = f(t)

# Fourier series approximations
series_1 = fourier_series(t, 1)
series_2 = fourier_series(t, 2)
series_3 = fourier_series(t, 3)

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(t/np.pi, original_function, label='Original Function $|\\sin(t)|$', color='black', linewidth=2)
plt.plot(t/np.pi, series_1, label='Fourier Series (1 term)', linestyle='--')
plt.plot(t/np.pi, series_2, label='Fourier Series (2 terms)', linestyle='--')
plt.plot(t/np.pi, series_3, label='Fourier Series (3 terms)', linestyle='--')
plt.title('Fourier Series Approximation of $|\\sin(t)|$')
plt.xlabel(r't($\pi$)')
plt.ylabel('$|\\sin(t)|$ and Fourier Series')
plt.legend()
plt.grid()
plt.xlim(0,  1)
plt.ylim(0, 1.5)
plt.show()
