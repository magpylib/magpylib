"""
Example 2
=========

some information 2 appears onmouse over
"""
import matplotlib.pyplot as plt
import numpy as np

xs = np.linspace(0, 10, 100)
plt.plot(xs, np.sin(xs))
plt.plot(xs, np.cos(xs))
plt.show()
