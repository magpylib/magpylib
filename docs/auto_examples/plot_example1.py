"""
Example 1
=========

some information 1 appears onmouse over
"""
import matplotlib.pyplot as plt
import numpy as np

xs = np.linspace(0, 10, 100)
plt.plot(xs, np.sin(xs))
plt.show()
