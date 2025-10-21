import numpy as np

dim = np.array([1, 2, 3])

dim_sort = np.sort(dim)
R1, R2, _ = dim_sort / max(dim_sort)

print(R1, R2)



