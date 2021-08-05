import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

# observer positions
poso = [(x,0,2) for x in np.linspace(-3,3,100)]

# getB_dict computation - only observer is a shape (n,x)-vector
B1 = magpy.getB_dict(
    source_type='Cuboid',
    magnetization=(0,0,1000),
    dimension=(1,1,1),
    observer=poso)

plt.plot(B1[:,0], 'r')
plt.plot(B1[:,2], 'r')

# changing dimension with observer position
dim = [(d,d,d) for d in np.linspace(0,2,100)]

# getB_dict computation - observer and dimension are shape (n,x)-vectors
#   -> the magnet increases in size as the observer changes position
B2 = magpy.getB_dict(
    source_type='Cuboid',
    magnetization=(0,0,1000),
    dimension=dim,
    observer=poso)

plt.plot(B2[:,0], 'g')
plt.plot(B2[:,2], 'g')

plt.show()
