from magpylib.source.magnet import Box
from numpy import linspace
import matplotlib.pyplot as plt

#create magnet
pm = Box(mag=[0,0,1],dim=[1,1,1])

#create sensor positions
posis = [[x,0,3] for x in linspace(-10,10,100)]

#calcualte fields
Bs = pm.getBsweep(posis)

#plot fields
plt.plot(Bs)