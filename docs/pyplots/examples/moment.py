"""
Define and display the Electromagnetic Field contour of 
a Dipole moment.
"""

from magpylib import source, Collection
import numpy as np
from matplotlib import pyplot as plt
d = source.moment.Dipole((100,20,300))

pmc = Collection(d)
fig = pmc.displaySystem(direc=True)

fig.suptitle("Source Moment")

xs = np.linspace(-15,15,50)
zs = np.linspace(-15,15,50)
Bs = np.array([[pmc.getB([x,0,z]) for x in xs] for z in zs])

# display fields
fig = plt.figure(figsize=(10, 8),facecolor='w', dpi=80)
AXS = [fig.add_subplot(1,1,i, axisbelow=True) for i in range(1,2)]
ax1 = AXS[0]

X,Y = np.meshgrid(xs,zs)
U,V = Bs[:,:,0], Bs[:,:,2]
amp = np.sqrt(U**2+V**2)
ax1.contourf( X, Y, amp,np.linspace(0,130,100),cmap=plt.cm.brg) # pylint: disable=no-member
ax1.streamplot(X, Y, U, V, color='w', density=3,linewidth=0.8)

ax1.set(
       title = 'B-field of Dipole Moment',
       xlabel = 'x-position [mm]',
       ylabel = 'z-position [mm]',
       xlim=[-15,15],
       ylim=[-15,15],
       aspect = 1)
