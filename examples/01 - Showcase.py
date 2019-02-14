#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% IMPORTS

# imports
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy
import gxPackage_v03 as gx
import plotConfig_v18 as pconfig 

# create magnets

M1 = magpy.magnet.Cube(mag=[100,0,100],dim=[5,5,5])

M2 = magpy.magnet.Cylinder(mag=[100,0,100],dim=[2,6],iterDia=10)

M3 = magpy.magnet.Sphere(mag=[100,0,100],dim=3)
M4 = magpy.magnet.Sphere(mag=[-100,0,-100],dim=2,pos=[6,0,-4])
M5 = magpy.magnet.Sphere(mag=[0,0,100],dim=1,pos=[6,0,-2])

C1a = magpy.current.Circular(curr=100,dim=4)
C1b = magpy.current.Circular(curr=100,dim=4)
C1c = magpy.current.Circular(curr=100,dim=4)
C1d = magpy.current.Circular(curr=100,dim=4)


C2 = magpy.current.Line(curr=100,
        vertices=[[-13,-15,-13],
        [-10, 15,-13],
        [ -7,-15,-13],
        [ -4, 15,-13],
        [ -1,-15,-13],
        [  2, 15,-13]])


M1.setOrientation(45,[0,1,0])
M1.move([6,0,6])

M2.setPosition([-5,0,10])
M2.rotate(45,[0,1,0],CoR=[-5,0,10])

M3.setPosition([5,0,-5])
M3.move([2,0,-2])
for M in [M3,M4,M5]:
    M.rotate(150,[0,1,0],CoR=[0,0,0])

C1a.setPosition([5,0,-5])
C1b.setPosition([4,0,-5])
C1c.setPosition([3,0,-5])
C1d.setPosition([2,0,-5])
for C in [C1a,C1b,C1c,C1d]:
    C.setOrientation(90,[0,1,0])

C2.move([-2,0,5])
C2.rotate(45,[0,1,0],CoR=[-9,0,-8])

pmc = magpy.Collection(M1,M2,M3,M4,M5,C1a,C1b,C1c,C1d,C2)
pmc.move([0,0,1])
pmc.rotate(90,[0,1,0],CoR=[0,0,0])

# displace system geometry
pmc.displaySystem()

magpy.name

# calculate B-fields on a grid
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
#ax1.pcolormesh(X,Y,amp, cmap=plt.cm.jet, vmin=np.amin(0), vmax=np.amax(100))
ax1.contourf( X, Y, amp,np.linspace(0,130,100),cmap=plt.cm.brg) # matplotlib stuff pylint: disable=no-member
ax1.streamplot(X, Y, U, V, color='w', density=3,linewidth=0.8)
#ax1.pcolormesh(X,Y,amp, cmap=plt.cm.RdBu, vmin=np.amin(amp), vmax=np.amax(amp))

ax1.set(
       title = 'H-field of a magnet and current assembly',
       xlabel = 'x-position [mm]',
       ylabel = 'z-position [mm]',
       xlim=[-15,15],
       ylim=[-15,15],
       aspect = 1)




#%% STYLE, PLOT, SAFE
plt.tight_layout()
pconfig.plotstyle([ax1], [ax1])
#plt.savefig('test.pdf', format='pdf', dpi=900)
plt.show()
