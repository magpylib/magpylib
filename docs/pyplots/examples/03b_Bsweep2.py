from magpylib.source.magnet import Cylinder
import matplotlib.pyplot as plt
import numpy as np
import magpylib.math as math

#system parameters
D,H = 5,4       #magnet dimension
M0 = 1200       #magnet magnetization amplitude
gap = 3         #airgap
d = 5           #distance magnet to center of tilt
thMAX = 15      #maximal joystick tilt angle

#create magnet
s = Cylinder(dim=[D,H],mag=[0,0,M0])

#initial central magnet position 
p0 = [0,0,H/2+gap]

#generate rotation axes
axes = [[np.cos(phi),np.sin(phi),0] for phi in np.linspace(0,2*np.pi,180)]

#generate a type2 INPUT
INPUT = []
for th in np.linspace(1,thMAX,15):
        
    #rotation of the magnet position outwards
    posis = [math.rotatePosition(p0,th,ax,anchor=[0,0,gap+H+d]) for ax in axes]
    
    #generate INPUT
    INPUT += [[[0,0,0],pos,[th,ax]] for pos,ax in zip(posis,axes)]

#calculate all fields in one sweep
Bs = np.array(s.getBsweep(INPUT)).reshape(15,180,3)    

#plot fields
fig = plt.figure()
ax = plt.axes(projection='3d')
cm = plt.get_cmap("jet") #colormap
for i in range(15):
    ax.plot(Bs[i,:,0],Bs[i,:,1],Bs[i,:,2],color=cm(i/15))

#annotate
ax.set(
       xlabel = 'Bx [mT]',
       ylabel = 'By [mT]',
       zlabel = 'Bz [mT]')

#display
plt.show()