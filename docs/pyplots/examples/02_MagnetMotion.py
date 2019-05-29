from magpylib.source.magnet import Cylinder
import matplotlib.pyplot as plt
import numpy as np

#system parameters
D,H = 5,4       #magnet dimension
M0 = 1200       #magnet magnetization amplitude
gap = 3         #airgap
d = 5           #distance magnet to center of tilt
thMAX = 15      #maximal joystick tilt angle

#define figure
fig = plt.figure()
ax = plt.axes(projection='3d')
cm = plt.get_cmap("jet") #colormap

#set tilt angle
for th in np.linspace(1,thMAX,30):
    
    #create magnet for joystick in center position
    s = Cylinder(dim=[D,H],mag=[0,0,M0],pos=[0,0,H/2+gap])
    
    #set joystick tilt th
    s.rotate(th,[0,1,0],anchor=[0,0,gap+H+d])
    
    #rotate joystick for fixed tilt
    Bs = np.zeros([181,3]) #store fields here    
    for i in range(181):
        
        #calculate field (sensor at [0,0,0]) and store in Bs
        Bs[i] = s.getB([0,0,0])
        
        #rotate magnet to next position
        s.rotate(2,[0,0,1],anchor=[0,0,0])

    #plot fields
    ax.plot(Bs[:,0],Bs[:,1],Bs[:,2],color=cm(th/15))

#annotate
ax.set(
       xlabel = 'Bx [mT]',
       ylabel = 'By [mT]',
       zlabel = 'Bz [mT]')

#display
plt.show()