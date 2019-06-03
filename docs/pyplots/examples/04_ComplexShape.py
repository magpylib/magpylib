from magpylib.source.magnet import Cylinder
from magpylib import Collection
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

#define figure
fig = plt.figure()

#create collection of 
s1 = Cylinder(mag=[0,0, 1000], dim=[5,5], pos=[0,0,0])
s2 = Cylinder(mag=[0,0,-1000], dim=[3,5], pos=[0,0,0])
c = Collection(s1,s2)

#calculate images of animation
ims = []
for y in np.linspace(-4,4,50):

    #create grid positions
    N = 50
    xs = np.linspace(-6,6,N)
    zs = np.linspace(-5,5,N)
    posis = [[x,y,z] for z in zs for x in xs]
    
    #calculate fields
    Bs = c.getBsweep(posis)
    Bs = np.array(Bs).reshape([N,N,3])
    Bamp = np.linalg.norm(Bs,axis=2)
    
    #add image to list
    ims.append((plt.pcolor(xs,zs,Bamp,vmax=1000,vmin=0,cmap="jet"),))

#create animation
im_ani = animation.ArtistAnimation(fig, ims, interval=300, repeat_delay=2000,
                                   blit=True)
#display
plt.show()