from magpylib import source, Collection
from matplotlib import pyplot

pyplot.ioff() # Enable interactivity suppression for figure export

cyl = source.magnet.Cylinder( mag=[-1,0,0],
                              dim=[2,1],
                              pos=[0,0,2])

b =  source.magnet.Box(  mag=[1,0,0],
                         dim=[1,1,0.5],
                         pos=[0,0,0])


pivot_position = [0,0,4]
pivot_marker = pivot_position + ["anchor"]

col = Collection(b,cyl)

fig = col.displaySystem(suppress=True,          # Suppress interactive images
                        markers=[pivot_marker], # Show given markers
                        direc=True)             # Show magnetization vectors

fig.suptitle("Cylinder and Box Assembly")
fig.set_size_inches(6, 6)

cyl.rotate(45,(0,1,0),anchor=pivot_position) # Rotate just the Cylinder Object

fig2 = col.displaySystem(suppress=True,         # Suppress interactive images
                         markers=[pivot_marker],# Show given markers
                         direc=True)            # Show magnetization vectors

fig2.suptitle("Cylinder rotated -90° in (0,1,0), pivoting around [0,0,4]")
fig2.set_size_inches(6, 6)

pyplot.show() # Show images
