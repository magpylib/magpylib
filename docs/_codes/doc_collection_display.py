import numpy as np
import magpylib as magpy

# create a Collection of three sources
s1 = magpy.magnet.Sphere(magnetization=(0,0,100), diameter=3, position=(3,0,0))
s2 = magpy.magnet.Cuboid(magnetization=(0,0,100), dimension=(2,2,2), position=(-3,0,0))
col = s1 + s2

# generate a spiral path
s1.move(np.linspace((.2,0.,0.),(20.,0.,0.),100), start=0)
s2.move(np.linspace((-.2,0.,0.),(-20.,0.,0.),100), start=0)
col.rotate_from_angax(np.linspace(5.,500.,100), 'z', anchor=0, start=0)

# display
magpy.show(*col, zoom=-.3, style_path_frames=10)
