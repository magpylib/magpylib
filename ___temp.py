import numpy as np
import magpylib as mag3
from scipy.spatial.transform import Rotation as R

src1 = mag3.magnet.Box(magnetization=(3,3,3), dimension=(3,3,3))
src2 = mag3.magnet.Cylinder(magnetization=(0,0,100), dimension=(1,2), position=(0,0,5))
src2.rotate_from_angax(angle=[10]*19, axis='x', start=1, anchor=0, increment=True)
src3a = mag3.magnet.Sphere(magnetization=(0,0,100), diameter=3, position=(0,4,0), orientation=R.from_rotvec((0,0,0)))
src3b = mag3.magnet.Sphere(magnetization=(0,0,100), diameter=2, position=(0,7,0), orientation=R.from_rotvec((0,.5,0)))
src3c = mag3.magnet.Sphere(magnetization=(0,0,100), diameter=1, position=(0,9,0), orientation=R.from_rotvec((0,1,0)))
src4 = mag3.current.Circular(current=1, diameter=20, position=(0,0,10))
src5 = mag3.current.Line(current=1, vertices=[(10,0,0),(0,10,0),(-10,0,0),(0,-10,0),(10,0,0)], position=(0,0,-10))
src6 = mag3.misc.Dipole(moment=(0,-100,0), position=(0,-8,0))
ts = np.linspace(-1,1,5)
sens = mag3.Sensor(position=(7,0,0),pixel=[(x,y,0) for x in ts for y in ts])
sens.rotate(R.from_rotvec([(0,.05,0)]*19), anchor=0, increment=True, start=1)

mag3.display(src1, src2, src3a, src3b, src3c, src4, src5, src6, sens,
    show_path=5, show_direction=True, size_sensors=1, size_dipoles=1, size_direction=1)
