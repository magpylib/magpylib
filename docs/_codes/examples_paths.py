from scipy.spatial.transform import Rotation as R
import magpylib as magpy

mag = (0,0,1000)
dia = 2

# define a path by hand
s1 = magpy.magnet.Sphere(mag, dia, position=(0,0,0))
s1.position = [(x,0,0) for x in [0,2,4,6,8,10]]
s1.orientation = R.from_rotvec([(0,0,0)]*6)

# define a path using the move method
s2 = magpy.magnet.Sphere(mag, dia, position=(0,0,-5))
s2.move([(x,0,0) for x in [0,2,4,6,8,10,12,14]])

# use increments instead of absolute positions
s3 = magpy.magnet.Sphere(mag, dia, position=(0,0,-10))
s3.move([(2,0,0)]*10, increment=True)

# combine different paths using different start positions
s4 = magpy.magnet.Sphere(mag, dia, position=(0,0,-15))
s4.move([(2,0,0)]*10, increment=True, start=1)
s4.move([(0,0,-1)]*10, increment=True, start=5)

# create complex motions by combining move and rotate-paths
s5 = magpy.magnet.Cuboid(mag, (1,1,2), position=(0,0,-20))
s5.move([(.5,0,0)]*26, increment=True, start=1)
s5.rotate_from_angax([14]*26, 'z', anchor=0, start=1, increment=True)

# paths are automatically displayed
magpy.display(s1, s2, s3, s4, s5, show_path=5, zoom=0)
