import magpylib as magpy

# create a Collection of three sources
s1 = magpy.magnet.Sphere(magnetization=(0,0,100), diameter=3, position=(3,0,0))
s2 = magpy.magnet.Cuboid(magnetization=(0,0,100), dimension=(2,2,2), position=(-3,0,0))
col = s1 + s2

# generate a spiral path
s1.move([(.2,0,0)]*100, increment=True)
s2.move([(-.2,0,0)]*100, increment=True)
col.rotate_from_angax([5]*100, 'z', anchor=0, increment=True, start=0)

# display
col.display(zoom=-.3, path=10)
