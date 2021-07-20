import magpylib as magpy

# define Magpylib sources
s1 = magpy.magnet.Sphere(magnetization=(0,0,100), diameter=1)
s2 = magpy.magnet.Cuboid(magnetization=(0,0,100), dimension=(1,1,1))
s3 = magpy.current.Circular(current=1, diameter=1)
s4 = magpy.magnet.Cylinder(magnetization=(0,0,100), dimension=(1,1))

# create a Collection of three sources
col = magpy.Collection(s1, s2, s3)

# add another source to the collection using __add__
col = col + s4

# access and manipulate individual sources in the collection
s1.position = (2, 0,0)
s2.move((0, 2,0))
# use the get_item
col[2].position = (-2,0,0)
col[3].move((0,-2,0))

# apply operations to all sources in the collection
col.move([(0,0,.1)]*36, increment=True)
col.rotate_from_angax([10]*36, 'z', anchor=0, increment=True, start=0)

# display collection
col.display()

# reset all paths
col.reset_path()

# compute the field at observer position (1,2,3)
B = col.getB((1,2,3))
print(B)
# Output: [0.22591388 0.45193042 0.32545691]

# cycle through collection and print source __repr__
for src in col:
    print(src)
# Output:
# Sphere(id=1551972272448)
# Cuboid(id=1551972271104)
# Cylinder(id=1551972885216)
# Circular(id=1551972929384)
