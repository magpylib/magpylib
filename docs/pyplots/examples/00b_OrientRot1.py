from magpylib.source.magnet import Box
import magpylib as magpy

# fixed magnet parameters
M = [1,0,0] #magnetization
D = [4,2,2] #dimension

# magnets with Euler angle orientations
s1 = Box(mag=M, dim=D, pos = [-4,0, 4])
s2 = Box(mag=M, dim=D, pos = [ 4,0, 4], angle=45, axis=[0,0,1])
s3 = Box(mag=M, dim=D, pos = [-4,0,-4], angle=45, axis=[0,1,0])
s4 = Box(mag=M, dim=D, pos = [ 4,0,-4], angle=45, axis=[1,0,0])

# collection
c = magpy.Collection(s1,s2,s3,s4)

# display collection
magpy.displaySystem(c,direc=True,figsize=(6,6))