import magpylib as magpy
from magpylib.source.magnet import Box

#fixed magnet parameters
M = [0,0,1] #magnetization
D = [3,3,3] #dimension

#rotation axis
rax = [-1,1,-1]

#magnets with different orientations
s1 = Box(mag=M, dim=D, pos=[-6,0,4], angle=0, axis=rax)
s2 = Box(mag=M, dim=D, pos=[ 0,0,4], angle=45, axis=rax)
s3 = Box(mag=M, dim=D, pos=[ 6,0,4], angle=90, axis=rax)

#magnets that are rotated differently
s4 =  Box(mag=M, dim=D, pos=[-6,0,-4])
s5 =  Box(mag=M, dim=D, pos=[ 0,0,-4])
s5.rotate(45,rax)
s6 = Box(mag=M, dim=D, pos=[ 6,0,-4])
s6.rotate(90,rax)

#collect all
c = magpy.Collection(s1,s2,s3,s4,s5,s6)

#display collection
magpy.displaySystem(c)