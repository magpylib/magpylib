import magpylib as magpy
from magpylib.source.magnet import Box

#fixed magnet parameters
M = [0,0,1] #magnetization
D = [2,4,1] #dimension

#magnets rotated with different pivot and anchor points
piv1 = [-7,0,5]
s1 = Box(mag=M, dim=D, pos = [-7,-3,5])

piv2 = [0,0,5]
s2 = Box(mag=M, dim=D, pos = [0,-3,5])
s2.rotate(-30,[0,0,1],anchor=piv2)

piv3 = [7,0,5]
s3 = Box(mag=M, dim=D, pos = [7,-3,5])
s3.rotate(-60,[0,0,1],anchor=piv3)

piv4 = [-7,0,-5]
anch4 = [-7,0,-2]
s4 = Box(mag=M, dim=D, pos = [-7,-3,-5])

piv5 = [0,0,-5]
anch5 = [0,0,-2]
s5 = Box(mag=M, dim=D, pos = [0,-3,-5])
s5.rotate(-45,[0,0,1],anchor=anch5)

piv6 = [7,0,-5]
anch6 = [7,0,-8]
s6 = Box(mag=M, dim=D, pos = [7,-3,-5])
s6.rotate(-45,[0,0,1],anchor=anch6)

#collect all
c = magpy.Collection(s1,s2,s3,s4,s5,s6)

#display collection
fig = c.displaySystem(markers=[piv1+['piv1'],piv2+['piv2'],piv3+['piv3'],
                               piv4+['piv4'],piv5+['piv5'],piv6+['piv6'],
                               anch4+['a4'],anch5+['a5'],anch6+['a6']])