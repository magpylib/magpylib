from magpylib.source.magnet import Box
import magpylib as magpy
from numpy import array
import matplotlib.pyplot as plt

#fixed magnet parameters
M = [0,0,1] #magnetization
D = [2,2,2] #dimension

# Translation of magnets can be realized in several ways
s1 = Box(mag=M, dim=D, pos = [-4,0, 4])

s2 = Box(mag=M, dim=D, pos = [-2,0, 4])
s2.move([0,0,-2])

s3 = Box(mag=M, dim=D, pos = [ 0,0, 4])
s3.move([0,0,-2])
s3.move([0,0,-2])

s4 = Box(mag=M, dim=D, pos = [ 2,0, 4])
s4.setPosition([2,0,-2])

s5 = Box(mag=M, dim=D, pos = [ 4,0, 4])
s5.position = array([4,0,0])

#collection
c = magpy.Collection(s1,s2,s3,s4,s5)

#display collection
fig = magpy.displaySystem(c,suppress=True)
fig.set_size_inches(6, 6)

plt.show(fig)