import magpylib as magpy
import matplotlib.pyplot as plt

#create sources
s1 = magpy.source.magnet.Cylinder( mag = [1,1,0],dim = [4,5], pos = [0,0,5])
s2 = magpy.source.magnet.Box( mag = [0,0,-1],dim = [1,2,3],pos=[0,0,-5])
s3 = magpy.source.current.Circular( curr = 1, dim =10)

#create collection
c = magpy.Collection(s1,s2,s3)

#display system
markerPos = [(0,0,0,'origin'),(10,10,10),(-10,-10,-10)]
magpy.displaySystem(c,markers=markerPos)