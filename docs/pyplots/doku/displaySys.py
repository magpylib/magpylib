import magpylib as magpy

# create sources
s1 = magpy.source.magnet.Cylinder( mag = [1,1,0],dim = [4,5], pos = [0,0,5])
s2 = magpy.source.magnet.Box( mag = [0,0,-1],dim = [1,2,3],pos=[0,0,-5])
s3 = magpy.source.current.Circular( curr = 1, dim =10)

#create collection
c = magpy.Collection(s1,s2,s3)

# create sensors
se1 = magpy.Sensor(pos=[10,0,0])
se2 = magpy.Sensor(pos=[10,0,0])
se3 = magpy.Sensor(pos=[10,0,0])
se2.rotate(70,[0,0,1],anchor=[0,0,0])
se3.rotate(140,[0,0,1],anchor=[0,0,0])

#display system
markerPos = [(0,0,0,'origin'),(10,10,10),(-10,-10,-10)]
magpy.displaySystem(c,sensors=[se1,se2,se3],markers=markerPos)