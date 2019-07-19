from magpylib import source, Sensor, Collection
from numpy import around
##### Define Sensors
sensor0Position = [2,1,-2]
sensor1Position = [-2,-1,-2]
sensor0 = Sensor(pos=sensor0Position,
                angle=90,
                axis=(0,0,1))
sensor1 = Sensor(sensor1Position,0,(0,0,1))

##### Define magnets, Collection system
cyl = source.magnet.Cylinder([1,2,300],[0.2,1.5])
box = source.magnet.Box([1,2,300],[1,1,0.5],[0,0,1])
col = Collection(cyl,box)

# Read from absolute position in Collection system
absoluteReading = [col.getB(sensor0Position), col.getB(sensor1Position)]
print(absoluteReading)
# [-0.34285586 -0.17269852  0.22153783]

# Rotated sensor reading
relativeReading = [sensor0.getB(col),sensor1.getB(col)]
print(relativeReading)
# [-0.17269852  0.34285586  0.22153783]

sensorList = [sensor0, sensor1]
markerText0 = "sensor0:{}".format(around(relativeReading[0],2))
markerText1 = "sensor1:{}".format(around(relativeReading[1],2))
markerList = [ sensor0Position + [markerText0],
               sensor1Position + [markerText1],
               [3,3,3]]

col.displaySystem(sensors=sensorList , markers=markerList, direc=True)