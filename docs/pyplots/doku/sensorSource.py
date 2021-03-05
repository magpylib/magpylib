from magpylib import source, Sensor, Collection


##### Single source example
sensorPosition = [5,0,0]
sensor = Sensor(pos=sensorPosition,
                angle=90,
                axis=(0,0,1))

cyl = source.magnet.Cylinder([1,2,300],[0.2,1.5])

# Read field from absolute position in system
absoluteReading = cyl.getB(sensorPosition)
print(absoluteReading)
# [ 0.50438605   1.0087721  297.3683702 ]

# Now, read from sensor and print the relative output
relativeReading = sensor.getB(cyl)
print(relativeReading)
# [ 1.0087721   -0.50438605 297.3683702 ]

