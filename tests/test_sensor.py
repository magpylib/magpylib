from magpylib._lib.classes.sensor import Sensor
from numpy import array

def test_getB():
    from magpylib import source,Collection
    b1 = source.magnet.Box([100,100,100],[1,2,3])
    b2 = source.magnet.Box([100,100,100],[1,2,3])
    sensorPosition = [1,2,3]
    angle = 0
    axis = [0,0,1]
    
    col = Collection(b1,b2)
    sensor = Sensor(sensorPosition,angle,axis)

    result = sensor.getB(b1,b2)
    expected = col.getB(sensorPosition)

    rounding = 4
    for i in range(0,3):
        assert round(result[i],rounding) == round(expected[i],rounding)

def test_getB_rotated():
    from magpylib import source,Collection
    sensorPosition = [1,2,3]
    sensorAngle = 90
    sensorAxis = (0,0,1)

    b1 = source.magnet.Box([100,100,100],[1,2,3],[0,0,0],sensorAngle,sensorAxis)
    b2 = source.magnet.Box([100,100,100],[1,2,3],[0,0,0],sensorAngle,sensorAxis)
    

    col = Collection(b1,b2)
    sensor = Sensor(sensorPosition,sensorAngle,sensorAxis)

    result = sensor.getB(b1,b2)
    expected = col.getB(sensorPosition)

    rounding = 4
    for i in range(0,3):
        assert round(result[i],rounding) == round(expected[i],rounding)