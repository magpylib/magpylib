from magpylib._lib.classes.sensor import Sensor
from numpy import array, around


def test_getB():
    errMsg = "Unexpected result for Sensor getB"
    from magpylib import source, Collection
    b1 = source.magnet.Box([100, 100, 100], [1, 2, 3])
    b2 = source.magnet.Box([100, 100, 100], [1, 2, 3])
    sensorPosition = [1, 2, 3]
    angle = 0
    axis = [0, 0, 1]

    col = Collection(b1, b2)
    sensor = Sensor(sensorPosition, angle, axis)

    result = sensor.getB(b1, b2)
    expected = col.getB(sensorPosition)

    rounding = 4
    for i in range(0, 3):
        assert round(result[i], rounding) == round(
            expected[i], rounding), errMsg


def test_getB_col():
    errMsg = "Unexpected result for Sensor getB"
    from magpylib import source, Collection
    sensorPosition = [1, 2, 3]
    sensorAngle = 0
    sensorAxis = (0, 0, 1)
    boxMagnetization = [100, 100, 100]
    boxDimensions = [1, 2, 3]
    boxPos = [0, 0, 0]
    b1 = source.magnet.Box(boxMagnetization, boxDimensions,
                           boxPos, sensorAngle, sensorAxis)
    b2 = source.magnet.Box(boxMagnetization, boxDimensions,
                           boxPos, sensorAngle, sensorAxis)

    col = Collection(b1, b2)
    sensor = Sensor(sensorPosition, sensorAngle, sensorAxis)

    result = sensor.getB(col)
    expected = col.getB(sensorPosition)

    rounding = 4
    for i in range(0, 3):
        assert round(result[i], rounding) == round(
            expected[i], rounding), errMsg


def test_getB_rotated_XYZ():
    # Rotate the sensor in Y and X for a Z oriented magnetization vector
    from magpylib import source
    errMsg = "Unexpected result for Sensor getB"
    # Definitions
    boxMagnetization = [0, 0, 126]
    boxDimensions = [0.5, 1, 1]
    boxPos = [1, 1, 1]
    expected = [array([0., 0., 100.19107165]),
                array([1.00191072e+02, 6.85197764e-14, 2.13162821e-14]),
                array([5.68434189e-14, -1.00191072e+02, -1.42108547e-14])]
    results = []

    # Run
    s = Sensor(pos=boxPos)

    box = source.magnet.Box(boxMagnetization,
                            boxDimensions,
                            pos=boxPos)

    s.rotate(180, [0, 0, 1])  # 180 in Z
    results.append(s.getB(box))  # Check unchanged
    s.rotate(90, [0, 1, 0])  # 90 in Y
    results.append(s.getB(box))  # Check change
    s.rotate(90, [1, 0, 0])  # 90 in X
    results.append(s.getB(box))  # Check change

    rounding = 4
    for j in range(0, len(expected)):
        for i in range(0, 3):
            assert around(results[j][i], rounding) == around(
                expected[j][i], rounding), errMsg
