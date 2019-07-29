from magpylib.source.magnet import Facet
import numpy as np
import pytest
def test_init():
    # Check if constructor is working.
    magnetization = [1,2,3]
    vertices = [ [0,0,0],
                 [1,2,3],
                 [4,5,6] ] 
    position = [0,0,0]

    f = Facet(magnetization,vertices,position)
    for i in range(0,3):
        for j in range(0,3):
            assert all(np.isclose(vertices[i][j],f.vertices[i][j]) for i in range(0,3))

def test_init_fail():
    # Check if input protection is working.
    magnetization = [1,2,3]
    vertices = [ [0,0,0],
                 [1,2,3],
                 [4,5,"6"] ] 
    position = [0,0,0]

    with pytest.raises(AssertionError):
        f = Facet(magnetization,vertices,position)
        for i in range(0,3):
            for j in range(0,3):
                assert all(np.isclose(vertices[i][j],f.vertices[i][j]) for i in range(0,3))

def test_init_repeating_vertix_fail():
    # Check if input protection is working.
    magnetization = [1,2,3]
    vertices = [ [0,0,0],
                 [1,2,3],
                 [0,0,0] ] 
    position = [0,0,0]

    with pytest.raises(AssertionError):
        f = Facet(magnetization,vertices,position)
        for i in range(0,3):
            for j in range(0,3):
                assert all(np.isclose(vertices[i][j],f.vertices[i][j]) for i in range(0,3))

def test_display():
    # Visually check if Facet is drawn and rotated. 
    # Uncomment plt.show() and rerun tests.
    from magpylib import Collection
    from matplotlib import pyplot as plt
    from magpylib import math
    magnetization = [1,2,3]
    vertices = [ [0,0,0],
                 [1,2,3],
                 [4,5,6] ] 
    position = [0,0,0]

    f = Facet(magnetization,vertices,position)


    rotatedVec = []
    for vec in vertices:
        rotatedVec.append(math.rotatePosition(vec,90,(0,0,1)))
    
    rotatedVec = list(rotatedVec)
    for i in range(0,3):
        vertices[i].append(str(vertices[i]))
        rotatedVec[i] = list(rotatedVec[i])
        rotatedVec[i].append(str(rotatedVec[i]))


    f.rotate(90,(0,0,1))
    Collection(f).displaySystem(suppress=True, markers=vertices + rotatedVec)
    #plt.show()