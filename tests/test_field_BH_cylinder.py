import numpy as np
import pickle
from magpylib3._lib.fields.field_BH_cylinder import field_B_cylinder

# GENERATE TEST DATA
# n = 1000
# dim = np.random.rand(n,2)
# mag = (np.random.rand(n,3)-0.5)*1000
# poso= (np.random.rand(n,3)-0.5)*5 
# B = field_B_cylinder(mag,dim,poso,100)
# pickle.dump([dim,mag,poso,B],open('testdata_field_B_cylinder.p','wb'))

def test_field_B_cylinder_general():
    dim,mag,poso,B = pickle.load(open('tests/testdata/testdata_field_B_cylinder.p','rb'))
    B_test = field_B_cylinder(mag,dim,poso,100)
    assert np.allclose(B, B_test), 'Bad general cylinder field'