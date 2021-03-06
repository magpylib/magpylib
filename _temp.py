import magpylib as mag3
import numpy as np
# getB_level2 selfconsistent test ----------------------
# field tests were performed
# here we only want to test all the magnet and observer functionality

mag = (1,2,3)
dim_box = (1,2,3)
dim_cyl = (1,2)
pm1 = mag3.magnet.Box(mag,dim_box)
pm2 = mag3.magnet.Box(mag,dim_box)
pm3 = mag3.magnet.Cylinder(mag,dim_cyl)
pm4 = mag3.magnet.Cylinder(mag,dim_cyl)
col1 = mag3.Collection([pm1])
col2 = mag3.Collection([pm1,pm2])
col3 = mag3.Collection([pm1,pm2,pm3])
col4 = mag3.Collection([pm1,pm2,pm3,pm4])
pos_obs1 = (1,2,3)
pos_obs2 = [[(1,2,3),(1,2,3)],[(1,2,3),(1,2,3)]]
sens1 = mag3.Sensor(pos=pos_obs1)
sens2 = mag3.Sensor(pos_pix=pos_obs1)
sens3 = mag3.Sensor(pos=(1,2,0),pos_pix=(0,0,3))

fb1 = mag3.getB(pm1,pos_obs1)
fc1 = mag3.getB(pm3,pos_obs1)
fb2 = mag3.getB(pm1,pos_obs2)
fc2 = mag3.getB(pm3,pos_obs2)

fb1_22 = np.array([[fb1,fb1],[fb1,fb1]])
fb2_22 = np.array([[fb2,fb2],[fb2,fb2]])

def test_():
    """Testing getB_level2 functionality:
    different combinations between sorces and sensors
    """
    for poso,fb,fc in zip(
        [pos_obs1,pos_obs2,sens1,sens2,sens3,[sens1,sens2]],
        [fb1,fb2,fb1,fb1,fb1,[fb1,fb1]],
        [fc1,fc2,fc1,fc1,fc1,[fc1,fc1]]
        ):
        src_obs_res = [
            [pm1, poso, fb],
            [pm3, poso, fc],
            [[pm1,pm2], poso, [fb,fb]],
            [[pm1,pm2,pm3], poso, [fb,fb,fc]],
            [col1, poso, fb],
            [col2, poso, 2*fb],
            [col3, poso, 2*fb+fc],
            [col4, poso, 2*fb+2*fc],
            [[pm1,col1], poso, [fb, fb]],
            [[pm1,col1,col2,pm2,col4], poso , [fb,fb,2*fb,fb,2*fb+2*fc]],
            ]

        for sor in src_obs_res:
            sources, observers, result = sor
            result = np.array(result)

            B = mag3.getB(sources, observers)
            assert np.allclose(B, result), 'FAILOR'



test_()

# def test_single():
#     pos_obs = (1,2,3)

#     B = mag3.getB(pm, pos_obs)
#     H = mag3.getH(pm, pos_obs)

#     B2 = pm.getB(pos_obs)
#     assert np.allclose(B,B2), 'err test_single 1'
#     H2 = pm.getH(pos_obs)
#     assert np.allclose(H,H2), 'err test_single 1'

#     sens = mag3.Sensor(pos=pos_obs)
#     B2 = mag3.getB(pm,sens)
#     assert np.allclose(B,B2), 'err test_single 2'
    
#     B2 = sens.getB(pm)
#     assert np.allclose(B,B2), 'err test_single 2'

#     sens2 = mag3.Sensor(pos_pix=pos_obs)
#     B2 = sens.getB(pm)
#     assert np.allclose(B,B2), 'err test_single 3'

#     B2 = mag3.getB(pm,sens2)
#     assert np.allclose(B,B2), 'err test_single 3'

#     sens3 = mag3.Sensor(pos_pix=(1,2,0),pos=(0,0,3))
#     B2 = sens.getB(pm)
#     assert np.allclose(B,B2), 'err test_single 4'


