from scipy.spatial.transform import Rotation as R
import magpylib as mag3
import numpy as np
n = 200
minAmp = 5
gap = 3
delta = 3
vol = 110
L = 40
possis = np.linspace(0,L,n)

def get_fields_long(d,dd):
    h = (vol/2)/d**2*4/np.pi
    pm1 = mag3.magnet.Cylinder(mag=( 1000,0,0),dim=(d,h),pos=(-dd/2-h/2,0,0))
    pm2 = mag3.magnet.Cylinder(mag=(-1000,0,0),dim=(d,h),pos=( dd/2+h/2,0,0))
    col = mag3.Collection(pm1,pm2)
    col.rotate_from_angax(90,'y')
    col.move_by((1e-6,0,d/2+gap))
    col.move_by((L,0,0),steps=n-1)
    sens1 = mag3.Sensor(pos_pix=[(-delta/2,0,0),(delta/2,0,0)])
    B = mag3.getB_from_sensor(col, sens1)
    s1p1x, s1p1z = B[:,0,0], B[:,0,2]
    s1p2x, s1p2z = B[:,1,0], B[:,1,2]
    s1odd = -s1p1x+s1p2x
    s1eve = -s1p1z+s1p2z
    s1a = np.sqrt(s1odd**2+s1eve**2)
    s1z = np.arctan2(s1odd,s1eve)
    return s1a,s1z


def get_fields_perp(d,dd):
    h = (vol/2)/d**2*4/np.pi
    pm1 = mag3.magnet.Cylinder(mag=( 1000,0,0),dim=(d,h),pos=(-dd/2-h/2,0,0))
    pm2 = mag3.magnet.Cylinder(mag=(-1000,0,0),dim=(d,h),pos=( dd/2+h/2,0,0))
    col = mag3.Collection(pm1,pm2)
    col.rotate_from_angax(90,'y')
    col.move_by((1e-6,0,d/2+gap))
    col.move_by((L,0,0),steps=n-1)
    sens2 = mag3.Sensor(pos_pix=[(0,0,0),(0,0,-delta)])
    B = mag3.getB_from_sensor(col, sens2)
    s2p1x, s2p1z = B[:,0,0], B[:,0,2]
    s2p2x, s2p2z = B[:,1,0], B[:,1,2]
    s2odd = -s2p1z+s2p2z
    s2eve =  s2p1x-s2p2x
    s2a = np.sqrt(s2odd**2+s2eve**2)
    s2z = np.arctan2(s2odd,s2eve)
    return s2a,s2z

def get_lim_long(variables):
    d,dd = variables
    s1a,s1z = get_fields_long(d,dd)
    for i in range(n-1):
        if s1a[i]<minAmp:
            lim_s1a = possis[i]
            break
        lim_s1a = possis[-1]
    for i in range(n-1):
        if s1z[i+1]<s1z[i]:
            lim_s1z = possis[i]
            break
        lim_s1z = possis[-1]
    return -min(lim_s1a,lim_s1z)

def get_lim_perp(variables):
    d,dd = variables
    s2a,s2z = get_fields_long(d,dd)
    for i in range(n-1):
        if s2a[i]<minAmp:
            lim_s2a = possis[i]
            break
        lim_s2a = possis[-1]
    for i in range(n-1):
        if s2z[i]<0:
            lim_s2z = possis[i]
            break
        lim_s2z = possis[-1]
    return -min(lim_s2a,lim_s2z)


import scipy.optimize as opt
bounds = [(1,10),(1,10)]
result = opt.differential_evolution(
    get_lim_long,
    bounds,
    popsize=15,
    disp=True,
    polish=False,
    workers=1,
    updating='deferred')
print(result)
s1a,s1z = get_fields_long(result.x[0], result.x[1])
result = opt.differential_evolution(
    get_lim_perp,
    bounds,
    popsize=15,
    disp=True,
    polish=False,
    workers=1,
    updating='deferred')
print(result)
s2a,s2z = get_fields_perp(result.x[0], result.x[1])

import matplotlib.pyplot as plt
plt.plot(possis,s1z)
plt.plot(possis,s2z)

plt.twinx()
plt.plot(possis,s1a,ls=':')
plt.plot(possis,s2a,ls=':')

plt.show()