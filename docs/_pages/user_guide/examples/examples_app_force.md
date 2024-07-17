---
orphan: true
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(examples-app-force)=

# Magnet Force and Torque

This example demonstrates a dynamic simulation of two magnetic objects using magpylib. We simulate the motion and rotation of a cuboid magnet and a spherical magnet under the influence of magnetic forces and torques. The simulation updates the positions and orientations of the magnets over time, visualizing their trajectories, rotational dynamics, and the forces acting on them.

In this simulation, we model the interactions between two distinct types of magnets: a cuboid magnet and a spherical magnet. The goal is to observe how these magnets influence each other through magnetic forces and torques, and how these interactions affect their motion and rotation.


```{code-cell} ipython3
import glob
import numpy as np
from PIL import Image
import magpylib as magpy
from magpylib_force import getFT
from scipy.spatial.transform import Rotation as R



def inverse_inertia_tensor_cuboid_solid(mass, dimensions):
    dimensions_sq = dimensions**2
    inv_tensor = 12/mass * np.array([[1/(dimensions_sq[1]+dimensions_sq[2]),0.,0.], [0.,1/(dimensions_sq[0]+dimensions_sq[2]),0.], [0.,0.,1/(dimensions_sq[0]+dimensions_sq[1])]])
    return inv_tensor

def inverse_inertia_tensor_sphere_solid(mass, diameter):
    return 10 / mass / diameter**2 * np.identity(3)

def apply_movement(targets, dt):
    """defines magnet system that is capable for moving according to force and torque
    Parameters
    ----------
    targets: magpylib collection
        Target magnets where movement is performed on
    dt: float
        finite time step for movement simulation
    """
    n_targets = len(targets)

    # calculate force and torque
    FTs = np.zeros((n_targets, 2, 3))
    for i in range(n_targets):
        # sources are all magnets instead of target
        FTs[i,:,:] = getFT(targets[:i] + targets[i+1:], [targets[i]], anchor=None)
    print("FTs",FTs)
    print()
    print ("velocities", targets[0].velocity,targets[1].velocity)
    print ("angular velocities", targets[0].angular_velocity, targets[1].angular_velocity)


    # simulate movement
    for i in range(n_targets):
        # calculate movement and rotation
        targets[i].velocity = targets[i].velocity + dt/targets[i].mass * FTs[i,0,:]
        targets[i].angular_velocity = targets[i].angular_velocity + dt*targets[i].orientation.apply(np.dot(targets[i].inverse_inertia_tensor, targets[i].orientation.inv().apply(FTs[i,1,:])))
        targets[i].position = targets[i].position + dt * targets[i].velocity
        targets[i].orientation = R.from_rotvec(dt*targets[i].angular_velocity)*targets[i].orientation

        print()
        print('magnet', i)
        print('position after', targets[i].position)
        print('velocity after' , targets[i].velocity)
        print('angular velocity after' , targets[i].angular_velocity )
        print()




if __name__ == "__main__":
# TARGETS: Magpylib target objects that move according to field
    t1 = magpy.magnet.Cuboid(position=(2.,0.,2.), dimension=(1.,1.,1.), polarization=(0.,0.,0.92283), orientation=R.from_euler('y', 0, degrees=True))
    t1.meshing = (20,20,20)
    t1.mass = 2.32
    t1.inverse_inertia_tensor = inverse_inertia_tensor_cuboid_solid(t1.mass, t1.dimension)
    t1.velocity = np.array([99., 0., 0.])
    t1.angular_velocity = np.array([0.,0,0.])

    t2 = magpy.magnet.Sphere(position=(2.,0.,4.001), diameter=1.241, polarization=(0.,0.,0.92583), orientation=R.from_euler('y', 0, degrees=True))
    t2.meshing = 20
    t2.mass = 2.32
    t2.inverse_inertia_tensor = inverse_inertia_tensor_sphere_solid(t2.mass, t2.diameter)
    t2.velocity = np.array([-99., 0., 0.])
    t2.angular_velocity = np.array([0.,0.,0.])

    targets = magpy.Collection(t1, t2)
    dt = 0.0005

    
    for i in range(255):
        apply_movement(targets, dt)




```
<img src="../../../_static/videos/example_gif4.gif" width=50% align="center">

```python
# this cell only shows your code
```
