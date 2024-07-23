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

This example demonstrates a dynamic simulation of two magnetic objects using magpylib. We simulate the motion and rotation of a cuboid magnet and a spherical magnet under the influence of magnetic forces and torques. The simulation updates the positions and orientations of the magnets over time, visualizing their trajectories, rotational dynamics, and the forces acting on them. A first order semi-implicit Euler method leads to the following equations for the position $\mathbf{s}$, the velocity $\mathbf{v} = \dot{\mathbf{s}}$, the rotation angle $\mathbf{\varphi}$ and the angular velocity $\mathbf{\omega}$ in each time step $\Delta t$:

$$\mathbf{v}(t+\Delta t) = \mathbf{v}(t) + \frac{\Delta t}{m} \mathbf{F}(t)$$

$$\mathbf{s}(t+\Delta t) = \mathbf{s}(t) + \Delta t  \mathbf{v} (t + \Delta t)$$

$$\mathbf{\omega} (t + \Delta t) = \mathbf{ω}(t) + \Delta t \cdot J^{-1} \mathbf{T}(t)$$

$$\mathbf{\varphi} (t + \Delta t) = \mathbf{\varphi}(t) + \Delta t \cdot \mathbf{\omega} (t + \Delta t) $$

$\mathbf{F}$ denotes the force and $\mathbf{T}$ the torque acting on the magnet with mass $m$ and inertia tensor $J$.

In this simulation, we model the interactions between two distinct types of magnets: a cuboid magnet and a spherical magnet. The goal is to observe how these magnets influence each other through magnetic forces and torques, and how these interactions affect their motion and rotation.

   >[!WARNING]
   >
   >The installation of Magpylib-force is required!


```{code-cell} ipython3
import os
import glob
import numpy as np
import pyvista as pv
from PIL import Image
import magpylib as magpy
from magpylib_force import getFT
from scipy.spatial.transform import Rotation as R


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
    


    # simulate movement
    for i in range(n_targets):
        # calculate movement and rotation
        targets[i].velocity = targets[i].velocity + dt/targets[i].mass * FTs[i,0,:]
        targets[i].angular_velocity = targets[i].angular_velocity + dt*targets[i].orientation.apply(np.dot(targets[i].inverse_inertia_tensor, targets[i].orientation.inv().apply(FTs[i,1,:])))
        targets[i].position = targets[i].position + dt * targets[i].velocity
        targets[i].orientation = R.from_rotvec(dt*targets[i].angular_velocity)*targets[i].orientation



if __name__ == "__main__":
# TARGETS: Magpylib target objects that move according to field
    t1 = magpy.magnet.Cuboid(position=(2.,0.,2.), dimension=(1.,1.,1.), polarization=(0.,0.,0.92283), orientation=R.from_euler('y', 0, degrees=True))
    t1.meshing = (20,20,20)
    t1.mass = 2.32
    t1.inverse_inertia_tensor = 2.5862069 * np.eye(3)
    t1.velocity = np.array([99., 0., 0.])
    t1.angular_velocity = np.array([0.,0,0.])

    t2 = magpy.magnet.Sphere(position=(2.,0.,4.001), diameter=1.241, polarization=(0.,0.,0.92583), orientation=R.from_euler('y', 0, degrees=True))
    t2.meshing = 20
    t2.mass = 2.32
    t2.inverse_inertia_tensor = 2.798778 * np.eye(3)
    t2.velocity = np.array([-99., 0., 0.])
    t2.angular_velocity = np.array([0.,0.,0.])

    targets = magpy.Collection(t1, t2)
    dt = 0.0005

    
    for i in range(150):
        apply_movement(targets, dt)
        


    
    #cuboid values after movement
    print('cuboid')
    print('position after', t1.position)
    print('velocity after' , t1.velocity)
    print('angular velocity after' , t1.angular_velocity )
    print()

    #sphere values after movement
    print('sphere')
    print('position after', t2.position)
    print('velocity after' , t2.velocity)
    print('angular velocity after' , t2.angular_velocity )



```
<img src="../../../_static/videos/example_force_gif_bigMagnets.gif" width=50% align="center">

## Features

- calculation of the force and torques between magnet objects
- update of magnets velocities, angular velocities and positions

## Explenation 

The functions `inverse_interia_tensor_cuboid_solid` and `inverse_interia_tensor_sphere_solid` calculate the inverse interia for cuboids und spheres. The important parameters to calculate are the mass of the object and the dimension or rather the diameter. 

`apply_movement` function updates the positions of the magnets based on the calculated force and torques. 


Until now, there were only the definitions. With the if statement `if __name__ == __"main"__:` the functional code starts by running the simulation loop.

First of all the **position**, the **dimension**, the **polarization** and the **orientation** have to be defined. Also the **meshing**, the **mass**, the **velocity** and the **angular_velocity** you can define on your own. The **inverse_interia_tensor** has to be defined by call up the previously created function (difference between sphere and cuboid!). This steps have to be repeated as often as you want magnets. 

Now there is only the for loop left. The number in the pracets is the amount of repetitions you want to have values of. 
In the for loop the function `apply_movement` is callen up. The position will be updated as often as the number specified in the parentheses. Note that this is index numbering and starts at 0.



## Visualization

Keep in mind, that if you want to visualize it like the animation above, you need some extra code lines.

There is an [Example Animations - Custom export Pyvista](https://magpylib.readthedocs.io/en/latest/_pages/user_guide/examples/examples_vis_animations.html#custom-export-pyvista), where you can read up the functionality of visualizing code with Pyvista.

But if you only want to let this example run, without any background information, you can copy the underneath code in the same file, which you have created for the values.

```python

os.makedirs('tmp', exist_ok=True)


def display(targets):

    n_targets = len(targets)
    

    p = magpy.show(targets, backend='pyvista', return_fig=True,style_legend_show=False)

    for i in range(n_targets):
        # sources are all magnets instead of target
        FTs = getFT(targets[:i] + targets[i+1:], [targets[i]], anchor=None)


        force_torque_mag = np.linalg.norm(FTs, axis=-1)
        velocities_mag = np.linalg.norm(targets[i].velocity)
        angular_velocity_mag = np.linalg.norm(targets[i].angular_velocity)

        p.add_arrows(cent=targets[i].position, direction=FTs[0,:], mag=1/force_torque_mag[0], color='g')
        p.add_arrows(cent=targets[i].position, direction=targets[i].velocity, mag=1/velocities_mag, color='b')
        p.add_arrows(cent=targets[i].position, direction=FTs[1,:], mag=1/force_torque_mag[1], color='r')
        p.add_arrows(cent=targets[i].position, direction=targets[i].angular_velocity, mag=1/angular_velocity_mag, color='m')

        p.camera.position = (0., -15., 0.)
        p.camera.focal_point = (2.,0.,3.)
    return p



# #creation of the gif
def make_gif(filename, duration=25, loop=0):
    frames = [Image.open(image) for image in glob.glob(f"tmp/*.png")]
    frames[0].save(
        f"{filename}.gif",
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=loop,
        disposal=2, # remove previous image that becomes visible through transparency
    )

make_gif("test", duration=50)

```

> Also the for loop, has to be extended: 
```python
    for i in range(255):
        apply_movement(targets, dt)
        p = display(targets)
        p.off_screen = True
        p.screenshot('tmp/{:04d}.png'.format(i))
        p.show()
        
        p.close()

```


For your understanding, the first line has to be pasted to the beginning of the document right after the imports. It makes sure, if the appropriate folder is existing. Otherwise the folder will be created. The other code can be pasted after the code block from above. Only the for loop in the end has to be adjusted with the last for loop from the programm above.


