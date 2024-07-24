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

(examples-app-force3)=

# Magnetic force and torque
Imagine visualizing the invisible forces at play in the magnetic interactions between objects—this is what we achieve using Magpylib. In this example, we simulate and bring to life the dynamic magnetic interactions between a stationary cube magnet and two moving sources: a cylindrical magnet and a current-carrying loop. As the sources traverse their predefined paths, we calculate and display the magnetic forces and torques acting on the cube magnet, culminating in an animated GIF that vividly captures these interactions.

```{code-cell} ipython3

import os
import glob
import numpy as np
import pyvista as pv
from PIL import Image
import magpylib as magpy
from magpylib_force import getFT
from scipy.spatial.transform import Rotation as R


os.makedirs('tmp', exist_ok=True)

def display(targets,sources):
    n_targets = len(targets)
    p = magpy.show(targets + sources, backend='pyvista', return_fig=True,style_legend_show=False)
    FTs = getFT(sources,targets, anchor=None)
    print ("FTs:", FTs) 
    force_torque_mag = np.linalg.norm(FTs, axis=-1)

    for i  in range(n_targets):
        p.add_arrows(cent=targets[i].position, direction=FTs[0,:], mag=1/1500, color='g')
        p.add_arrows(cent=targets[i].position, direction=FTs[1,:], mag=1/1500, color='r')
        
    p.camera.position = (0., -40., 0.)
    p.camera.focal_point = (8,0.,8.)
    return p


if __name__ == "__main__":
# TARGETS: Magpylib target objects that move according to field

    cube = magpy.magnet.Cuboid(position=(8.,1.,8), dimension=(1.,1.,1.), polarization=(1.,0.,0), orientation=R.from_euler('x', 45, degrees=True))
    cube.meshing = (20,20,20)
       
    cylinder = magpy.magnet.Cylinder(position = np.array([2.,0.,6.]), orientation=R.from_euler('y', 70, degrees=True), dimension=(3.5,3.5), polarization=(0.,0.,0.2))
    
    circle = magpy.current.Circle(position = np.array([13.,0.,6.5]), orientation=R.from_euler('y', -55, degrees=True), diameter=5, current = -110000 )
    
    targets = [cube]
    sources = [cylinder, circle]
    dt = 0.01


    for i in range(25):
        p = display(targets,sources)
        current_positionCylinder = np.array(cylinder.position)
        new_positionCylinder = current_positionCylinder + np.array([-0.12,0.,0.])
        cylinder.move(new_positionCylinder-current_positionCylinder)
            
        current_positionCircle = np.array(circle.position)
        new_positionCircle = current_positionCircle + np.array([-0.12,0.,0.]) 
        circle.move(new_positionCircle-current_positionCircle)
        p.off_screen = True
        p.screenshot('tmp/{:04d}.png'.format(i))
        p.show()
        p.close()

            


    for i in range (25,65) :
        p = display(targets,sources)
        current_positionCylinder = np.array(cylinder.position)
        new_positionCylinder = current_positionCylinder + np.array([0.12,0.,0.]) 
        cylinder.move(new_positionCylinder-current_positionCylinder)
            
        current_positionCircle = np.array(circle.position)
        new_positionCircle = current_positionCircle + np.array([0.12,0.,0.]) 
        circle.move(new_positionCircle-current_positionCircle)
        p.off_screen = True
        p.screenshot('tmp/{:04d}.png'.format(i))
        p.show()

    for i in range(65,80):
        p = display(targets,sources)
        current_positionCylinder = np.array(cylinder.position)
        new_positionCylinder = current_positionCylinder + np.array([-0.12,0.,0.])
        cylinder.move(new_positionCylinder-current_positionCylinder)
            
        current_positionCircle = np.array(circle.position)
        new_positionCircle = current_positionCircle + np.array([-0.12,0.,0.]) 
        circle.move(new_positionCircle-current_positionCircle)
        p.off_screen = True
        p.screenshot('tmp/{:04d}.png'.format(i))
        p.show()
        p.close() 
    



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

def init_plotter():
    """ Init Pyvista plotter with custom scene layout"""
    pl = pv.Plotter(notebook=False, off_screen=True, window_size=[300, 300])
    pl.camera_position = [
        (0, -20, 0),  # Position of the camera
        (0, 0, 0),  # Focal point (what the camera is looking at)
        (0, 0, 1)   # View up direction
    ]
    pl.camera.zoom(0.5)
    pl.set_background("k")  # For better transparency
    return pl
```

## Code Explenation
