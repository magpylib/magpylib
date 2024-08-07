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
Imagine visualizing the invisible forces at play in the magnetic interactions between objects—this is what we achieve using Magpylib. 

### But what is force and torque exactly on the topic of magnetization? 
#### FORCE
In the context of magnetic interactions, the force is a vector quantity that represents the push or pull experienced by a magnetic object due to the presence of another magnetic object. This force can cause the object to move if it is free to do so. The force vector has both a magnitude (how strong the force is) and a direction (the direction in which the force acts).

#### Calculation 

The force 𝐹 on a magnetic dipole 𝑚 in a magnetic field 𝐵 is given by:

$$𝐹 = ∇(𝑚*𝐵)$$

where 
𝑚 is the magnetic moment of the target object. This force is calculated by taking the gradient of the dot product of the magnetic moment and the magnetic field.


<br>

#### TORQUE

Torque, also known as the moment of force, is a measure of the rotational effect produced by the force on a magnetic object. It is a vector quantity that depends on both the magnitude of the force and the distance from the point where the force is applied to the axis of rotation. Torque causes an object to rotate or change its rotational motion. In magnetic interactions, torque is significant when the magnetic moments of the objects interact, causing them to experience rotational forces.

#### Calculation

The torque 𝑇 on a magnetic dipole 𝑚 in a magnetic field 𝐵 is given by:

 $$𝑇 = 𝑚 \times 𝐵$$

where 
𝑚 is the magnetic moment of the target object. This torque is calculated by taking the cross product of the magnetic moment and the magnetic field.

## For the beginning - Step for Step
We will begin with a straightforward example to illustrate how force and torque calculations work.

### Imports & Functional Code
> [!IMPORTANT]
> Imports are necessairy to run the programm with the functions we want to use. 

```{code-cell} ipython3
import numpy as np
import magpylib as magpy
from magpylib_force import getFT
from scipy.spatial.transform import Rotation as R

cube = magpy.magnet.Cuboid(position=(8.,0.,8), dimension=(1.,1.,1.), polarization=(1.,0.,0), orientation=R.from_euler('x', 45, degrees=True))
cube.meshing = (20,20,20)
    
cylinder = magpy.magnet.Cylinder(position = np.array([3.5,0.,6.]), orientation=R.from_euler('y', 70, degrees=True), dimension=(3.5,3.5), polarization=(0.,0.,0.2))

circle = magpy.current.Circle(position = np.array([13.,0.,6.5]), orientation=R.from_euler('y', -55, degrees=True), diameter=5, current = -110000 )

targets = [cube]
sources = [cylinder, circle]

FTs = getFT(sources, targets, anchor=None)
print ("FTs:", FTs)   

```
In this coe block the magnetic objects are initialized by their position, dimension or diameter, polarization and the orientation. Be aware of, that not every type of object has the same amount of parameters required, to be sure about the construction click [here](https://magpylib.readthedocs.io/en/stable/_pages/user_guide/docs/docs_classes.html) to get more information.  

The allocation, which magnet is a target and which a source is important for the calculation of force and torque. Sources let act the force on targets, so only the targets will be calculated.

The getFT function is included in the magpylib-force package and is executed in the background. The values are printed out.



### Display
```{code-cell} ipython3
def display(targets,sources, FTs):

    n_targets = len(targets)
    p = magpy.show(targets + sources, backend='pyvista', return_fig=True,style_legend_show=False)

    FTs = getFT(sources, targets, anchor=None)
    print ("FTs:", FTs)   
    
    for i  in range(n_targets):
        p.add_arrows(cent=targets[i].position, direction=FTs[0,:], mag=1/1500, color='g')
        p.add_arrows(cent=targets[i].position, direction=FTs[1,:], mag=1/1500, color='r')
        

    p.camera.position = (8., -40., 15.)
    p.camera.focal_point = (8,0.,8.)
    return p

p = display(targets, sources, FTs)
p.show()
p.close()
```
This function displays the output and calculates force and torque. The getFT function is included in the magpylib-force package and is executed in the background. The add_arrows command is important for visualizing the invisible force and torque. The lines of code concerning the camera settings are used to set the camera position and the focal point of the camera.

Outside this function and in the end of this code there are three extra lines, which displays the interactive window, where you can see this forces and torques visualized. 




## Advanced

> [!IMPORTANT]
> Don't forget to copy the code from above first! This code is a developed version of the simple example above. The additions are listed below.

In this example, we simulate and bring to life the dynamic magnetic interactions between a stationary cube magnet and two moving sources: a cylindrical magnet and a current-carrying loop. As the sources traverse their predefined paths, we calculate and display the magnetic forces and torques acting on the cube magnet, culminating in an animated GIF that vividly captures these interactions.

### Functional Code
```{code-cell} ipython3
import os
from PIL import Image
import glob

os.makedirs('tmp', exist_ok=True)


for i in range(25):
    p = display(targets,sources,FTs)
    cylinder.move([-0.12,0.,0.])
    circle.move([-0.12,0.,0.])
    p.off_screen = True
    p.screenshot('tmp/{:04d}.png'.format(i))
    p.show()
    p.close()

        
for i in range (25,65) :
    p = display(targets,sources,FTs)
    cylinder.move([0.12,0.,0.])
    circle.move([0.12,0.,0.])
    p.off_screen = True
    p.screenshot('tmp/{:04d}.png'.format(i))
    p.show()

for i in range(65,80):
    p = display(targets,sources,FTs)
    cylinder.move([-0.12,0.,0.])
    circle.move([-0.12,0.,0.])
    p.off_screen = True
    p.screenshot('tmp/{:04d}.png'.format(i))
    p.show()
    p.close() 
```

Now there are three for loops, and in each for loop, the position of the cylinder and the circle is updated according to the values in brackets. The numbers in brackets represent the naming of the pictures, which will be saved. Overlapping numbers are not allowed, as this would result in the files being overwritten.


### make gif function
```python
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

Additionally, the make_gif function has been added. This function creates a GIF from the pictures saved in the 'tmp' folder.

> [!WARNING]
> Incorrect file savings in the folder will result in a non-functional GIF! If you reduce the numbers in the brackets in the for loops, make sure to delete the folder first to prevent errors in GIF creation. 

<img src="../../../_static/videos/example_force3_gif.gif" width=50% align="center">

> [!NOTE]
> The greater the force or torque, the larger the arrow.



