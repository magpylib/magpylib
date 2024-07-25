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
**Force:** In the context of magnetic interactions, the force is a vector quantity that represents the push or pull experienced by a magnetic object due to the presence of another magnetic object. This force can cause the object to move if it is free to do so. The force vector has both a magnitude (how strong the force is) and a direction (the direction in which the force acts).

#### Calculation 

The force 𝐹 on a magnetic dipole 𝑚 in a magnetic field 𝐵 is given by:

                         𝐹 = ∇(𝑚*𝐵)

where 
𝑚 is the magnetic moment of the target object. This force is calculated by taking the gradient of the dot product of the magnetic moment and the magnetic field.


<br>



**Torque:** Torque, also known as the moment of force, is a measure of the rotational effect produced by the force on a magnetic object. It is a vector quantity that depends on both the magnitude of the force and the distance from the point where the force is applied to the axis of rotation. Torque causes an object to rotate or change its rotational motion. In magnetic interactions, torque is significant when the magnetic moments of the objects interact, causing them to experience rotational forces.

#### Calculation

The torque 𝑇 on a magnetic dipole 𝑚 in a magnetic field 𝐵 is given by:

                         𝑇 = 𝑚 * 𝐵

where 
𝑚 is the magnetic moment of the target object. This torque is calculated by taking the cross product of the magnetic moment and the magnetic field.

## For the beginning - Step for Step
We will start with a simple example to show the functionality of calculation force and torque.


### Imports 
> [!IMPORTANT]
> Imports are necessairy to run the programm with the functions we want to use. 

```{code-cell} ipython3
import os
import glob
import numpy as np
import pyvista as pv
from PIL import Image
import magpylib as magpy
from magpylib_force import getFT
from scipy.spatial.transform import Rotation as R

```
### Display
```{code-cell} ipython3
def display(targets,sources):

    n_targets = len(targets)
    p = magpy.show(targets + sources, backend='pyvista', return_fig=True,style_legend_show=False)
    FTs = getFT(sources,targets, anchor=None)

    print ("FTs:", FTs) 
    

    force_torque_mag = np.linalg.norm(FTs, axis=-1)

    for i  in range(n_targets):
        p.add_arrows(cent=targets[i].position, direction=FTs[0,:], mag=1/1500, color='g')
        p.add_arrows(cent=targets[i].position, direction=FTs[1,:], mag=1/1500, color='r')
        

    p.camera.position = (8., -40., 15.)
    p.camera.focal_point = (8,0.,8.)
    return p
```
This functions displays the output and calculates force and torque. The `getFT` function is inclucded in the magpylib-force package and will be executed in the background. `add_arrows` is important to visualize the unvisible force and torque. The codelines concerning camera settings are there to set camera positon and the focal point of the camera. 

### Functional Code
```{code-cell} ipython3
if __name__ == "__main__":
# TARGETS: Magpylib target objects that move according to field

    cube = magpy.magnet.Cuboid(position=(8.,0.,8), dimension=(1.,1.,1.), polarization=(1.,0.,0), orientation=R.from_euler('x', 45, degrees=True))
    cube.meshing = (20,20,20)
       
    cylinder = magpy.magnet.Cylinder(position = np.array([3.5,0.,6.]), orientation=R.from_euler('y', 70, degrees=True), dimension=(3.5,3.5), polarization=(0.,0.,0.2))
    
    circle = magpy.current.Circle(position = np.array([13.,0.,6.5]), orientation=R.from_euler('y', -55, degrees=True), diameter=5, current = -110000 )
    
    targets = [cube]
    sources = [cylinder, circle]
    


    for i in range(1):
        p = display(targets,sources)
        os.makedirs('tmp', exist_ok=True)
        p.off_screen = True
        p.screenshot('tmp/{:04d}.png'.format(i))
        p.show()
```
In this if statement the magnetic objects are initialized by their position, dimension or diameter, polarization and the orientation. Be aware of, that not every type of object has the same amount of parameters required, to be sure about the construction click [here](https://magpylib.readthedocs.io/en/stable/_pages/user_guide/docs/docs_classes.html) to get more information.  

The allocation, which magnet is a target and which a source is important for the calculation of force and torque. Sources let act the force on targets, so only the targets will be calculated.

For the for loop it's good to know that the number in brackets is the amount of passes of the loop (for the animation above you can see there are much more steps to create a video). The command `os.makedirs('tmp', exist_ok = true)`checks if there is a folder named 'tmp' in the same folder as the python file is. If no, the script create this folder automatically. If it exists, the files will be overwritten because of the same naming as you can see two lines underneath in brackets : `('tmp/{:04d}.png.format(i))`. The `{:04d}` is responsible for a clean naming: the files will be saved as number.png and the max is 9999.



## Advanced

In this example, we simulate and bring to life the dynamic magnetic interactions between a stationary cube magnet and two moving sources: a cylindrical magnet and a current-carrying loop. As the sources traverse their predefined paths, we calculate and display the magnetic forces and torques acting on the cube magnet, culminating in an animated GIF that vividly captures these interactions.

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
    


    for i in range(25):
        p = display(targets,sources)
        cylinder.move([-0.12,0.,0.])
        circle.move([-0.12,0.,0.])
        p.off_screen = True
        p.screenshot('tmp/{:04d}.png'.format(i))
        p.show()
        p.close()

            
    for i in range (25,65) :
        p = display(targets,sources)
        cylinder.move([0.12,0.,0.])
        circle.move([0.12,0.,0.])
        p.off_screen = True
        p.screenshot('tmp/{:04d}.png'.format(i))
        p.show()

    for i in range(65,80):
        p = display(targets,sources)
        cylinder.move([-0.12,0.,0.])
        circle.move([-0.12,0.,0.])
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



```
<img src="../../../_static/videos/example_force3_gif.gif" width=50% align="center">

> [!NOTE]
> The greater the force or torque, the larger the arrow.



## Code Explenation

This is just a developed code from the simple example above. Underneath there are listed the changes or rather the adds. 

```python
for i in range(25):
        p = display(targets,sources)
        cylinder.move([-0.12,0.,0.])
        circle.move([-0.12,0.,0.])
        p.off_screen = True
        p.screenshot('tmp/{:04d}.png'.format(i))
        p.show()
        p.close()

            
    for i in range (25,65) :
        p = display(targets,sources)
        cylinder.move([0.12,0.,0.])
        circle.move([0.12,0.,0.])
        p.off_screen = True
        p.screenshot('tmp/{:04d}.png'.format(i))
        p.show()

    for i in range(65,80):
        p = display(targets,sources)
        cylinder.move([-0.12,0.,0.])
        circle.move([-0.12,0.,0.])
        p.off_screen = True
        p.screenshot('tmp/{:04d}.png'.format(i))
        p.show()
        p.close() 

```
The for loops got extended. There are now three for loops and in every for loop the position of the cylinder and the circle got updated for the values in brackets. The numbers in brackets stands for the naming of the pictures, which will be saved. No numbers are allowed to overlap, because in this case the files will be overwritten. 



```pthon
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
Also the `make_gif` function got added. This function make a gif out of the pictures, which are saved in the 'tmp' folder. 

> [!WARNING]
> Wrong Savings in the folder lead to a not functional gif! If you reduce the numbers in the brackets in the for loops, make sure that you delete the folder first to prevent wrong gif making. 
