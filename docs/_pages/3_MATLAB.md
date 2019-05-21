# [WIP] Guide - MATLAB Integration

MATLAB is a numerical computing environment proprietary of MathWorks, widely
used in the scientific and engineering industry. 

As of version R2015b, MATLAB
allows you to call libraries made in other programming languages, including
Python, which enables users to run Magpylib from the MATLAB interface.

<div style="text-align:center;">
    <img src="https://www.mathworks.com/content/mathworks/www/en/products/matlab/matlab-and-python/jcr:content/mainParsys/columns_copy/2/image.adapt.full.high.svg/1535462691919.svg">
</div>

&nbsp;

The following guide intends to provide a digest of the [Official MATLAB
documentation](https://www.mathworks.com/help/matlab/call-python-libraries.html)
 with a focus on utilizing this interface for applying magpylib.


- [[WIP] Guide - MATLAB Integration](#wip-guide---matlab-integration)
  - [Linking Python to Matlab](#linking-python-to-matlab)
    - [Linking the user space Interpreter](#linking-the-user-space-interpreter)
    - [Linking the Anaconda Interpreter](#linking-the-anaconda-interpreter)
  - [Calling MagPylib](#calling-magpylib)
    - [Limitations](#limitations)

## Linking Python to Matlab
### Linking the user space Interpreter

Running the following should yield you the information about the user space interpreter:

```matlab
>>> pyversion
```

### Linking the Anaconda Interpreter

 Anaconda provides Python environments with scientific packages.

<details>

<a href=#linking-the-anaconda-interpreter><summary> Click here for Steps </summary></a>

To couple an Anaconda environment with Matlab, do the following in **Anaconda Navigator**:

````eval_rst

**1.** Select your environments tab

**2.** Create a new environment

|install1| 

**3.** Name your environment

**4.** Choose the Python version (3.5 and up)


|install2|

Keep note of the location as this will be necessary.

**5.** Start conda terminal and install magpylib into the environment.

|install3|

**6.** Find your environment location:

|install4|

.. |install1| image:: ../_static/images/matlab_guide/install1.png

.. |install2| image:: ../_static/images/matlab_guide/install2.png

.. |install3| image:: ../_static/images/matlab_guide/install3.png

.. |install4| image:: ../_static/images/matlab_guide/install4.png

**7.** Enter the following snippet **into your MATLAB console** with your environment's Python Interpreter location:

````

```matlab
>>> pyversion C:\Users\Gabriel\AppData\Local\Continuum\anaconda3\envs\magpy\python.exe
```
</details>

---

## Calling MagPylib

The following MATLAB script showcases most functionalities.

```
%%%%%%%%%%%%%%%%%% magpytest.m %%%%%%%%%%%%%%
%% Showcase Python + MATLAB Interoperability.    
%% Define and calculate the field of a 
%% Cuboid magnet inside a Collection.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Import the library
py.importlib.import_module("magpylib")

%% Define Python types for input
vec3 = py.list({1,2,3})
scalar = py.int(90)

%% Define input
mag = vec3
dim = vec3
angle = scalar
sensorPos = vec3

%% Execute Python
% 2 positional and 1 keyword argument in Box
box = py.magpylib.source.magnet.Box(mag,dim,pyargs('angle',angle))
col = py.magpylib.Collection(box)
pythonResult = col.getB(sensorPos)

%% Convert Python Result to MATLAB data format
matlabResult = double(pythonResult) 
```
### Limitations

```eval_rst
MATLAB does not support Tkinter, which disables matplotlib. This means that :meth:`~magpylib.Collection.displaySystem()` will not generate a display and might interrupt the program.
```