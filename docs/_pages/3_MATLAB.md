# Guide - MATLAB Integration

```eval_rst

.. note::

   MATLAB does not support Tkinter, which disables matplotlib. This means that :meth:`~magpylib.Collection.displaySystem()` will not generate a display and might interrupt the program.

```

## Setting Python Interpreter

As of version R2015b, MATLAB allows you to call libraries from other 
programming languages, including Python, which enables users to run 
magpylib from the MATLAB interface. The following guide intends to 
provide a digest of the [Official MATLAB documentation](https://www.mathworks.com/help/matlab/call-python-libraries.html) 
with a focus on utilizing this interface with magpylib.

<div style="text-align:center;">
    <img src="https://www.mathworks.com/content/mathworks/www/en/products/matlab/matlab-and-python/jcr:content/mainParsys/columns_copy/2/image.adapt.full.high.svg/1535462691919.svg">
</div>

&nbsp;

Running the following line in the MATLAB console tells you which 
Python environment (user space interpreter) is connected to your MATLAB interface.

```matlab
>>> pyversion
```

If magpylib is already installed in this environment you can directly 
call it, as shown in the [example](#example:-Calling-magpylib-from-MATLAB) below.
If not please follow the [installation guide](1_how2install.md) and install magpylib.

If you choose to install magpylib in a different environment than the one that is
currently connected to your MATLAB interpreter, use the following command 
in the MATLAB console to connect the new environment instead (choose correct 
path pointing at your Python interpreter).

```matlab
>>> pyversion C:\Users\...\AppData\Local\Continuum\anaconda3\envs\magpy\python.exe
```

## Example: Calling magpylib from MATLAB

The following MATLAB script showcases most functionalities.

```matlab
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