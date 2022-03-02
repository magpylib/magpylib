(matlab)=

# MATLAB Integration

```{note}
MATLAB does not support Tkinter, which disables matplotlib. This means that {meth}`~magpylib.Collection.show()` will not generate a window and might interrupt the program.
```

## Setting Python Interpreter

As of version R2015b, MATLAB allows you to call libraries from other programming languages, including Python, which enables users to run magpylib from the MATLAB interface. The following guide intends to provide a digest of the [Official MATLAB documentation](https://www.mathworks.com/help/matlab/call-python-libraries.html) with a focus on utilizing this interface with magpylib.

Running `>>> pyversion` following line in the MATLAB console tells you which Python environment (user space interpreter) is connected to your MATLAB interface.

If magpylib is already installed in this environment you can directly call it, as shown in the [Example] below. If not please follow the {ref}`installation` instructions and install magpylib.

If you choose to install magpylib in a different environment than the one that is currently connected to your MATLAB interpreter, use the following command in the MATLAB console to connect the new environment instead (choose correct path pointing at your Python interpreter).

```matlab
>>> pyversion C:\Users\...\AppData\Local\Continuum\anaconda3\envs\magpy\python.exe
```

## Example

The following MATLAB 2019 script showcases most functionalities.

```matlab
%%%%%%%%%%%%%%%%%% magpytest.m %%%%%%%%%%%%%%
%% Showcase Python + MATLAB Interoperability.
%% Define and calculate the field of a
%% Cuboid magnet.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Import the magpylib v4 library
py.importlib.import_module("magpylib")

%% Define Python types for input
vec3 = py.list({1,2,3})

%% Define input
mag = py.list({22,33,44})
dim = py.list({2,2,2})
pos = py.list({1,2,3})
observer = py.list({2,3,4})

%% Execute Python
% 2 positional and 1 keyword argument in Cuboid class
src = py.magpylib.magnet.Cuboid(mag, dim, pyargs('position', pos))
pythonResult = src.getB(observer)

%% Convert Python Result to MATLAB data format
matlabResult = double(pythonResult)
```

```{note}
With old versions of Matlab the *double(pythonResult)* type conversion might give an error message.
```
