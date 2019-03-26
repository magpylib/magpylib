# Installation Instructions

#### Prequisites:
 - [Git Software](https://git-scm.com/)
 - Python 3.2+
 - An internet connection for automatically installing the dependencies
    - Matplotlib
    - Numpy


## Installing into default Python user environment:

1. Clone Repo
    ```
    git clone https://github.com/ortnermichael/magpylib
    ```
2. Enter the Repo
    ```
    cd magpylib
    ```
3. Install the library into the current environment
    ```
    pip install .
    ```
```eval_rst
4. Run the example script :doc:`_guide/x_examples` using your Python interpreter.
```

## Installing into an Anaconda3 environment for Spyder/Jupyter:
1. Enter a folder, clone Repo
    ```
    cd C:\Users\you\Desktop
    git clone https://github.com/ortnermichael/magpylib
    ```
2. Start your Anaconda3 instance
3. In the interface, go to environments and choose the environment. For this example, we will use the base environment:
   ![](./../_static/install_guide/anaconda.png)
4. Click the arrow, choose Terminal
5. Go to the local git repository
   ```
   (base): cd C:\Users\you\Desktop\magpylib
   ```
6. Run pip
    ```
    (base) pip install .
    ```
Note:

    _If your conda environment doesn't have `pip`, install it with conda:_
    ```
    (base) conda install pip
    (base) pip install .
    ```



### Generating the documentation on Linux:
1. Clone Repo
    ```
    git clone https://github.com/ortnermichael/magpylib
    ```
2. Enter the docs Repo
    ```
    cd magpylib/docs
    ```
3. Run generation with `make`
    ```
    make html
    ```

Documentation is now in `magpylib/docs/_build/html`