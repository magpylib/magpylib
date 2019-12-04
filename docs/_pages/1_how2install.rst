.. _installation:

*************************
Installation
*************************

.. warning::
    Magpylib works only with Python 3.6 or later !

Content 
#######

* `Install with pip`_
* `Windows`_
* `Linux`_
* `Download Sites`_


Install with pip
################

The quickest installation on any platform is through pip.

.. code-block:: console
    
    pip install magpylib

If you are unfamiliar with pip, please follow the detailed guides below:



Windows
#######

Anaconda 3 Install
------------------

If you have little experience with Python we recommand using `Anaconda <https://www.anaconda.com>`_.

1. Download & install Anaconda3
2. Start Anaconda Navigator 
3. On the interface, go to `Environments` and choose the environment you wish to install magpylib in. For this example, we will use the base environment: 

    .. image:: ../_static/images/install_guide/anaconda0.png
   
4. Click the arrow, and open the conda terminal 

    .. image:: ../_static/images/install_guide/anaconda1.png

5. Input the following to install from conda-forge:

    .. code-block:: console

        conda install -c conda-forge magpylib 

6. Dont forget to select the proper environment in your IDE.

    .. image:: ../_static/images/install_guide/anaconda2.png


Clean Python 3 Install
----------------------

If you want to have a custom environment without using conda, you may simply install the library with pip. A simple guide for installation and functionality of pip is found `here <https://projects.raspberrypi.org/en/projects/using-pip-on-windows/5>`_



Linux
#######

Recommended: use Anaconda environment. Simply download Anaconda3 and follow installation steps as under Windows.

Terminal Python 3 Install
--------------------------

1. Install Python3.
2. Open your Terminal and install with

    .. code-block:: console

        pip install magpylib



Download Sites
#################

Currently magpylib is hosted at:

* `Conda Cloud <https://anaconda.org/conda-forge/magpylib>`_ 
* `Python Package Index <https://pypi.org/project/magpylib/>`_
* `GitHub repository <https://github.com/magpylib/magpylib>`_