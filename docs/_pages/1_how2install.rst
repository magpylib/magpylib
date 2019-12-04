*************************
Installation Instructions
*************************

Content 
#######

* `Install with pip`_
* `Windows`_
* `Linux`_
* `Download Sites`_

.. warning::
    Magpylib works only with Python 3.6 or later !

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

If you have little experience with Python we recommand using `Anaconda`__.



__ _https://www.anaconda.com

1. Download & install Anaconda3
2. Start Anaconda Navigator 
3. On the interface, go to `Environments` and choose the environment you wish to install magpylib in. For this example, we will use the base environment: 
    
    ..image:: ../_static/images/install_guide/anaconda0.png
   
4. Click the arrow, and open the conda terminal 

    ..image:: ../_static/images/install_guide/anaconda1.png

5. Input the following to install from conda-forge:

   ..code-block:: console
    
    conda install -c conda-forge magpylib 

6. Dont forget to select the proper environment in your IDE.

    ..image:: ../_static/images/install_guide/anaconda2.png


Clean Python 3 Install
----------------------

If you want to have a custom environment without using conda, you may simply install the library with pip:

1. Install Python3
2. Open `cmd.exe`
3. Add Python to your path
4. Install magpylib with the following command:

    ..code-block:: console

        python -m pip install magpylib

