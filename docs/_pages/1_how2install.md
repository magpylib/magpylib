# Installation Instructions

- [Installation Instructions](#installation-instructions)
  - [Install with pip](#install-with-pip)
  - [Windows Install](#windows-install)
      - [Anaconda 3 Install](#anaconda-3-install-for-spyder-or-jupyter-notebook)
      - [Clean Python 3 install](#clean-python-3-install)
  - [Linux Install](#linux-install)
      - [Anaconda 3 Install](#anaconda-3-install)
      - [Terminal Python 3 Install](#terminal-python-3-install)
  - [Download Sites](#download-pages)
  
---

## Install with pip

The quickest installation on any platform ist through pip.

```
pip install magpylib
```

If you are unfamiliar with pip please follow the detailed guides below

---

## Windows Install

#### Anaconda 3 Install

If you have little experience with Python we recommand using [Anaconda](https://www.anaconda.com).

<details>

<a href=#anaconda-3-install><summary> Click here for Steps </summary></a>

1. [Download Anaconda][anaconda]
2. Start Anaconda Navigator 
3. On the interface, go to `Environments` and choose the environment you wish to install magpylib in. For this example, we will use the base environment:
   ![](../_static/images/install_guide/anaconda0.png)
4. Click the arrow, and open the conda terminal 
   ![](../_static/images/install_guide/anaconda1.png)
5. Input the following to install from conda-forge:
   ```
   conda install -c conda-forge magpylib 
   ```
6. On the Anaconda interface, in the Home tab, select your environment and Open Spyder/Jupyter 
   ![](../_static/images/install_guide/anaconda2.png)

</details>

&nbsp;
&nbsp;

#### Clean Python 3 Install

If you want to have a custom environment without using conda, you may simply install the library with [pip]

<details>

<a href=#clean-python-3-install><summary> Click here for Steps </summary></a>

1. Install [Python][python3]
2. Open `cmd.exe`
3. Add Python to your path
   - [External Guide on setting up Python + pip](https://projects.raspberrypi.org/en/projects/using-pip-on-windows/5)
4. Install magpylib with the following command:
    ```
    python -m pip install magpylib
    ```
</details>

&nbsp;
&nbsp;

---

## Linux Install

#### Anaconda 3 Install

<details>

<a href="#anaconda-3-install"><summary> Click here for Steps </summary></a>

1. [Download Anaconda][anaconda]
2. Open a terminal window and type `anaconda-navigator`
3. On the interface, go to `Environments` and choose the environment you wish to install magpylib in. For this example, we will use the base environment:
   ![](../_static/images/install_guide/anaconda0.png)
4. Click the arrow, and open the conda terminal 
   ![](../_static/images/install_guide/anaconda1.png)
5. Open the conda terminal and input the following to install from conda-forge:
   ```
   conda install -c conda-forge magpylib
   ```
6. On the Anaconda interface, in the Home tab, select your environment and Open Spyder/Jupyter 
   ![](../_static/images/install_guide/anaconda2.png)

</details>

&nbsp;
&nbsp;

#### Terminal Python 3 Install

<details>

<a href="#terminal-python-3-install"><summary> Click here for Steps </summary></a>

1. Install [Python][python3]
2. Open your Terminal
3. Install magpylib with the following command:
    ```
    pip install magpylib
    ```
</details>

&nbsp;
&nbsp;

---

## Download Sites

Currently magpylib is hosted over at:
- [Conda Cloud][CondaCloud]
- [PyPi][PyPi]

Find the source code at GitHub:
- [GitHub repository][GitHub]



[pip]: https://pip.pypa.io/en/stable/installing/
[anaconda]: https://www.anaconda.com/distribution/
[python3]: https://www.python.org/downloads/
[CondaCloud]:  https://anaconda.org/conda-forge/magpylib
[GitHub]: https://github.com/magpylib/magpylib
[PyPi]:  https://pypi.org/project/magpylib/