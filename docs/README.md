## About magPyLib Documentation

- Documentation is done with [Sphinx](http://www.sphinx-doc.org/en/master/) v4.2.0.
- Sphinx configuration is [conf.py](./conf.py);
- Docstring format is under the [Numpy Convention](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html).
- Sphinx is configured to read Docstring information from the codebase and convert it into pages utilizing the [autodoc extension](http://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html). 
   
  - These generated files are created at build time and put into a folder called `_autogen`

- Handwritten document pages and guides are kept in the [_pages](./_pages) folder.
  - They use a combination of [Markdown](https://commonmark.org/help/) and [restructuredText](http://docutils.sourceforge.net/docs/ref/rst/directives.html), utilizing [recommonmark](https://github.com/rtfd/recommonmark) as interface.
  - These documents are converted to `.html` files by Sphinx during build time.

- Example code with visual output **generated during build time** are kept in the [pyplots](./pyplots) folder.
  - This utilizes the Matplotlib's [plot directive for restructuredText](https://matplotlib.org/devel/plot_directive.html), placing the code and its graphical output when it is referenced within the directive in the documentation pages.

- Images, web code and videos are kept in the [_static](./_static) folder.

---

### Building Locally

This repository is set up to be easily built on [ReadTheDocs](https://readthedocs.org/) as the codebase is updated. 

##### To build locally on Linux, 
1. Install the dependencies on [requirements.txt](./requirements.txt):
    ```
    pip install requirements.txt
    ```


2. Run [make](http://man7.org/linux/man-pages/man1/make.1.html) to build the documentation:

    ```bash

    make html
    ```

This will create a `_build` folder with an `index.html`, containing the built documentation webpage structure.

---

##### To build locally on Windows,

1. [Install Sphinx](http://www.sphinx-doc.org/en/master/usage/installation.html) 
2. Install the dependencies on [requirements.txt](./requirements.txt):
    ```
    pip install -r requirements.txt
    ```

3. Build the documentation with the `.bat` script:

    ```bash

    make.bat html
    ```

