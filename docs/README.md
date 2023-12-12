## About Magpylib Documentation

The Documentation is built with [Sphinx](http://www.sphinx-doc.org/en/main/) v5.3.0 and the configuration file is [conf.py](./conf.py). Files get converted to `.html` files by Sphinx during build time. Images, web code and videos are kept in the [_static](./_static) folder.

### API docs
 The docstring format is under the [Numpy Convention](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html). Sphinx is configured to read Docstring information from the codebase and convert it into pages utilizing the [autodoc extension](http://www.sphinx-doc.org/en/main/usage/extensions/autodoc.html). The generated files are created at build time and put into a folder called `_autogen`

### Handwritten documents
Handwritten pages and guides are kept in the [_pages](./_pages) folder. They are all written in [Markdown](https://www.markdownguide.org/) using [myst-parser](https://github.com/executablebooks/MyST-Parser) as interface. Some documents like in the examples folder are dynamically computed with [myst-nb](https://github.com/executablebooks/myst-nb) as jupyter notebooks. With the help of the [jupytext](https://github.com/mwouts/jupytext) library ands its jupyterlab extension, examples can be written and executed within the jupyterlab ecosystem and saved as markdown file. It is recommended to use the [jupyterlab-myst](https://github.com/executablebooks/jupyterlab-myst) extension to be able to work with the full set of myst markdown flavor within jupyterlab. When editing the docs with vscode, use the [MyST-Markdown](https://marketplace.visualstudio.com/items?itemName=ExecutableBookProject.myst-highlight) extension to visualize the rendered document.

