# The Magpylib Documentation

::::{grid}
:class-container: sd-px-1
:gutter: 1

:::{grid-item}
:columns: 12
Magpylib is an **open-source Python package** for calculating static **magnetic fields** of magnets, currents and other sources. It uses **explicit expressions**, solutions to the macroscopic magnetostatic problems, implemented in **vectorized** form which makes the computation **extremely fast**. Make use of the open-source Python ecosystem for spectacular visualization.
:::

:::{grid-item}
:columns: 12
<div class="sectiontext">
<h2>How it works</h2>
</div>
:::

:::{grid-item}
:columns: 12
![](_static/images/index_flowchart.png)
:::

:::{grid-item}
:columns: 12
In Magpylib, **sources** (magnets, currents, ...) and **observers** (sensors, position grids, ...) are created as Python objects with position and orientation attributes. These objects can be **grouped** and **moved** around. The system can be **viewed** graphically through various backends. The **magnetic field** is computed in the observer reference frame. Magpylib collects all inputs, and vectorizes the computation for maximal performance.
:::

:::{grid-item}
:columns: 12
<div class="sectiontext">
<h2>Resources</h2>
</div>
:::

:::{grid-item-card}
:columns: 2
:padding: 1
:link: get-started
:link-type: ref
:link-alt: link to Getting Started
:img-top: _static/images/index_icon_getting_started.png
:text-align: center
**Get Started**
:::

:::{grid-item-card}
:columns: 2
:padding: 1
:link: docu-index
:link-type: ref
:link-alt: link to Documentation
:img-top: _static/images/index_icon_docu.png
:text-align: center
**Docs**
:::

:::{grid-item-card}
:columns: 2
:padding: 1
:link: gallery
:link-type: ref
:link-alt: link to Examples
:img-top: _static/images/index_icon_gallery.png
:text-align: center
**Examples**
:::

:::{grid-item-card}
:columns: 2
:padding: 1
:link: contributing
:link-type: ref
:link-alt: link to Contribution Guide
:img-top: _static/images/index_icon_contributing.png
:text-align: center
**Contribute**
:::

:::{grid-item-card}
:columns: 2
:padding: 1
:link: https://github.com/magpylib/magpylib
:link-alt: link to Github
:img-top: _static/images/index_icon_github.png
:text-align: center
**Project**
:::

:::{grid-item-card}
:columns: 2
:padding: 1
:link: https://www.sciencedirect.com/science/article/pii/S2352711020300170
:link-alt: link to Journal
:img-top: _static/images/index_icon_academic.png
:text-align: center
**Paper**
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

_pages/reso_news.md
_pages/reso_get_started.md
_pages/docu/docu_index.md
_pages/gallery/gallery_index.md
_pages/reso_contributing.md
_pages/reso_code_of_conduct.md
_pages/reso_license.md
_pages/reso_changelog.md
_pages/reso_site_notice.md
```
