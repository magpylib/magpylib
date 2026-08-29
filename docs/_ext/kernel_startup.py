"""Register the docs renderers in every notebook kernel.

``conf.py`` copies this into an ipython profile it points the kernels at, so it
runs before a page's first cell but *inside* the shell. Importing it earlier
than that - from ``sitecustomize``, say - is measurably wrong: matplotlib then
resolves its backend before ipython can offer the inline one, picks the
platform's gui backend instead, and the first ``plt.show()`` blocks forever.
"""

import magpydocs  # noqa: F401
