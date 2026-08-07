"""A minimal three.js display backend for Magpylib -- prototype, not shipped.

Deliberately narrow: `mesh3d` traces only (which covers magnets), no
animation, no subplots. The point is to exercise
`magpylib.graphics.backend` from the outside, the way a real third-party
backend would, and find out where the contract is thin. Findings are
written up in README.md next to this file.

`show()` returns a self-contained HTML page as a string.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

import magpylib as magpy

_HERE = Path(__file__).parent

#: The page body, kept beside this file as real JavaScript.
_VIEWER_JS = (_HERE / "viewer.js").read_text()

THREE_VERSION = "0.160.0"

#: Magpylib's `line_width` and `marker_size` are nominal: every backend scales
#: them into its own library's units, and nothing in the contract says what a
#: width of 2 should look like -- `backend_plotly` carries its own
#: `SIZE_FACTORS_TO_PLOTLY`, and the others calibrate elsewhere. These are
#: Plotly's numbers, which transfer directly because `Line2` and
#: `PointsMaterial(sizeAttenuation=False)` also measure in pixels.
SIZE_FACTORS = {"line_width": 2.2, "marker_size": 0.7}


def _hex_to_rgb(color):
    """'#rrggbb' -> (r, g, b) floats in 0..1."""
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255 for i in (0, 2, 4))


LUT_SIZE = 256


def _colorscale_lut(colorscale):
    """Flatten a Magpylib colorscale into a `LUT_SIZE` RGB lookup table.

    The colorscale arrives as ``((stop, '#rrggbb'), ...)``, the Plotly spelling
    Magpylib's trace dialect inherited, and it is *piecewise*: the default
    tricolor scheme holds green to 0.16, grey from 0.26 to 0.74, then red.

    Sampling it per vertex does not work. A Cuboid has eight vertices whose
    intensities are all exactly 0 or 1, so nothing lands on the grey plateau
    and the GPU blends corner colours into a straight green-to-red ramp. What
    has to be interpolated across the face is the *intensity*, with the
    colorscale applied per fragment -- hence a lookup texture, indexed by an
    intensity-valued UV. That is what Plotly's shader does too.
    """
    stops = np.array([s for s, _ in colorscale], dtype=float)
    colors = np.array([_hex_to_rgb(c) for _, c in colorscale], dtype=float)
    samples = np.linspace(0.0, 1.0, LUT_SIZE)
    lut = np.stack(
        [np.interp(samples, stops, colors[:, channel]) for channel in range(3)]
        # RGBA: three.js dropped RGBFormat in r137, so RGBAFormat it is
        + [np.ones_like(samples)],
        axis=1,
    )
    return np.round(lut * 255).astype(int).ravel().tolist()


def _mesh_to_payload(trace):
    """Convert one `mesh3d` trace into what the JS side needs."""
    position = np.stack(
        [np.asarray(trace[axis], dtype=float) for axis in "xyz"], axis=1
    )
    index = np.stack(
        [np.asarray(trace[axis], dtype=int) for axis in ("i", "j", "k")], axis=1
    )

    intensity = trace.get("intensity")
    colorscale = trace.get("colorscale")
    graded = intensity is not None and colorscale is not None
    # per-triangle colours, e.g. a Sensor: body in the object's colour, pixels
    # black, arrow heads red/green/blue. Mixes CSS names with hex, so the
    # strings go over as-is for THREE.Color to parse.
    facecolor = trace.get("facecolor")
    facecolor = None if facecolor is None else [str(c) for c in facecolor]

    uv = None
    if graded:
        intensity = np.clip(np.asarray(intensity, dtype=float), 0, 1)
        uv = np.stack([intensity, np.full(len(position), 0.5)], axis=1)

    return {
        "name": trace.get("name") or "",
        "object_id": trace.get("object_id"),
        "opacity": float(trace.get("opacity", 1) or 1),
        "position": position.ravel().tolist(),
        "index": index.ravel().tolist(),
        "color": trace.get("color"),
        "uv": None if uv is None else uv.ravel().tolist(),
        "lut": _colorscale_lut(colorscale) if graded else None,
        "facecolor": facecolor,
        # legend swatch: the flat colour, else whichever face colour dominates
        "legend_color": (
            trace.get("color")
            or (Counter(facecolor).most_common(1)[0][0] if facecolor else "#2e91e5")
        ),
    }


def _scatter_to_payload(trace):
    """Convert one `scatter3d` trace into what the JS side needs.

    `mode` is a combination, not an enum -- "markers+lines" and
    "markers+text+lines" both occur -- so it is split into tokens and each is
    handled on its own.
    """
    position = np.stack(
        [np.asarray(trace[axis], dtype=float) for axis in "xyz"], axis=1
    )
    modes = set(str(trace.get("mode") or "lines").split("+"))
    return {
        "name": trace.get("name") or "",
        "object_id": trace.get("object_id"),
        "opacity": float(trace.get("opacity", 1) or 1),
        "position": position.ravel().tolist(),
        "lines": "lines" in modes,
        "markers": "markers" in modes,
        # see README: WebGL ignores line width, and neither dash patterns nor
        # marker symbols have a primitive here. Carried so the gap is visible
        # in the payload rather than silently dropped in Python.
        "line_color": trace.get("line_color") or "#2e91e5",
        "line_width": float(trace.get("line_width") or 1) * SIZE_FACTORS["line_width"],
        "line_dash": trace.get("line_dash") or "solid",
        "marker_color": trace.get("marker_color") or "#2e91e5",
        "marker_size": (
            float(trace.get("marker_size") or 3) * SIZE_FACTORS["marker_size"]
        ),
        "marker_symbol": trace.get("marker_symbol") or "o",
    }


_SHELL = """<!doctype html>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  body { margin: 0; overflow: hidden; background: #ffffff; }
  #legend {
    position: absolute; top: 8px; left: 8px; font: 12px sans-serif;
    background: rgba(255,255,255,.85); padding: 6px 8px; border-radius: 4px;
  }
  #legend div { display: flex; align-items: center; gap: 6px; }
  #legend i { width: 10px; height: 10px; border-radius: 2px; }
</style>
<div id="legend"></div>
<script type="importmap">
{"imports": {
  "three": "https://unpkg.com/three@__VERSION__/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@__VERSION__/examples/jsm/"
}}
</script>
<script type="module">
__VIEWER_JS__
</script>
"""


def render_page(scene, extra_js="", anchors=None):
    """Build the HTML page for `scene`, optionally with extra JS appended.

    `anchors` maps ``object_id`` to the object's true origin, in the same
    length unit as the geometry. Magpylib bakes position and orientation into
    the vertices and sends no transform, but `object_id` is documented as valid
    for "an interactive viewer holding the same objects" -- so a host that owns
    them looks the transform up itself and passes it here. Without it the mesh
    falls back to its bounding-box centre, which is wrong for any object whose
    origin is not its centroid.
    """
    panel = scene.panel(1, 1)
    traces = [trace for frame in scene.frames for trace in frame.traces]
    data = {
        "meshes": [_mesh_to_payload(t) for t in traces if t["type"] == "mesh3d"],
        "scatters": [
            _scatter_to_payload(t) for t in traces if t["type"] == "scatter3d"
        ],
        "ranges": panel.ranges.tolist(),
        "labels": panel.labels,
        "anchors": {str(k): list(v) for k, v in (anchors or {}).items()},
    }
    # Substitution order matters: the payload goes in last, so a value
    # inside it can never be mistaken for one of the tokens.
    return (
        _SHELL.replace("__VERSION__", THREE_VERSION)
        .replace("__TITLE__", scene.title or "Magpylib")
        .replace("__VIEWER_JS__", _VIEWER_JS)
        .replace("__EXTRA_JS__", extra_js)
        .replace("__DATA__", json.dumps(data))
    )


def show_threejs(scene):
    """Render a Magpylib `Scene` to a self-contained three.js page."""
    html = render_page(scene)

    if scene.canvas is not None:  # see README: canvas has no meaning here
        msg = "the threejs backend cannot draw onto an existing canvas"
        raise NotImplementedError(msg)

    # `return_fig` is advisory: magpylib passes the backend's return value
    # straight back to the caller either way, so honouring it is on us.
    if scene.return_fig:
        return html
    return _display(html)


def _display(html):
    """Show the page: inline in a notebook, otherwise in a browser tab."""
    try:
        from IPython.display import HTML, display  # noqa: PLC0415
    except ImportError:
        pass
    else:
        if get_ipython_shell() is not None:
            display(HTML(html))
            return

    _open_in_browser(html)


def _open_in_browser(html):
    """Serve `html` once on an ephemeral port and open a browser at it.

    Plotly's browser renderer works this way, and it beats a temp file on two
    counts: nothing is left behind on disk, and the page is reached over
    ``http``, which viewers that refuse ``file://`` will accept.

    The server handles one request and stops, so the process does not outlive
    the page. `timeout` keeps it from blocking forever if no browser appears.
    """
    import webbrowser  # noqa: PLC0415
    from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: PLC0415

    payload = html.encode()

    class OneShot(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            """Quieten the per-request logging."""

    server = HTTPServer(("127.0.0.1", 0), OneShot)
    server.timeout = 30
    webbrowser.open(f"http://127.0.0.1:{server.server_port}")
    server.handle_request()
    server.server_close()


def get_ipython_shell():
    """The active IPython shell, or None outside one."""
    try:
        from IPython import get_ipython  # noqa: PLC0415
    except ImportError:
        return None
    return get_ipython()


def register():
    """Register the backend under the name 'threejs'."""
    return magpy.register_backend(
        "threejs",
        show_threejs,
        # three.js does vertex colours natively, so take the gradient unsliced
        supports_colorgradient=True,
        # every object keeps its own mesh, so it can be addressed and moved
        merge_traces=False,
        # prototype scope
        supports_animation=False,
        supports_subplots=False,
        supports_animation_output=False,
        supports_native_traces=False,
        handles_traces=frozenset({"mesh3d", "scatter3d"}),
        accepts_options=frozenset(),
    )
