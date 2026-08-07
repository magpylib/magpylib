"""The editor backend: the viewer, plus picking, gizmos and an edit history.

Registered as ``"threejs-edit"`` on import, so using it is the same as using
any other backend::

    import magpylib as magpy
    import threejs_edit  # noqa: F401  -- the import registers it

    magpy.show(*objects, backend="threejs-edit")

Nothing else is exposed. The host-side policy an editor needs -- which objects
resize by scaling, how to read a polarization -- lives here rather than in the
caller, because it is knowledge about magpylib objects, not about the scene
being drawn.

To get at those objects the backend resolves ``object_id`` back through
`ctypes`. That token is documented for "an interactive viewer holding the same
objects", which a *host* is and a *backend* is not: `show` hands over a Scene
and nothing else. It is safe here only because magpylib still holds every
object while it is rendering them. See finding 13 in README.md.
"""

from __future__ import annotations

# it reports the URL it is serving on
# ruff: noqa: T201
import ctypes
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from magpylib_threejs import register, render_page

import magpylib as magpy

_HERE = Path(__file__).parent

#: The editor front-end, kept beside this file as real JavaScript.
_EDITOR_JS = (_HERE / "editor.js").read_text()

#: The addon the editor needs. `import` must sit with the module's other
#: imports, which is why the backend takes it separately from the JS body.
_IMPORT = """
import { TransformControls } from 'three/addons/controls/TransformControls.js';
"""


#: Classes whose mesh is *exactly* the base mesh scaled, so a resize can be
#: previewed with `mesh.scale` and magpylib told only the final value. Every
#: entry below was verified by rendering twice and comparing vertices.
#:
#: `constraint` records which scale axes are independent, which follows from
#: how many numbers the parameter has: a Cylinder's `dimension` is
#: (diameter, height), so x and y are locked together and z is free.
#:
#: Excluded, and why:
#:  * `CylinderSegment` -- its angles do not scale with its radii.
#:  * `Sensor` -- a composite. Its pixels sit at real coordinates while the
#:    cross is styled, so `style.size` scales only part of the mesh; scaling
#:    the whole thing would displace the pixels.
#:  * meshes (`Tetrahedron`, `TriangularMesh`) -- vertices are the parameter,
#:    so there is no single scale to drag.
SCALE_COVARIANT = {
    "Cuboid": ("dimension", "free"),
    "Sphere": ("diameter", "uniform"),
    "Cylinder": ("dimension", "xy"),
    # A Dipole has no physical size at all -- its arrow is styled geometry, and
    # `style.size` is one scalar, so the resize is uniform by construction and
    # there is no aspect ratio to preserve. Only holds with sizemode="absolute";
    # under "scaled" the value is a multiplier on the scene-derived autosize.
    "Dipole": ("style.size", "uniform"),
}


def _resolve(obj, path):
    """Read a dotted attribute path, falling back to the library default.

    An unset style property reads as `None` on the object and only takes its
    value from `magpy.defaults.display.style.<family>` when the figure is
    drawn. `merged()` does not help: it resolves set-vs-inherited *within* the
    object's own tree, so `Dipole().style.merged().size` is still `None` while
    the effective size is 1. A host reading style has to consult the default
    itself.
    """
    value = obj
    for part in path.split("."):
        value = getattr(value, part)
    if value is None and path.startswith("style."):
        node = magpy.defaults.display.style
        for part in (type(obj).__name__.lower(), *path.split(".")[1:]):
            node = getattr(node, part)
        value = node
    return value


def shape_of(obj):
    """The scale-covariant shape parameter of `obj`, or None."""
    entry = SCALE_COVARIANT.get(type(obj).__name__)
    if entry is None:
        return None
    attr, constraint = entry
    value = _resolve(obj, attr)
    return {
        "kind": type(obj).__name__,
        "attr": attr,
        "value": value.tolist() if hasattr(value, "tolist") else float(value),
        "constraint": constraint,
    }


def polarization_of(obj):
    """The polarization vector of `obj`, or None if it has none.

    Reported in the object's **local** frame, which is how magpylib stores it.
    The rendered `intensity` is the vertex projected on the *world* vector, so
    the two differ by the object's orientation -- reading the attribute as if
    it were world-space is wrong for anything rotated (measured: 0.44 off for a
    50 degree rotation). That is one reason this edit is not previewed in the
    browser; see `_MAGNETIZATION_NOTE`.
    """
    pol = getattr(obj, "polarization", None)
    if pol is None:
        return None
    return {
        "attr": "polarization",
        "value": [float(component) for component in pol],
    }


def _objects_in(scene):
    """Map ``object_id`` to the object it came from, for one `show` call.

    CPython specific, and only valid while the object is alive -- which it is
    here, because magpylib is holding it in order to draw it.
    """
    found = {}
    for frame in scene.frames:
        for trace in frame.traces:
            oid = trace.get("object_id")
            if oid is not None and oid not in found:
                found[oid] = ctypes.cast(oid, ctypes.py_object).value
    return found


def _serve(html, server):
    """Serve `html` at ``/`` and a heartbeat at ``/alive`` until interrupted.

    A viewer can be served once and forgotten, which is what the plain backend
    does. An editor cannot: the page is only useful while Python is there to
    receive edits, so it stays up and answers a heartbeat, and the page says so
    when the answers stop.
    """
    payload = html.encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            alive = self.path.startswith("/alive")
            body = b'{"alive":true}' if alive else payload
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/json" if alive else "text/html; charset=utf-8",
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            """Quieten the per-request logging."""

    server.RequestHandlerClass = Handler
    url = f"http://127.0.0.1:{server.server_port}/"
    threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    print(f"editing at {url} -- Ctrl+C to stop (the page will say so)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped; the page should now show the backend as gone")
    finally:
        server.server_close()


def show(scene):
    """Render `scene` with the editor layer attached."""
    objects = _objects_in(scene)
    # the port has to be known before the page is built, since the page needs
    # to be told where to send its heartbeat
    server = None if scene.return_fig else ThreadingHTTPServer(("127.0.0.1", 0), None)
    html = render_page(
        scene,
        extra_js=_EDITOR_JS,
        extra_imports=_IMPORT,
        extra_data={
            "UNIT": next(iter(scene.panel(1, 1).labels.values()), "").strip("xyz ()")
            or "m",
            "INFO": {
                "backend": "threejs-edit",
                "magpylib": magpy.__version__,
                "python": ".".join(str(n) for n in sys.version_info[:3]),
                # None when there is no server, i.e. a page saved to a file:
                # the heartbeat is skipped and the page says it is static
                "alive": None
                if server is None
                else f"http://127.0.0.1:{server.server_port}/alive",
            },
            "SHAPES": {
                str(oid): s
                for oid, obj in objects.items()
                if (s := shape_of(obj)) is not None
            },
            "POLARIZATION": {
                str(oid): p
                for oid, obj in objects.items()
                if (p := polarization_of(obj)) is not None
            },
        },
        anchors={oid: obj.position for oid, obj in objects.items()},
    )
    if scene.canvas is not None:  # see README: canvas has no meaning here
        msg = "the threejs backend cannot draw onto an existing canvas"
        raise NotImplementedError(msg)
    if scene.return_fig:
        return html
    return _serve(html, server)


# Importing an editing backend pins the two scalings magpylib derives from the
# scene as a whole, because a viewer that keeps and updates a scene cannot have
# one object's geometry change when another moves (finding 1). Mutating global
# defaults on import is a liberty a shipped backend should not take -- it would
# declare the requirement and warn instead -- but it is what keeps the calling
# script free of anything that is really this backend's business.
magpy.defaults.display.units.length = "m"
magpy.defaults.display.style.sensor.sizemode = "absolute"
magpy.defaults.display.style.dipole.sizemode = "absolute"

register("threejs-edit", show)
