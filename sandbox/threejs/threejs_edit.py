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
import contextlib
import ctypes
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from magpylib_threejs import register, render_page
from scipy.spatial.transform import Rotation as R

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


#: Constructor arguments worth emitting per class, in order. Anything not
#: listed is either a default or reachable through position/orientation.
_CONSTRUCTOR_ARGS = {
    "Cuboid": ("dimension", "polarization"),
    "Cylinder": ("dimension", "polarization"),
    "CylinderSegment": ("dimension", "polarization"),
    "Sphere": ("diameter", "polarization"),
    "Tetrahedron": ("vertices", "polarization"),
    "Dipole": ("moment",),
    "Circle": ("diameter", "current"),
    "Polyline": ("vertices", "current"),
    "Sensor": ("pixel",),
}


def _literal(value):
    """A numpy value as something that can be pasted into a script."""
    array = np.asarray(value)
    if array.ndim == 0:
        return repr(round(float(array), 6))
    return repr(np.round(array, 6).tolist())


def to_code(objects):
    """The current objects as a runnable magpylib script.

    This is the round-trip the whole exercise is for: edit in the browser, get
    python back. It reads the live objects, so it reflects every edit that has
    been applied -- which is why the edits have to actually reach python rather
    than only being logged.
    """
    lines = ["import magpylib as magpy", ""]
    rotated = [o for o in objects if o.orientation.magnitude() > 1e-12]
    if rotated:
        lines.insert(1, "from scipy.spatial.transform import Rotation as R")

    for index, obj in enumerate(objects, start=1):
        kind = type(obj).__name__
        module = {
            "Sensor": "magpy",
            "Dipole": "magpy.misc",
            "Circle": "magpy.current",
            "Polyline": "magpy.current",
        }.get(kind, "magpy.magnet")
        args = [
            f"{name}={_literal(getattr(obj, name))}"
            for name in _CONSTRUCTOR_ARGS.get(kind, ())
            if getattr(obj, name, None) is not None
        ]
        args.append(f"position={_literal(obj.position)}")
        if obj.orientation.magnitude() > 1e-12:
            args.append(
                f"orientation=R.from_quat({_literal(obj.orientation.as_quat())})"
            )
        name = f"{kind.lower()}{index}"
        joined = ",\n    ".join(args)
        lines.append(f"{name} = {module}.{kind}(\n    {joined},\n)")

    names = ", ".join(
        f"{type(o).__name__.lower()}{i}" for i, o in enumerate(objects, 1)
    )
    lines += ["", f"magpy.show({names})"]
    return "\n".join(lines)


def build_tree(objects_by_id, colors):
    """The Collection hierarchy, as nested nodes for the tree view.

    It cannot come from the payload. Every trace under a Collection carries the
    same `legendgroup` -- that of the *outermost* one -- so a nested collection
    leaves no trace of itself, and three levels arrive looking like one. The
    objects do know: `obj.parent` walks up, so the host rebuilds the tree from
    the objects it already resolved. Another face of finding 13.
    """
    nodes, roots = {}, []

    def node_for(obj):
        oid = id(obj)
        if oid in nodes:
            return nodes[oid]
        node = {
            "id": oid,
            "kind": type(obj).__name__,
            "label": getattr(obj.style, "label", None) or type(obj).__name__,
            "color": colors.get(oid),
            "children": [],
        }
        nodes[oid] = node
        parent = getattr(obj, "parent", None)
        (roots if parent is None else node_for(parent)["children"]).append(node)
        return node

    for obj in objects_by_id.values():
        node_for(obj)
    return roots


def apply_edit(obj, field, value):
    """Apply one ``{field: value}`` message from the browser to `obj`."""
    if field == "quaternion":
        obj.orientation = R.from_quat(value)
    elif "." in field:  # a style path, e.g. style.size
        node = obj
        *parents, leaf = field.split(".")
        for part in parents:
            node = getattr(node, part)
        setattr(node, leaf, value)
    else:
        setattr(obj, field, value)


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


#: A fixed port, so re-running the script serves the *same* URL and an open
#: tab shows the new build on reload. An ephemeral port meant every run needed
#: a fresh tab, and reloading an old one silently showed an old build.
PORT = 8770


def _bind():
    """Bind `PORT`, taking it over from a previous run of this script.

    A fixed port is what lets an open tab reload onto the new build, but it
    also means re-running the script finds its own predecessor holding the
    socket. Rather than fail, ask that one to stop -- it is recognisable by
    answering `/alive` -- and then take the port. Anything else on the port is
    left alone and an ephemeral one is used instead.
    """
    ThreadingHTTPServer.allow_reuse_address = True
    for attempt in range(12):
        try:
            return ThreadingHTTPServer(("127.0.0.1", PORT), None)
        except OSError:
            if attempt == 0:
                try:
                    with urlopen(f"http://127.0.0.1:{PORT}/alive", timeout=0.4) as r:
                        ours = b"alive" in r.read()
                except OSError:
                    ours = False
                if not ours:
                    print(f"port {PORT} is taken by something else; using another")
                    return ThreadingHTTPServer(("127.0.0.1", 0), None)
                print(f"taking port {PORT} from the previous run")
                with contextlib.suppress(OSError):
                    urlopen(f"http://127.0.0.1:{PORT}/shutdown", timeout=0.4).read()
            time.sleep(0.25)
    return ThreadingHTTPServer(("127.0.0.1", 0), None)


def _serve(html, server, objects_by_id):
    """Serve `html` at ``/`` and a heartbeat at ``/alive`` until interrupted.

    A viewer can be served once and forgotten, which is what the plain backend
    does. An editor cannot: the page is only useful while Python is there to
    receive edits, so it stays up and answers a heartbeat, and the page says so
    when the answers stop.
    """
    payload = html.encode()

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, body, kind="application/json"):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/shutdown"):
                self._reply(b'{"stopping":true}')
                threading.Thread(target=server.shutdown, daemon=True).start()
            elif self.path.startswith("/alive"):
                self._reply(b'{"alive":true}')
            elif self.path.startswith("/export"):
                # read the live objects, so this reflects every applied edit
                code = to_code(list(objects_by_id.values()))
                self._reply(code.encode(), "text/plain; charset=utf-8")
            else:
                self._reply(payload, "text/html; charset=utf-8")

        def do_POST(self):
            """Apply one edit to the object it names."""
            size = int(self.headers.get("Content-Length", 0))
            message = json.loads(self.rfile.read(size) or b"{}")
            obj = objects_by_id.get(message.pop("object_id", None))
            if obj is not None:
                for field, value in message.items():
                    apply_edit(obj, field, value)
            self._reply(b'{"ok":true}')

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
    objects_by_id = _objects_in(scene)
    # the port has to be known before the page is built, since the page needs
    # to be told where to send its heartbeat
    server = None if scene.return_fig else _bind()
    colors = {
        trace["object_id"]: trace.get("color")
        for frame in scene.frames
        for trace in frame.traces
        if trace.get("object_id") is not None
    }
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
                "root": None
                if server is None
                else f"http://127.0.0.1:{server.server_port}/",
            },
            "SHAPES": {
                str(oid): s
                for oid, obj in objects_by_id.items()
                if (s := shape_of(obj)) is not None
            },
            "TREE": build_tree(objects_by_id, colors),
            "POLARIZATION": {
                str(oid): p
                for oid, obj in objects_by_id.items()
                if (p := polarization_of(obj)) is not None
            },
        },
        anchors={oid: obj.position for oid, obj in objects_by_id.items()},
    )
    if scene.canvas is not None:  # see README: canvas has no meaning here
        msg = "the threejs backend cannot draw onto an existing canvas"
        raise NotImplementedError(msg)
    if scene.return_fig:
        return html
    return _serve(html, server, objects_by_id)


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
