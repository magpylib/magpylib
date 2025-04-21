"""Generic trace drawing functionalities"""

# pylint: disable=C0302
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=too-many-nested-blocks
# pylint: disable=cyclic-import
# pylint: disable=too-many-positional-arguments

import numbers
import warnings
from collections import Counter
from itertools import chain
from itertools import cycle
from typing import Tuple

import numpy as np

import magpylib as magpy
from magpylib._src.defaults.defaults_classes import default_settings
from magpylib._src.defaults.defaults_utility import ALLOWED_LINESTYLES
from magpylib._src.defaults.defaults_utility import ALLOWED_SYMBOLS
from magpylib._src.defaults.defaults_utility import linearize_dict
from magpylib._src.display.traces_utility import draw_arrowed_line
from magpylib._src.display.traces_utility import get_legend_label
from magpylib._src.display.traces_utility import get_objects_props_by_row_col
from magpylib._src.display.traces_utility import get_rot_pos_from_path
from magpylib._src.display.traces_utility import get_scene_ranges
from magpylib._src.display.traces_utility import getColorscale
from magpylib._src.display.traces_utility import getIntensity
from magpylib._src.display.traces_utility import group_traces
from magpylib._src.display.traces_utility import place_and_orient_model3d
from magpylib._src.display.traces_utility import rescale_traces
from magpylib._src.display.traces_utility import slice_mesh_from_colorscale
from magpylib._src.style import DefaultMarkers
from magpylib._src.utility import format_obj_input
from magpylib._src.utility import get_unit_factor
from magpylib._src.utility import style_temp_edit
from magpylib._src.utility import unit_prefix


class MagpyMarkers:
    """A class that stores markers 3D-coordinates."""

    def __init__(self, *markers):
        self._style = DefaultMarkers()
        self.markers = np.array(markers)

    @property
    def style(self):
        """Style property"""
        return self._style

    def get_trace(self, **kwargs):
        """Create the plotly mesh3d parameters for a Sensor object in a dictionary based on the
        provided arguments."""
        style = self.style
        x, y, z = self.markers.T
        marker_kwargs = {
            f"marker_{k}": v
            for k, v in style.marker.as_dict(flatten=True, separator="_").items()
        }
        marker_kwargs["marker_color"] = (
            style.marker.color if style.marker.color is not None else style.color
        )
        trace = {
            "type": "scatter3d",
            "x": x,
            "y": y,
            "z": z,
            "mode": "markers",
            "showlegend": style.legend.show,  # pylint: disable=no-member
            **marker_kwargs,
            **kwargs,
        }
        name = "Marker" if len(x) == 1 else "Markers"
        suff = "" if len(x) == 1 else f" ({len(x)} points)"
        trace["name"] = f"{name}{suff}"
        return trace


def update_magnet_mesh(
    mesh_dict, mag_style=None, magnetization=None, color_slicing=False
):
    """
    Updates an existing plotly mesh3d dictionary of an object which has a magnetic vector. The
    object gets colorized, positioned and oriented based on provided arguments.
    Slicing allows for matplotlib to show colorgradients approximations by slicing the mesh into
    the colorscales colors, remesh it and merge with assigning facecolor for each part.
    """
    mag_color = mag_style.color
    if magnetization is None:
        magnetization = np.array([0.0, 0.0, 0.0], dtype=float)
    if mag_style.show:
        vertices = np.array([mesh_dict[k] for k in "xyz"]).T
        color_middle = mag_color.middle
        if mag_color.mode == "tricycle":
            color_middle = mesh_dict["color"]
        elif mag_color.mode == "bicolor":
            color_middle = False
        ct = mag_color.transition
        cs = getColorscale(
            color_transition=0 if color_slicing else ct,
            color_north=mag_color.north,
            color_middle=color_middle,
            color_south=mag_color.south,
        )
        if color_slicing:
            if np.all(magnetization == 0):
                mesh_dict["color"] = mag_color.middle
            else:
                tr = slice_mesh_from_colorscale(mesh_dict, magnetization, cs)
                mesh_dict.update(tr)
        else:
            mesh_dict["colorscale"] = cs
            mesh_dict["intensity"] = getIntensity(
                vertices=vertices,
                axis=magnetization,
            )
        mesh_dict["showscale"] = False
        mesh_dict.pop("color_slicing", None)
    return mesh_dict


def make_mag_arrows(obj):
    """draw direction of magnetization of faced magnets

    Parameters
    ----------
    - obj: object with magnetization vector to be drawn
    - colors: colors of faced_objects
    - show_path(bool or int): draw on every position where object is displayed
    """
    # pylint: disable=protected-access

    # vector length, color and magnetization
    style = obj.style
    arrow = style.magnetization.arrow
    length = 1
    color = style.color if arrow.color is None else arrow.color
    if arrow.sizemode == "scaled":
        if hasattr(obj, "diameter"):
            length = obj.diameter  # Sphere
        elif isinstance(obj, magpy.misc.Triangle):
            length = np.amax(obj.vertices) - np.amin(obj.vertices)
        elif hasattr(obj, "mesh"):
            length = np.amax(np.ptp(obj.mesh.reshape(-1, 3), axis=0))
        elif hasattr(obj, "vertices"):
            length = np.amax(np.ptp(obj.vertices, axis=0))
        else:  # Cuboid, Cylinder, CylinderSegment
            length = np.amax(obj.dimension[:3])
        length *= 1.5
    length *= arrow.size
    mag = obj.magnetization
    # collect all draw positions and directions
    pos = getattr(obj, "_barycenter", obj._position)[0] - obj._position[0]
    # we need initial relative barycenter, arrow gets orientated later
    pos = obj._orientation[0].inv().apply(pos)
    direc = mag / (np.linalg.norm(mag) + 1e-6) * length
    x, y, z = draw_arrowed_line(
        direc, pos, sign=1, arrow_pos=arrow.offset, pivot="tail"
    ).T
    trace = {
        "type": "scatter3d",
        "mode": "lines",
        "line_width": arrow.width,
        "line_dash": arrow.style,
        "line_color": color,
        "x": x,
        "y": y,
        "z": z,
        "showlegend": False,
    }
    return trace


def make_path(input_obj, label=None):
    """draw obj path based on path style properties"""
    style = input_obj.style
    x, y, z = np.array(input_obj.position).T
    txt_kwargs = (
        {"mode": "markers+text+lines", "text": list(range(len(x)))}
        if style.path.numbering
        else {"mode": "markers+lines"}
    )
    marker = style.path.marker.as_dict()
    marker["symbol"] = marker["symbol"]
    marker["color"] = style.color if marker["color"] is None else marker["color"]
    line = style.path.line.as_dict()
    line["dash"] = line["style"]
    line["color"] = style.color if line["color"] is None else line["color"]
    line = {k: v for k, v in line.items() if k != "style"}
    scatter_path = {
        "type": "scatter3d",
        "x": x,
        "y": y,
        "z": z,
        "name": label,
        **{f"marker_{k}": v for k, v in marker.items()},
        **{f"line_{k}": v for k, v in line.items()},
        **txt_kwargs,
        "opacity": style.opacity,
    }
    return scatter_path


def get_trace2D_dict(
    BH,
    *,
    field_str,
    coords_str,
    obj_lst_str,
    focus_inds,
    frames_indices,
    mode,
    label_suff,
    units_polarization,
    units_magnetization,
    **kwargs,
):
    """return a 2d trace based on field and parameters"""
    coords_inds = ["xyz".index(k) for k in coords_str]
    y = BH.T[list(coords_inds)]
    if len(coords_inds) == 1:
        y = y[0]
    else:
        y = np.linalg.norm(y, axis=0)
    marker_size = np.array([3] * len(frames_indices))
    marker_size[focus_inds] = 15
    title = f"{field_str}{''.join(coords_str)}"
    unit = (
        units_polarization
        if field_str in "BJ"
        else units_magnetization if field_str in "HM" else ""
    )
    trace = {
        "mode": "lines+markers",
        "legendgrouptitle_text": f"{title}"
        + (f" ({label_suff})" if label_suff else ""),
        "text": mode,
        "hovertemplate": (
            "<b>Path index</b>: %{x}    "
            f"<b>{title}</b>: %{{y:.3s}}{unit}<br>"
            f"<b>{'sources'}</b>:<br>{obj_lst_str['sources']}<br>"
            f"<b>{'sensors'}</b>:<br>{obj_lst_str['sensors']}"
            # "<extra></extra>",
        ),
        "x": frames_indices,
        "y": y[frames_indices],
        "marker_size": marker_size,
        "showlegend": True,
        "legendgroup": f"{title}{label_suff}",
    }
    trace.update(kwargs)
    return trace


def get_traces_2D(
    *objects,
    output=("Bx", "By", "Bz"),
    row=None,
    col=None,
    sumup=True,
    pixel_agg=None,
    in_out="auto",
    styles=None,
    units_polarization="T",
    units_magnetization="A/m",
    # pylint: disable=unused-argument
    units_length="m",
    zoom=0,
):
    """draws and animates sensor values over a path in a subplot"""
    # pylint: disable=import-outside-toplevel
    from magpylib._src.fields.field_wrap_BH import getBH_level2

    sources = format_obj_input(objects, allow="sources+collections")
    sources = [
        s
        for s in sources
        if not (isinstance(s, magpy.Collection) and not s.sources_all)
    ]
    sensors = format_obj_input(objects, allow="sensors+collections")
    sensors = [
        sub_s
        for s in sensors
        for sub_s in (s.sensors_all if isinstance(s, magpy.Collection) else [s])
    ]

    if not isinstance(output, (list, tuple)):
        output = [output]
    output_params = {}
    field_str_list = []
    for out, linestyle in zip(output, cycle(ALLOWED_LINESTYLES[:6])):
        field_str, *coords_str = out
        if not coords_str:
            coords_str = list("xyz")
        if field_str not in "BHMJ" and set(coords_str).difference(set("xyz")):
            raise ValueError(
                "The `output` parameter must start with 'B', 'H', 'M', 'J' "
                "and be followed by a combination of 'x', 'y', 'z' (e.g. 'Bxy' or ('Bxy', 'Bz') )"
                f"\nreceived {out!r} instead"
            )
        field_str_list.append(field_str)
        output_params[out] = {
            "field_str": field_str,
            "coords_str": coords_str,
            "line_dash": linestyle,
        }
    field_str_list = list(dict.fromkeys(field_str_list))
    BH_array = {}
    for field_str in field_str_list:
        BH_array[field_str] = getBH_level2(
            sources,
            sensors,
            sumup=sumup,
            squeeze=False,
            field=field_str,
            pixel_agg=pixel_agg,
            output="ndarray",
            in_out=in_out,
        )
        # swap axes to have sensors first, path second
        BH_array[field_str] = BH_array[field_str].swapaxes(1, 2)
    frames_indices = np.arange(0, BH_array[field_str_list[0]].shape[2])

    def get_focus_inds(*objs):
        focus_inds = []
        for obj in objs:
            style = styles.get(obj, obj.style)
            frames = style.path.frames
            inds = [] if frames is None else frames
            if isinstance(inds, numbers.Number):
                # pylint: disable=invalid-unary-operand-type
                inds = frames_indices[::-frames]
            focus_inds.extend(inds)
        focus_inds = list(dict.fromkeys(focus_inds))
        return focus_inds if focus_inds else [-1]

    def get_obj_list_str(objs):
        if len(objs) < 8:
            obj_lst_str = "<br>".join(f" - {s}" for s in objs)
        else:
            counts = Counter(s.__class__.__name__ for s in objs)
            obj_lst_str = "<br>".join(f" {v}x {k}" for k, v in counts.items())
        return obj_lst_str

    def get_label_and_color(obj):
        style = styles.get(obj, None)
        style = obj.style if style is None else style
        label = get_legend_label(obj, style=style)
        color = getattr(style, "color", None)
        return label, color

    obj_lst_str = {
        "sources": get_obj_list_str(sources),
        "sensors": get_obj_list_str(sensors),
    }
    mode = "sources" if sumup else "sensors"

    traces = []
    for src_ind, src in enumerate(sources):
        if src_ind == 1 and sumup:
            break
        label_src, color_src = get_label_and_color(src)
        symbols = cycle(ALLOWED_SYMBOLS[:6])
        for sens_ind, sens in enumerate(sensors):
            focus_inds = get_focus_inds(src, sens)
            label_sens, color_sens = get_label_and_color(sens)
            label_suff = label_sens
            label = label_src
            line_color = color_src
            marker_color = color_sens if len(sensors) > 1 else None
            if sumup:
                line_color = color_sens
                label = label_sens
                label_suff = (
                    f"{label_src}" if len(sources) == 1 else f"{len(sources)} sources"
                )
            num_of_pix = (
                len(sens.pixel.reshape(-1, 3))
                if (not isinstance(sens, magpy.Collection))
                and sens.pixel is not None
                and sens.pixel.ndim != 1
                else 1
            )
            pix_suff = ""
            num_of_pix_to_show = 1 if pixel_agg else num_of_pix
            for pix_ind in range(num_of_pix_to_show):
                marker_symbol = next(symbols)
                if num_of_pix > 1:
                    if pixel_agg:
                        pix_suff = f" - {num_of_pix} pixels {pixel_agg}"
                    else:
                        pix_suff = f" - pixel {pix_ind}"
                for param in output_params.values():
                    BH = BH_array[param["field_str"]][src_ind, sens_ind, :, pix_ind]
                    traces.append(
                        get_trace2D_dict(
                            BH,
                            **param,
                            obj_lst_str=obj_lst_str,
                            focus_inds=focus_inds,
                            frames_indices=frames_indices,
                            mode=mode,
                            label_suff=label_suff,
                            name=f"{label}{pix_suff}",
                            line_color=line_color,
                            marker_color=marker_color,
                            marker_line_color=marker_color,
                            marker_symbol=marker_symbol,
                            type="scatter",
                            row=row,
                            col=col,
                            units_polarization=units_polarization,
                            units_magnetization=units_magnetization,
                        )
                    )
    return traces


def process_extra_trace(model):
    "process extra trace attached to some magpylib object"
    extr = model["model3d"]
    model_kwargs = {**(extr.kwargs() if callable(extr.kwargs) else extr.kwargs)}
    model_args = extr.args() if callable(extr.args) else extr.args
    trace3d = {
        "constructor": extr.constructor,
        "kwargs": model_kwargs,
        "args": model_args,
        "coordsargs": extr.coordsargs,
        "kwargs_extra": model["kwargs_extra"],
    }
    kwargs, args, coordsargs = place_and_orient_model3d(
        model_kwargs=model_kwargs,
        model_args=model_args,
        orientation=model["orientation"],
        position=model["position"],
        coordsargs=extr.coordsargs,
        scale=extr.scale,
        return_model_args=True,
        return_coordsargs=True,
    )
    trace3d["coordsargs"] = coordsargs
    trace3d["kwargs"].update(kwargs)
    trace3d["args"] = args
    return trace3d


def get_generic_traces3D(
    input_obj,
    autosize=None,
    legendgroup=None,
    legendtext=None,
    showlegend=None,
    supports_colorgradient=True,
    extra_backend=False,
    row=1,
    col=1,
    **kwargs,
) -> list:
    """
    This is a helper function providing the plotly traces for any object of the magpylib library. If
    the object is not supported, the trace representation will fall back to a single scatter point
    with the object name marked above it.

    - If the object has a path (multiple positions), the function will return both the object trace
    and the corresponding path trace. The legend entry of the path trace will be hidden but both
    traces will share the same `legendgroup` so that a legend entry click will hide/show both traces
    at once. From the user's perspective, the traces will be merged.

    - The argument caught by the kwargs dictionary must all be arguments supported both by
    `scatter3d` and `mesh3d` plotly objects, otherwise an error will be raised.
    """

    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-nested-blocks
    # pylint: disable=protected-access
    # pylint: disable=import-outside-toplevel
    style = input_obj.style
    is_mag_arrows = False
    is_mag = hasattr(input_obj, "magnetization") and hasattr(style, "magnetization")
    if is_mag and style.magnetization.show:
        magstyl = style.magnetization
        if magstyl.mode == "auto":
            magstyl.mode = "color" if supports_colorgradient else "arrow"
        is_mag_arrows = "arrow" in magstyl.mode
        magstyl.show = "color" in magstyl.mode

    make_func = getattr(input_obj, "get_trace", None)
    make_func_kwargs = {"legendgroup": legendgroup, **kwargs}
    if getattr(input_obj, "_autosize", False):
        make_func_kwargs["autosize"] = autosize

    has_path = hasattr(input_obj, "position") and hasattr(input_obj, "orientation")
    path_traces_extra_non_generic_backend = []
    if not has_path and make_func is not None:
        tr = make_func(**make_func_kwargs)
        tr["row"] = row
        tr["col"] = col
        out = {"generic": [tr]}
        if extra_backend:
            out.update({extra_backend: path_traces_extra_non_generic_backend})
        return out

    orientations, positions, pos_orient_inds = get_rot_pos_from_path(
        input_obj, style.path.frames
    )
    traces_generic = []
    if pos_orient_inds.size != 0:
        if style.model3d.showdefault and make_func is not None:
            p_trs = make_func(**make_func_kwargs)
            p_trs = [p_trs] if isinstance(p_trs, dict) else p_trs
            for p_tr in p_trs:
                is_mag = p_tr.pop("ismagnet", is_mag)
                if is_mag and p_tr.get("type", "") == "mesh3d":
                    p_tr = update_magnet_mesh(
                        p_tr,
                        mag_style=style.magnetization,
                        magnetization=input_obj.magnetization,
                        color_slicing=not supports_colorgradient,
                    )

                traces_generic.append(p_tr)

        extra_model3d_traces = (
            style.model3d.data if style.model3d.data is not None else []
        )
        for extr in extra_model3d_traces:
            if not extr.show:
                continue
            extr.update(extr.updatefunc())  # update before checking backend
            if extr.backend == "generic":
                extr.update(extr.updatefunc())
                tr_non_generic = {"opacity": style.opacity}
                ttype = extr.constructor.lower()
                obj_extr_trace = extr.kwargs() if callable(extr.kwargs) else extr.kwargs
                obj_extr_trace = {"type": ttype, **obj_extr_trace}
                if ttype == "scatter3d":
                    for k in ("marker", "line"):
                        tr_non_generic[f"{k}_color"] = tr_non_generic.get(
                            f"{k}_color", style.color
                        )
                elif ttype == "mesh3d":
                    tr_non_generic["showscale"] = tr_non_generic.get("showscale", False)
                    tr_non_generic["color"] = tr_non_generic.get("color", style.color)
                else:  # pragma: no cover
                    raise ValueError(
                        f"{ttype} is not supported, only 'scatter3d' and 'mesh3d' are"
                    )
                tr_non_generic.update(linearize_dict(obj_extr_trace, separator="_"))
                traces_generic.append(tr_non_generic)

    if is_mag_arrows:
        mag = input_obj.magnetization
        mag = np.array([0.0, 0.0, 0.0]) if mag is None else mag
        if not np.all(mag == 0):
            mag_arrow_tr = make_mag_arrows(input_obj)
            traces_generic.append(mag_arrow_tr)

    legend_label = get_legend_label(input_obj)

    path_traces_generic = []
    for tr in traces_generic:
        temp_rot_traces = []
        name_suff = tr.pop("name_suffix", None)
        name = tr.get("name", "") if legendtext is None else legendtext
        for orient, pos in zip(orientations, positions):
            tr1 = place_and_orient_model3d(tr, orientation=orient, position=pos)
            if name_suff is not None:
                tr1["name"] = f"{name}{name_suff}"
            temp_rot_traces.append(tr1)
        path_traces_generic.extend(group_traces(*temp_rot_traces))

    if np.array(input_obj.position).ndim > 1 and style.path.show:
        scatter_path = make_path(input_obj, legend_label)
        path_traces_generic.append(scatter_path)

    path_traces_generic = group_traces(*path_traces_generic)

    for tr in path_traces_generic:
        tr.update(place_and_orient_model3d(tr))
        tr.update(row=row, col=col)
        if tr.get("opacity", None) is None:
            tr["opacity"] = style.opacity
        if tr.get("legendgroup", None) is None:
            # allow invalid trimesh traces to have their own legendgroup
            tr["legendgroup"] = legendgroup
        if legendtext is not None:
            tr["name"] = legendtext
        elif "name" not in tr:
            tr["name"] = legend_label
        if tr.get("facecolor", None) is not None:
            # this allows merging of 3d meshes, ignoring different colors
            tr["color"] = None
        tr_showleg = tr.get("showlegend", None)
        # tr_showleg = True if tr_showleg is None else tr_showleg
        tr["showlegend"] = (
            showlegend
            if showlegend is not None
            else tr_showleg if style.legend.show else False
        )
    out = {"generic": path_traces_generic}

    if extra_backend:
        for extr in extra_model3d_traces:
            if not extr.show:
                continue
            extr.update(extr.updatefunc())  # update before checking backend
            if extr.backend == extra_backend:
                for orient, pos in zip(orientations, positions):
                    tr_non_generic = {
                        "model3d": extr,
                        "position": pos,
                        "orientation": orient,
                        "kwargs_extra": {
                            "opacity": style.opacity,
                            "color": style.color,
                            "legendgroup": legendgroup,
                            "showlegend": (
                                showlegend
                                if showlegend is not None
                                else None if style.legend.show else False
                            ),
                            "name": legendtext if legendtext else legend_label,
                            "row": row,
                            "col": col,
                        },
                    }
                    tr_non_generic = process_extra_trace(tr_non_generic)
                    path_traces_extra_non_generic_backend.append(tr_non_generic)
        out.update({extra_backend: path_traces_extra_non_generic_backend})
    return out


def clean_legendgroups(frames, clean_2d=False):
    """removes legend duplicates for a plotly figure"""
    for fr in frames:
        legendgroups = {}
        for tr in chain(fr["data"], fr["extra_backend_traces"]):
            if "z" in tr or clean_2d or "kwargs_extra" in tr:
                tr = tr.get("kwargs_extra", tr)
                lg = tr.get("legendgroup", None)
                if lg is not None:
                    tr_showlegend = tr.get("showlegend", None)
                    tr["showlegend"] = True if tr_showlegend is None else tr_showlegend
                    if lg not in legendgroups:
                        legendgroups[lg] = {"traces": [], "backup": None}
                    else:  # tr.legendgrouptitle.text is None:
                        tr["showlegend"] = False
                    legendgroups[lg]["traces"].append(tr)
                    if tr_showlegend is None:
                        legendgroups[lg]["backup"] = tr
    # legends with showlegend
    for lg in legendgroups.values():  # pragma: no cover
        if lg["backup"] is not None and all(
            tr["showlegend"] is False for tr in lg["traces"]
        ):
            lg["backup"]["showlegend"] = True


def process_animation_kwargs(obj_list, animation=False, **kwargs):
    """Update animation kwargs"""
    flat_obj_list = format_obj_input(obj_list)
    # set animation and animation_time
    if isinstance(animation, numbers.Number) and not isinstance(animation, bool):
        kwargs["animation_time"] = animation
        animation = True
    if (
        not any(
            getattr(obj, "position", np.array([])).ndim > 1 for obj in flat_obj_list
        )
        and animation is not False
    ):  # check if some path exist for any object
        animation = False
        warnings.warn("No path to be animated detected, displaying standard plot")

    # pylint: disable=no-member
    anim_def = default_settings.display.animation.copy()
    anim_def.update({k[10:]: v for k, v in kwargs.items()}, _match_properties=False)
    animation_kwargs = {f"animation_{k}": v for k, v in anim_def.as_dict().items()}
    kwargs = {k: v for k, v in kwargs.items() if not k.startswith("animation")}
    return kwargs, animation, animation_kwargs


def extract_animation_properties(
    objs,
    *,
    animation_maxfps,
    animation_time,
    animation_fps,
    animation_maxframes,
    # pylint: disable=unused-argument
    animation_slider,
    animation_output,
):
    """Extract animation properties"""
    # pylint: disable=import-outside-toplevel
    from magpylib._src.obj_classes.class_Collection import Collection

    path_lengths = []
    for obj in objs:
        subobjs = [obj]
        if isinstance(obj, Collection):
            subobjs.extend(obj.children)
        for subobj in subobjs:
            path_len = getattr(subobj, "_position", np.array((0.0, 0.0, 0.0))).shape[0]
            path_lengths.append(path_len)

    max_pl = max(path_lengths)
    if animation_fps > animation_maxfps:
        warnings.warn(
            f"The set `animation_fps` at {animation_fps} is greater than the max allowed of"
            f" {animation_maxfps}. `animation_fps` will be set to"
            f" {animation_maxfps}. "
            f"You can modify the default value by setting it in "
            "`magpylib.defaults.display.animation.maxfps`"
        )
        animation_fps = animation_maxfps

    maxpos = min(animation_time * animation_fps, animation_maxframes)

    if max_pl <= maxpos:
        path_indices = np.arange(max_pl)
    else:
        round_step = max_pl / (maxpos - 1)
        ar = np.linspace(0, max_pl, max_pl, endpoint=False)
        path_indices = np.unique(np.floor(ar / round_step) * round_step).astype(
            int
        )  # downsampled indices
        path_indices[-1] = (
            max_pl - 1
        )  # make sure the last frame is the last path position

    # calculate exponent of last frame index to avoid digit shift in
    # frame number display during animation
    exp = (
        np.log10(path_indices.max()).astype(int) + 1
        if path_indices.ndim != 0 and path_indices.max() > 0
        else 1
    )

    frame_duration = int(animation_time * 1000 / path_indices.shape[0])
    new_fps = int(1000 / frame_duration)
    if max_pl > animation_maxframes:
        warnings.warn(
            f"The number of frames ({max_pl}) is greater than the max allowed "
            f"of {animation_maxframes}. The `animation_fps` will be set to {new_fps}. "
            f"You can modify the default value by setting it in "
            "`magpylib.defaults.display.animation.maxframes`"
        )

    return path_indices, exp, frame_duration


def get_traces_3D(flat_objs_props, extra_backend=False, autosize=None, **kwargs):
    """Return traces, traces to resize and extra_backend_traces"""
    extra_backend_traces = []
    traces_dict = {}
    for obj, params in flat_objs_props.items():
        params = {**params, **kwargs}
        if autosize is None and getattr(obj, "_autosize", False):
            # temporary coordinates to be able to calculate ranges
            # pylint: disable=protected-access
            x, y, z = obj._position.T
            rc_dict = {k: v for k, v in params.items() if k in ("row", "col")}
            traces_dict[obj] = [{"x": x, "y": y, "z": z, "_autosize": True, **rc_dict}]
        else:
            traces_dict[obj] = []
            with style_temp_edit(obj, style_temp=params.pop("style", None), copy=True):
                out_traces = get_generic_traces3D(
                    obj,
                    extra_backend=extra_backend,
                    autosize=autosize,
                    **params,
                )
                if extra_backend:
                    extra_backend_traces.extend(out_traces.get(extra_backend, []))
                traces_dict[obj].extend(out_traces["generic"])
    return traces_dict, extra_backend_traces


def draw_frame(objs, *, colorsequence, rc_params, style_kwargs, **kwargs) -> Tuple:
    """
    Creates traces from input `objs` and provided parameters, updates the size of objects like
    Sensors and Dipoles in `kwargs` depending on the canvas size.

    Returns
    -------
    traces_dicts, kwargs: dict, dict
        returns the traces in a obj/traces_list dictionary and updated kwargs
    """
    if colorsequence is None:
        # pylint: disable=no-member
        colorsequence = default_settings.display.colorsequence
    # dipoles and sensors use autosize, the trace building has to be put at the back of the queue.
    # autosize is calculated from the other traces overall scene range
    objs_rc = get_objects_props_by_row_col(
        *objs,
        colorsequence=colorsequence,
        style_kwargs=style_kwargs,
    )
    traces_dict = {}
    extra_backend_traces = []
    rc_params = {} if rc_params is None else rc_params
    for rc, props in objs_rc.items():
        if props["rc_params"]["output"] == "model3d":
            rc_params[rc] = rc_params.get(rc, {})
            rc_params[rc]["units_length"] = props["rc_params"]["units_length"]
            rc_keys = ("row", "col")
            rc_kwargs = {k: v for k, v in props["rc_params"].items() if k in rc_keys}
            traces_d1, traces_ex1 = get_traces_3D(
                props["objects"], **rc_kwargs, **kwargs
            )
            rc_params[rc]["autosize"] = rc_params.get(rc, {}).get("autosize", None)
            if rc_params[rc]["autosize"] is None:
                zoom = rc_params[rc]["zoom"] = props["rc_params"]["zoom"]
                traces = [t for tr in traces_d1.values() for t in tr]
                ranges_rc = get_scene_ranges(*traces, *traces_ex1, zoom=zoom)
                # pylint: disable=no-member
                factor = default_settings.display.autosizefactor
                rc_params[rc]["autosize"] = np.mean(np.diff(ranges_rc[rc])) / factor
            to_resize_keys = {
                k for k, v in traces_d1.items() if v and "_autosize" in v[0]
            }
            flat_objs_props = {
                k: v for k, v in props["objects"].items() if k in to_resize_keys
            }
            traces_d2, traces_ex2 = get_traces_3D(
                flat_objs_props,
                autosize=rc_params[rc]["autosize"],
                **rc_kwargs,
                **kwargs,
            )
            traces_dict.update(
                {(k, *rc): v for k, v in {**traces_d1, **traces_d2}.items()}
            )
            extra_backend_traces.extend([*traces_ex1, *traces_ex2])
    traces = group_traces(*[t for tr in traces_dict.values() for t in tr])

    styles = {
        obj: params.get("style", None)
        for o_rc in objs_rc.values()
        for obj, params in o_rc["objects"].items()
    }
    for props in objs_rc.values():
        if props["rc_params"]["output"] != "model3d":
            traces2d = get_traces_2D(
                *props["objects"],
                **props["rc_params"],
                styles=styles,
            )
            traces.extend(traces2d)
    return traces, extra_backend_traces, rc_params


def get_frames(
    objs,
    colorsequence=None,
    title=None,
    animation=False,
    supports_colorgradient=True,
    backend="generic",
    style_kwargs=None,
    **kwargs,
):
    """This is a helper function which generates frames with generic traces to be provided to
    the chosen backend. According to a certain zoom level, all three space direction will be equal
    and match the maximum of the ranges needed to display all objects, including their paths.
    """
    # infer title if necessary
    if objs:
        style = objs[0]["objects"][0].style
        label = getattr(style, "label", None)
        title = label if len(objs[0]["objects"]) == 1 else None
    else:
        title = "No objects to be displayed"

    # make sure the number of frames does not exceed the max frames and max frame rate
    # downsample if necessary
    obj_list_semi_flat = format_obj_input(
        [o["objects"] for o in objs], allow="sources+sensors+collections"
    )
    kwargs, animation, animation_kwargs = process_animation_kwargs(
        obj_list_semi_flat, animation=animation, **kwargs
    )
    path_indices = [-1]
    if animation:
        path_indices, exp, frame_duration = extract_animation_properties(
            obj_list_semi_flat, **animation_kwargs
        )
    # create frame for each path index or downsampled path index
    frames = []

    title_str = title
    rc_params = {}
    for i, ind in enumerate(path_indices):
        extra_backend_traces = []
        if animation:
            style_kwargs["style_path_frames"] = [ind]
            title = "Animation 3D - " if title is None else title
            title_str = f"""{title}path index: {ind+1:0{exp}d}"""
        traces, extra_backend_traces, rc_params_temp = draw_frame(
            objs,
            colorsequence=colorsequence,
            rc_params=rc_params,
            supports_colorgradient=supports_colorgradient,
            extra_backend=backend,
            style_kwargs=style_kwargs,
            **kwargs,
        )
        if i == 0:  # get the dipoles and sensors autosize from first frame
            rc_params = rc_params_temp
        frames.append(
            {
                "data": traces,
                "name": str(ind + 1),
                "layout": {"title": title_str},
                "extra_backend_traces": extra_backend_traces,
            }
        )
    clean_legendgroups(frames)
    traces = [t for frame in frames for t in frame["data"]]
    zoom = {rc: v["zoom"] for rc, v in rc_params.items()}
    ranges_rc = get_scene_ranges(*traces, *extra_backend_traces, zoom=zoom)
    labels_rc = {(1, 1): {k: "" for k in "xyz"}}
    scale_factors_rc = {}
    for rc, params in rc_params.items():
        units_length = params["units_length"]
        if units_length == "auto":
            rmax = np.amax(np.abs(ranges_rc[rc]))
            units_length = f"{unit_prefix(rmax, as_tuple=True)[2]}m"
        unit_str = "" if not (units_length) else f" ({units_length})"
        labels_rc[rc] = {k: f"{k}{unit_str}" for k in "xyz"}
        scale_factors_rc[rc] = get_unit_factor(units_length, target_unit="m")
        ranges_rc[rc] *= scale_factors_rc[rc]

    for frame in frames:
        for key in ("data", "extra_backend_traces"):
            frame[key] = rescale_traces(frame[key], factors=scale_factors_rc)

    out = {
        "frames": frames,
        "ranges": ranges_rc,
        "labels": labels_rc,
        "input_kwargs": {**kwargs, **animation_kwargs},
    }
    if animation:
        out.update(
            {
                "frame_duration": frame_duration,
                "path_indices": path_indices,
            }
        )
    return out
