"""
=================
Display styling
=================
"""

# %%
# Displaying objects may not yield by default the visual representation the user wants. For this cases, the library includes
# a variety of styling options that can be applied at multiple levels in the user's code.

# %%
# Hierarchy of arguments
# ----------------------
# from LOWEST to HIGHEST precedence
#
# - library ``defaults``
# - individual object ``style`` or at ``Collection`` level
# - in the ``display`` function

# %%
# Examples
# --------

#%%
import magpylib as magpy
import plotly.graph_objects as go

magpy.defaults.reset()

cuboid = magpy.magnet.Cuboid(
    magnetization=(1, 0, 0), dimension=(1, 1, 1), position=(0, 0, 0)
)
cylinder = magpy.magnet.Cylinder(
    magnetization=(0, 1, 0), dimension=(1, 1), position=(2, 0, 0)
)
sphere = magpy.magnet.Sphere(magnetization=(0, 1, 1), diameter=1, position=(4, 0, 0))
col = magpy.Collection(cuboid, cylinder, sphere)

#%%
# **Setting library defaults**:
my_default_magnetization_style = {
    "show": True,
    "color": {
        "transition": 1,
        "mode": "tricolor",
        "middle": "white",
        "north": "magenta",
        "south": "turquoise",
    },
}
magpy.defaults.display.style.magnet.magnetization = my_default_magnetization_style

fig = go.Figure()
magpy.display(col, canvas=fig, backend="plotly")
fig

#%%
# **Setting style via ``Collection``**:
#
# .. note::
#   The ``Collection`` object does not hold any ``style`` attribute on its own but the helper method ``set_styles``
#   allows setting the styles of all its children where the set arguments match existing child style attributes.

col.set_styles(magnetization_color_south='blue')

fig = go.Figure()
magpy.display(col, canvas=fig, backend="plotly")
fig

#%%
# **Setting individual styles**:
cylinder.style.update(magnetization_color_mode = 'bicolor')
cuboid.style.magnetization.color = dict(mode = 'tricycle')

fig = go.Figure()
magpy.display(col, canvas=fig, backend="plotly")
fig

#%%
# **Overriding style at display time**:
#
# .. note::
#   Setting style parameters in the ``display`` function does not change the default styles nor the set object style. It only
#   affects the current representation to be displayed.
#
# The provided styling properties as function arguments will override temporarily the styles set by any of the afermentioned methods.
# All styling properties need to start with ``style`` and underscore magic is supported.

fig = go.Figure()
magpy.display(col, canvas=fig, backend="plotly", style_magnetization_show=False)
fig
# %%
# List of available styles
# ------------------------
style = magpy.defaults.display.style.as_dict(flatten=True)
print("\n".join(f"{k!r}: {v!r}" for k, v in style.items()))

# %%
# .. warning::
#   Even if both ``matplotlib`` and ``plotly`` backends can display all object of the library, there is no 100% feature parity between
#   them. Some of the differences include (non-exhaustive list):
#
#   - ``magnetization.size`` -> ``matplotlib`` only
#   - ``magnetization.color`` -> ``plotly`` only