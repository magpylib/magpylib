# -------------------------------------------------------------------------------
# magpylib -- A Python 3 toolbox for working with magnetic fields.
# Copyright (C) Silicon Austria Labs, https://silicon-austria-labs.com/,
#               Michael Ortner <magpylib@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License along
# with this program.  If not, see <https://www.gnu.org/licenses/>.
# The acceptance of the conditions of the GNU Affero General Public License are
# compulsory for the usage of the software.
#
# For contact information, reach out over at <magpylib@gmail.com> or our issues
# page at https://www.github.com/magpylib/magpylib/issues.
# -------------------------------------------------------------------------------
"""
Source Classes
==============
   
Available source types for generating magnetic fields, 
compatible with :class:`~magpylib.Collection`

.. currentmodule:: magpylib.source.magnet


Quicklist for :mod:`~magpylib.source.magnet` sources:

.. autosummary::

   Box
   Sphere
   Cylinder

.. currentmodule:: magpylib.source.current

Quicklist for :mod:`~magpylib.source.current` sources:

.. autosummary::

   Line
   Circular


.. currentmodule:: magpylib.source.moment

Quicklist for :mod:`~magpylib.source.moment` sources:

.. autosummary::

   Dipole

All source objects possess the following attributes and methods:

.. image:: ../_static/images/documentation/lib_structure.JPG
   :scale: 50 %

"""
__all__ = ["magnet", "current", "moment"]  # This is for Sphinx

import magpylib._lib.classes.magnets as magnet
import magpylib._lib.classes.currents as current
import magpylib._lib.classes.moments as moment
