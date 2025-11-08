"""Tests for path-varying target properties in getFT."""

import numpy as np

import magpylib as magpy


def test_path_varying_circle_current_diameter():
    """Test getFT with path-varying current and diameter on Circle target.

    Compares the vectorized path-varying implementation against a manual
    loop approach where each path step is computed separately.
    """
    # Create a dipole source at fixed position
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -2))

    # Define path-varying properties
    diameters = np.array([1.0, 2.0, 3.0, 4.0])
    currents = np.array([100.0, 200.0, 300.0, 400.0])
    positions = np.array([[0, 0, i] for i in range(4)])

    # VECTORIZED: Create circle with path-varying properties
    circle_varying = magpy.current.Circle(
        diameter=diameters, current=currents, position=positions, meshing=20
    )

    F_vectorized, T_vectorized = magpy.getFT(dipole, circle_varying)

    # MANUAL LOOP: Compute each path step separately
    F_manual = []
    T_manual = []

    for i in range(len(diameters)):
        # Create a new circle for each path step with single values
        circle_single = magpy.current.Circle(
            diameter=diameters[i],
            current=currents[i],
            position=positions[i],
            meshing=20,
        )

        F_i, T_i = magpy.getFT(dipole, circle_single)
        F_manual.append(F_i)
        T_manual.append(T_i)

    F_manual = np.array(F_manual)
    T_manual = np.array(T_manual)

    # Verify that both approaches give identical results
    np.testing.assert_allclose(
        F_vectorized,
        F_manual,
        rtol=1e-10,
        err_msg="Force calculation differs between vectorized and manual loop approach",
    )

    np.testing.assert_allclose(
        T_vectorized,
        T_manual,
        rtol=1e-10,
        err_msg="Torque calculation differs between vectorized and manual loop approach",
    )

    # Verify output shapes
    assert F_vectorized.shape == (4, 3), (
        f"Expected shape (4, 3), got {F_vectorized.shape}"
    )
    assert T_vectorized.shape == (4, 3), (
        f"Expected shape (4, 3), got {T_vectorized.shape}"
    )

    # Verify that values change along the path (not all the same)
    assert not np.allclose(F_vectorized[0], F_vectorized[1]), (
        "Forces should vary along path"
    )
    assert not np.allclose(F_vectorized[1], F_vectorized[2]), (
        "Forces should vary along path"
    )
    assert not np.allclose(T_vectorized[0], T_vectorized[1]), (
        "Torques should vary along path"
    )
