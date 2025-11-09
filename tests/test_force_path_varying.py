"""Tests for path-varying target properties in getFT."""

import numpy as np
import pytest

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


def test_path_varying_cuboid_dimension_magnetization():
    """Test getFT with path-varying dimension and magnetization on Cuboid target.

    Tests that both geometry (dimension) and magnetization can vary along the path.
    Compares the vectorized path-varying implementation against a manual loop approach.
    """
    # Create a dipole source at fixed position
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -5))

    # Define path-varying properties
    dimensions = np.array(
        [
            [0.001, 0.002, 0.003],  # Step 0: small cuboid
            [0.0015, 0.0025, 0.0035],  # Step 1: medium cuboid
            [0.002, 0.003, 0.004],  # Step 2: large cuboid
        ]
    )
    magnetizations = np.array(
        [
            [0, 0, 1e6],  # Step 0: magnetized in +z
            [0, 0, 1.2e6],  # Step 1: stronger magnetization
            [0, 0, 1.5e6],  # Step 2: even stronger
        ]
    )
    positions = np.array([[0, 0, i * 0.01] for i in range(3)])

    # VECTORIZED: Create cuboid with path-varying properties
    cuboid_varying = magpy.magnet.Cuboid(
        dimension=dimensions,
        magnetization=magnetizations,
        position=positions,
        meshing=50,
    )

    F_vectorized, T_vectorized = magpy.getFT(dipole, cuboid_varying)

    # MANUAL LOOP: Compute each path step separately
    F_manual = []
    T_manual = []

    for i in range(len(dimensions)):
        # Create a new cuboid for each path step with single values
        cuboid_single = magpy.magnet.Cuboid(
            dimension=dimensions[i],
            magnetization=magnetizations[i],
            position=positions[i],
            meshing=50,
        )

        F_i, T_i = magpy.getFT(dipole, cuboid_single)
        F_manual.append(F_i)
        T_manual.append(T_i)

    F_manual = np.array(F_manual)
    T_manual = np.array(T_manual)

    # Verify that both approaches give identical results
    np.testing.assert_allclose(
        F_vectorized,
        F_manual,
        rtol=1e-7,
        atol=1e-23,
        err_msg="Force calculation differs between vectorized and manual loop approach",
    )

    np.testing.assert_allclose(
        T_vectorized,
        T_manual,
        rtol=1e-7,
        atol=1e-27,
        err_msg="Torque calculation differs between vectorized and manual loop approach",
    )

    # Verify output shapes
    assert F_vectorized.shape == (3, 3), (
        f"Expected shape (3, 3), got {F_vectorized.shape}"
    )
    assert T_vectorized.shape == (3, 3), (
        f"Expected shape (3, 3), got {T_vectorized.shape}"
    )

    # Verify that values change along the path (not all the same)
    assert not np.allclose(F_vectorized[0], F_vectorized[1], rtol=0, atol=0), (
        "Forces should vary along path due to changing dimension and magnetization"
    )
    assert not np.allclose(F_vectorized[1], F_vectorized[2], rtol=0, atol=0), (
        "Forces should vary along path due to changing dimension and magnetization"
    )


def test_cuboid_aspect_ratio_warning():
    """Test that warning is triggered when cuboid aspect ratio changes significantly.

    The warning should be triggered when aspect ratio changes by more than 2.0x along the path.
    """
    # Create cuboid with dimensions that cause >2x aspect ratio change
    dimensions = np.array(
        [
            [0.001, 0.001, 0.001],  # Step 0: cube (1:1:1)
            [0.003, 0.001, 0.001],  # Step 1: elongated (3:1:1) -> 3x change
        ]
    )

    cuboid = magpy.magnet.Cuboid(
        dimension=dimensions,
        magnetization=[[0, 0, 1e6], [0, 0, 1e6]],
        meshing=10,
    )

    # Warning is triggered when mesh is generated (lazy evaluation)
    with pytest.warns(
        UserWarning,
        match="Cuboid mesh cells vary significantly in aspect ratio",
    ):
        _ = cuboid._generate_mesh()


def test_path_varying_polyline_vertices():
    """Test getFT with path-varying vertices on Polyline target.

    Tests that geometry itself can vary along the path, not just properties.
    """
    # Create a dipole source
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -2))

    # Define path-varying vertices - expanding square loop
    vertices_path = np.array(
        [
            [  # Step 0: small square
                [-0.5, -0.5, 0],
                [0.5, -0.5, 0],
                [0.5, 0.5, 0],
                [-0.5, 0.5, 0],
                [-0.5, -0.5, 0],
            ],
            [  # Step 1: medium square
                [-1.0, -1.0, 0],
                [1.0, -1.0, 0],
                [1.0, 1.0, 0],
                [-1.0, 1.0, 0],
                [-1.0, -1.0, 0],
            ],
            [  # Step 2: large square
                [-1.5, -1.5, 0],
                [1.5, -1.5, 0],
                [1.5, 1.5, 0],
                [-1.5, 1.5, 0],
                [-1.5, -1.5, 0],
            ],
        ]
    )

    currents = np.array([100.0, 100.0, 100.0])
    positions = np.array([[0, 0, i] for i in range(3)])

    # VECTORIZED: Create polyline with path-varying vertices
    polyline_varying = magpy.current.Polyline(
        vertices=vertices_path, current=currents, position=positions, meshing=20
    )

    F_vectorized, T_vectorized = magpy.getFT(dipole, polyline_varying)

    # MANUAL LOOP: Compute each path step separately
    F_manual = []
    T_manual = []

    for i in range(len(vertices_path)):
        polyline_single = magpy.current.Polyline(
            vertices=vertices_path[i],
            current=currents[i],
            position=positions[i],
            meshing=20,
        )

        F_i, T_i = magpy.getFT(dipole, polyline_single)
        F_manual.append(F_i)
        T_manual.append(T_i)

    F_manual = np.array(F_manual)
    T_manual = np.array(T_manual)

    # Verify that both approaches give identical results
    np.testing.assert_allclose(
        F_vectorized,
        F_manual,
        rtol=1e-10,
        err_msg="Force calculation differs for path-varying vertices",
    )

    np.testing.assert_allclose(
        T_vectorized,
        T_manual,
        rtol=1e-10,
        err_msg="Torque calculation differs for path-varying vertices",
    )

    # Verify forces decrease as polyline moves away (increasing z position)
    force_magnitudes = np.linalg.norm(F_vectorized, axis=1)
    assert force_magnitudes[1] < force_magnitudes[0], (
        "Force should decrease as polyline moves farther from dipole"
    )
    assert force_magnitudes[2] < force_magnitudes[1], (
        "Force should decrease as polyline moves farther from dipole"
    )


def test_mismatched_path_lengths():
    """Test getFT with different path lengths for source and target.

    Verifies that the result is padded to the maximum path length.
    """
    # Source with 5 path steps
    dipole = magpy.misc.Dipole(
        moment=(1e3, 0, 0), position=[(0, 0, -2 - i * 0.5) for i in range(5)]
    )

    # Target with 2 path steps (varying properties)
    circle = magpy.current.Circle(
        diameter=[1.0, 2.0],
        current=[100.0, 200.0],
        position=[(0, 0, 0), (0, 0, 1)],
        meshing=20,
    )

    F, T = magpy.getFT(dipole, circle)

    # Result should have max(5, 2) = 5 path steps
    assert F.shape == (5, 3), f"Expected shape (5, 3), got {F.shape}"
    assert T.shape == (5, 3), f"Expected shape (5, 3), got {T.shape}"

    # Steps 2-4 should use padded values (last values of diameter/current)
    # Verify by computing manually for step 2
    dipole_step2 = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -3))
    circle_step2 = magpy.current.Circle(
        diameter=2.0, current=200.0, position=(0, 0, 1), meshing=20
    )  # Padded values

    F_expected, T_expected = magpy.getFT(dipole_step2, circle_step2)

    np.testing.assert_allclose(
        F[2], F_expected, rtol=1e-10, err_msg="Padded step should match expected value"
    )
    np.testing.assert_allclose(
        T[2], T_expected, rtol=1e-10, err_msg="Padded step should match expected value"
    )


def test_single_element_array_vs_scalar():
    """Test that single-element arrays produce identical results to scalars.

    Ensures backward compatibility.
    """
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -2))

    # Scalar properties
    circle_scalar = magpy.current.Circle(
        diameter=2.0, current=100.0, position=(0, 0, 0), meshing=20
    )

    # Single-element array properties
    circle_array = magpy.current.Circle(
        diameter=[2.0], current=[100.0], position=[(0, 0, 0)], meshing=20
    )

    F_scalar, T_scalar = magpy.getFT(dipole, circle_scalar)
    F_array, T_array = magpy.getFT(dipole, circle_array)

    # Results should be identical
    np.testing.assert_allclose(
        F_scalar,
        F_array,
        rtol=1e-15,
        err_msg="Single-element array should produce same result as scalar",
    )

    np.testing.assert_allclose(
        T_scalar,
        T_array,
        rtol=1e-15,
        err_msg="Single-element array should produce same result as scalar",
    )


def test_path_varying_with_centroid_pivot():
    """Test that default pivot (centroid) works correctly with path-varying targets.

    Verifies that centroid is computed at each path step for varying geometry.
    """
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -2))

    # Path-varying circle (different diameters)
    diameters = np.array([1.0, 2.0, 3.0])
    currents = np.array([100.0, 100.0, 100.0])
    positions = np.array([[0, 0, i] for i in range(3)])

    circle = magpy.current.Circle(
        diameter=diameters, current=currents, position=positions, meshing=20
    )

    # Compute with default pivot (centroid)
    _F_default, T_default = magpy.getFT(dipole, circle)

    # Verify against manual loop with per-step centroids (explicit pivot)
    T_manual = []
    for i in range(len(diameters)):
        circle_single = magpy.current.Circle(
            diameter=diameters[i],
            current=currents[i],
            position=positions[i],
            meshing=20,
        )
        _, T_i = magpy.getFT(dipole, circle_single, pivot="centroid")
        T_manual.append(T_i)

    T_manual = np.array(T_manual)

    np.testing.assert_allclose(
        T_default,
        T_manual,
        rtol=1e-10,
        err_msg="Default pivot (centroid) should match manual per-step calculation",
    )


def test_path_varying_with_collections():
    """Test getFT with Collections containing path-varying targets.

    Verifies that forces are correctly summed across collection members.
    """
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -3))

    # Two circles with different path-varying properties
    circle1 = magpy.current.Circle(
        diameter=[1.0, 2.0], current=[100.0, 200.0], meshing=20
    )

    circle2 = magpy.current.Circle(
        diameter=[1.5, 2.5], current=[150.0, 250.0], meshing=20
    )

    # Create collection
    coll = magpy.Collection(circle1, circle2)
    coll.position = [(0, 0, 0), (0, 0, 1)]

    F_coll, T_coll = magpy.getFT(dipole, coll)

    # Compute individual forces and sum manually
    F1, T1 = magpy.getFT(dipole, [circle1, circle2])
    F_manual = F1[:, 0, :] + F1[:, 1, :]  # Sum across targets
    T_manual = T1[:, 0, :] + T1[:, 1, :]

    np.testing.assert_allclose(
        F_coll,
        F_manual,
        rtol=1e-10,
        err_msg="Collection should sum forces correctly",
    )

    np.testing.assert_allclose(
        T_coll,
        T_manual,
        rtol=1e-10,
        err_msg="Collection should sum torques correctly",
    )


def test_zero_current_in_path():
    """Test that zero current in path doesn't cause numerical issues."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -2))

    # Include zero current at first step
    circle = magpy.current.Circle(
        diameter=[2.0, 2.0, 2.0],
        current=[0.0, 100.0, 200.0],
        position=[(0, 0, i) for i in range(3)],
        meshing=20,
    )

    F, T = magpy.getFT(dipole, circle)

    # Verify no NaN or inf values
    assert not np.any(np.isnan(F)), "Result should not contain NaN"
    assert not np.any(np.isinf(F)), "Result should not contain inf"
    assert not np.any(np.isnan(T)), "Result should not contain NaN"
    assert not np.any(np.isinf(T)), "Result should not contain inf"

    # First step should have near-zero force (zero current)
    assert np.linalg.norm(F[0]) < 1e-10, "Zero current should produce near-zero force"

    # Later steps should have non-zero forces
    assert np.linalg.norm(F[1]) > 1e-6, "Non-zero current should produce force"
    assert np.linalg.norm(F[2]) > 1e-6, "Non-zero current should produce force"
