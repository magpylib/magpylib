"""Tests for path-varying target properties in getFT."""

import numpy as np

import magpylib as magpy

################################################################################
# CURRENT SOURCES TESTS
################################################################################


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


def test_path_varying_triangle_sheet_vertices_current_densities():
    """Test getFT with path-varying vertices and current_densities on TriangleSheet.

    Tests both geometry and current density varying along the path.
    """
    # Create a dipole source
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -2))

    # Define fixed vertices - simple mesh with 4 vertices forming 2 triangular faces
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ]
    )

    # Define faces (same for all path steps)
    faces = np.array([[0, 1, 2], [1, 3, 2]])

    # Path-varying current densities - rotating direction
    angles = np.array([0, np.pi / 4, np.pi / 2])
    current_densities_path = np.zeros((3, 2, 3))
    for i, angle in enumerate(angles):
        current_densities_path[i, :, 0] = 100 * np.cos(angle)  # x component
        current_densities_path[i, :, 1] = 100 * np.sin(angle)  # y component

    positions = np.array([[0, 0, i * 0.5] for i in range(3)])

    # VECTORIZED: Create TriangleSheet with path-varying current_densities only
    sheet_varying = magpy.current.TriangleSheet(
        vertices=vertices,
        faces=faces,
        current_densities=current_densities_path,
        position=positions,
        meshing=10,
    )

    F_vectorized, T_vectorized = magpy.getFT(dipole, sheet_varying)

    # MANUAL LOOP: Compute each path step separately
    F_manual = []
    T_manual = []

    for i in range(len(current_densities_path)):
        sheet_single = magpy.current.TriangleSheet(
            vertices=vertices,
            faces=faces,
            current_densities=current_densities_path[i],
            position=positions[i],
            meshing=10,
        )

        F_i, T_i = magpy.getFT(dipole, sheet_single)
        F_manual.append(F_i)
        T_manual.append(T_i)

    F_manual = np.array(F_manual)
    T_manual = np.array(T_manual)

    # Verify that both approaches give identical results
    np.testing.assert_allclose(
        F_vectorized,
        F_manual,
        rtol=1e-8,
        atol=1e-10,
        err_msg="Force calculation differs for path-varying TriangleSheet",
    )

    np.testing.assert_allclose(
        T_vectorized,
        T_manual,
        rtol=1e-8,
        atol=1e-10,
        err_msg="Torque calculation differs for path-varying TriangleSheet",
    )

    # Verify output shapes
    assert F_vectorized.shape == (3, 3), (
        f"Expected shape (3, 3), got {F_vectorized.shape}"
    )
    assert T_vectorized.shape == (3, 3), (
        f"Expected shape (3, 3), got {T_vectorized.shape}"
    )

    # Verify that values change along the path
    assert not np.allclose(F_vectorized[0], F_vectorized[1]), (
        "Forces should vary along path"
    )
    assert not np.allclose(F_vectorized[1], F_vectorized[2]), (
        "Forces should vary along path"
    )


def test_path_varying_triangle_strip_vertices_current():
    """Test getFT with path-varying vertices and current on TriangleStrip.

    Tests both geometry and current varying along the path.
    """
    # Create a dipole source
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -2))

    # Path-varying vertices - morphing strip from flat to curved
    vertices_path = np.array(
        [
            # Path step 0: flat strip
            [[0, 0, 0], [1, 0, 0], [0, 0.5, 0], [1, 0.5, 0]],
            # Path step 1: slightly curved
            [[0, 0, 0.1], [1, 0, 0.1], [0, 0.5, 0.15], [1, 0.5, 0.15]],
            # Path step 2: more curved
            [[0, 0, 0.2], [1, 0, 0.2], [0, 0.5, 0.3], [1, 0.5, 0.3]],
        ]
    )

    # Path-varying current - increasing magnitude
    currents_path = np.array([50, 100, 150])

    positions = np.array([[0, 0, i * 0.5] for i in range(3)])

    # VECTORIZED: Create TriangleStrip with path-varying vertices and current
    strip_varying = magpy.current.TriangleStrip(
        vertices=vertices_path,
        current=currents_path,
        position=positions,
        meshing=10,
    )

    F_vectorized, T_vectorized = magpy.getFT(dipole, strip_varying)

    # MANUAL LOOP: Compute each path step separately
    F_manual = []
    T_manual = []

    for i in range(len(currents_path)):
        strip_single = magpy.current.TriangleStrip(
            vertices=vertices_path[i],
            current=currents_path[i],
            position=positions[i],
            meshing=10,
        )

        F_i, T_i = magpy.getFT(dipole, strip_single)
        F_manual.append(F_i)
        T_manual.append(T_i)

    F_manual = np.array(F_manual)
    T_manual = np.array(T_manual)

    # Verify that both approaches give identical results
    np.testing.assert_allclose(
        F_vectorized,
        F_manual,
        rtol=1e-8,
        atol=1e-10,
        err_msg="Force calculation differs for path-varying TriangleStrip",
    )

    np.testing.assert_allclose(
        T_vectorized,
        T_manual,
        rtol=1e-8,
        atol=1e-10,
        err_msg="Torque calculation differs for path-varying TriangleStrip",
    )

    # Verify output shapes
    assert F_vectorized.shape == (3, 3), (
        f"Expected shape (3, 3), got {F_vectorized.shape}"
    )
    assert T_vectorized.shape == (3, 3), (
        f"Expected shape (3, 3), got {T_vectorized.shape}"
    )

    # Verify that values change along the path
    assert not np.allclose(F_vectorized[0], F_vectorized[1]), (
        "Forces should vary along path"
    )
    assert not np.allclose(F_vectorized[1], F_vectorized[2]), (
        "Forces should vary along path"
    )


################################################################################
# MAGNET SOURCES TESTS
################################################################################


def test_path_varying_cuboid_dimension_magnetization():
    """Test getFT with path-varying dimension and magnetization on Cuboid target.

    Tests both cases:
    1. Both dimension and magnetization varying (full vectorization)
    2. Only magnetization varying (optimized mesh reuse)
    """
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -5))

    # CASE 1: Both dimension and magnetization vary
    # Use dimensions with significantly different aspect ratios to test padding logic
    dimensions_varying = np.array(
        [
            [0.001, 0.001, 0.01],  # Elongated in z (1:1:10)
            [0.01, 0.001, 0.001],  # Elongated in x (10:1:1)
            [0.002, 0.002, 0.002],  # Cube (1:1:1)
        ]
    )
    magnetizations_varying = np.array(
        [
            [0, 0, 1e6],
            [0, 0, 1.2e6],
            [0, 0, 1.5e6],
        ]
    )
    positions = np.array([[0, 0, i * 0.01] for i in range(3)])

    cuboid_varying_both = magpy.magnet.Cuboid(
        dimension=dimensions_varying,
        magnetization=magnetizations_varying,
        position=positions,
        meshing=50,
    )

    F_vec_both, T_vec_both = magpy.getFT(dipole, cuboid_varying_both)

    # Manual loop for verification
    F_manual_both = []
    T_manual_both = []
    for i in range(3):
        cuboid_single = magpy.magnet.Cuboid(
            dimension=dimensions_varying[i],
            magnetization=magnetizations_varying[i],
            position=positions[i],
            meshing=50,
        )
        F_i, T_i = magpy.getFT(dipole, cuboid_single)
        F_manual_both.append(F_i)
        T_manual_both.append(T_i)

    F_manual_both = np.array(F_manual_both)
    T_manual_both = np.array(T_manual_both)

    np.testing.assert_allclose(F_vec_both, F_manual_both, rtol=1e-7, atol=1e-23)
    np.testing.assert_allclose(T_vec_both, T_manual_both, rtol=1e-7, atol=1e-27)

    # CASE 2: Only magnetization varies (optimized case - mesh reused)
    dimensions_constant = np.array(
        [
            [0.001, 0.002, 0.003],
            [0.001, 0.002, 0.003],  # Same dimensions
            [0.001, 0.002, 0.003],
        ]
    )

    cuboid_varying_mag = magpy.magnet.Cuboid(
        dimension=dimensions_constant,
        magnetization=magnetizations_varying,
        position=positions,
        meshing=50,
    )

    F_vec_mag, T_vec_mag = magpy.getFT(dipole, cuboid_varying_mag)

    # Manual loop for verification
    F_manual_mag = []
    T_manual_mag = []
    for i in range(3):
        cuboid_single = magpy.magnet.Cuboid(
            dimension=dimensions_constant[i],
            magnetization=magnetizations_varying[i],
            position=positions[i],
            meshing=50,
        )
        F_i, T_i = magpy.getFT(dipole, cuboid_single)
        F_manual_mag.append(F_i)
        T_manual_mag.append(T_i)

    F_manual_mag = np.array(F_manual_mag)
    T_manual_mag = np.array(T_manual_mag)

    np.testing.assert_allclose(F_vec_mag, F_manual_mag, rtol=1e-7, atol=1e-23)
    np.testing.assert_allclose(T_vec_mag, T_manual_mag, rtol=1e-7, atol=1e-27)

    # Verify shapes
    assert F_vec_both.shape == (3, 3)
    assert F_vec_mag.shape == (3, 3)

    # Verify path variation
    assert not np.allclose(F_vec_both[0], F_vec_both[1], rtol=0, atol=0)
    assert not np.allclose(F_vec_mag[0], F_vec_mag[1], rtol=0, atol=0)


def test_path_varying_sphere_diameter_magnetization():
    """Test getFT with path-varying diameter and magnetization on Sphere target.

    Tests both cases:
    1. Both diameter and magnetization varying (full path variation)
    2. Only magnetization varying (diameter constant)
    """
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -5))

    # CASE 1: Both diameter and magnetization vary
    diameters_varying = np.array([0.001, 0.0015, 0.002])
    magnetizations_varying = np.array(
        [
            [0, 0, 1e6],
            [0, 0, 1.5e6],
            [0, 0, 2e6],
        ]
    )
    positions = np.array([[0, 0, i * 0.5] for i in range(3)])

    sphere_both = magpy.magnet.Sphere(
        diameter=diameters_varying,
        magnetization=magnetizations_varying,
        position=positions,
    )

    F_vec_both, T_vec_both = magpy.getFT(dipole, sphere_both)

    # Compute with manual loop
    F_manual_both = []
    T_manual_both = []
    for i in range(3):
        sphere_single = magpy.magnet.Sphere(
            diameter=diameters_varying[i],
            magnetization=magnetizations_varying[i],
            position=positions[i],
        )
        F_i, T_i = magpy.getFT(dipole, sphere_single)
        F_manual_both.append(F_i)
        T_manual_both.append(T_i)

    F_manual_both = np.array(F_manual_both)
    T_manual_both = np.array(T_manual_both)

    np.testing.assert_allclose(
        F_vec_both,
        F_manual_both,
        rtol=1e-10,
        err_msg="Sphere with both varying should match manual loop",
    )

    np.testing.assert_allclose(
        T_vec_both,
        T_manual_both,
        rtol=1e-10,
        err_msg="Sphere torque with both varying should match manual loop",
    )

    # CASE 2: Only magnetization varies
    diameter_const = 0.001
    sphere_mag = magpy.magnet.Sphere(
        diameter=diameter_const,
        magnetization=magnetizations_varying,
        position=positions,
    )

    F_vec_mag, T_vec_mag = magpy.getFT(dipole, sphere_mag)

    # Compute with manual loop
    F_manual_mag = []
    T_manual_mag = []
    for i in range(3):
        sphere_single = magpy.magnet.Sphere(
            diameter=diameter_const,
            magnetization=magnetizations_varying[i],
            position=positions[i],
        )
        F_i, T_i = magpy.getFT(dipole, sphere_single)
        F_manual_mag.append(F_i)
        T_manual_mag.append(T_i)

    F_manual_mag = np.array(F_manual_mag)
    T_manual_mag = np.array(T_manual_mag)

    np.testing.assert_allclose(
        F_vec_mag,
        F_manual_mag,
        rtol=1e-10,
        err_msg="Sphere with magnetization varying should match manual loop",
    )

    np.testing.assert_allclose(
        T_vec_mag,
        T_manual_mag,
        rtol=1e-10,
        err_msg="Sphere torque with magnetization varying should match manual loop",
    )

    # Verify output shapes
    assert F_vec_both.shape == (3, 3)
    assert F_vec_mag.shape == (3, 3)

    # Verify path variation
    assert not np.allclose(F_vec_both[0], F_vec_both[1], rtol=0, atol=0)
    assert not np.allclose(F_vec_mag[0], F_vec_mag[1], rtol=0, atol=0)


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


################################################################################
# COLLECTIONS
################################################################################


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


################################################################################
# MISC TESTS
################################################################################


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
