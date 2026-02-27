"""
Integration tests for puzzle_reconstruction algorithm modules:
  - fractal.box_counting
  - fractal.css
  - fractal.divider
  - fractal.ifs
  - tangram.classifier
  - tangram.hull
  - tangram.inscriber
  - gradient_flow
  - synthesis
  - texture_descriptor
  - word_segmentation
"""
import unittest
import math
import numpy as np

# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _circle_contour(n=200, radius=50, cx=100, cy=100, seed=None):
    """Generate a smooth closed circle contour (N, 2)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(t), np.sin(t)]) * radius + np.array([cx, cy])


def _noisy_circle_contour(n=200, radius=50, cx=100, cy=100, noise=2.0, seed=42):
    """Generate a noisy circle contour."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(t), np.sin(t)]) * radius + np.array([cx, cy])
    pts += rng.randn(n, 2) * noise
    return pts


def _square_contour(size=80, cx=100, cy=100):
    """Generate a square contour."""
    half = size / 2
    corners = np.array([
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ])
    # Interpolate each edge to get ~200 pts
    pts = []
    for i in range(4):
        p0 = corners[i]
        p1 = corners[(i + 1) % 4]
        for j in range(50):
            t = j / 50.0
            pts.append(p0 + t * (p1 - p0))
    return np.array(pts, dtype=np.float64)


def _grey_image(h=64, w=64, seed=7):
    """Random uint8 grayscale image."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w), dtype=np.uint8)


def _text_image(h=128, w=256, seed=3):
    """Synthetic document-like image: white background + black rectangles (words)."""
    rng = np.random.RandomState(seed)
    img = np.full((h, w), 255, dtype=np.uint8)
    for _ in range(8):
        x = rng.randint(5, w - 30)
        y = rng.randint(5, h - 15)
        ww = rng.randint(15, 35)
        hh = rng.randint(6, 12)
        x2 = min(x + ww, w - 1)
        y2 = min(y + hh, h - 1)
        img[y:y2, x:x2] = rng.randint(0, 100)
    return img


# ===========================================================================
# 1. box_counting
# ===========================================================================

class TestBoxCountingFD(unittest.TestCase):
    """Tests for fractal.box_counting.box_counting_fd and box_counting_curve."""

    def setUp(self):
        from puzzle_reconstruction.algorithms.fractal.box_counting import (
            box_counting_fd, box_counting_curve,
        )
        self.box_counting_fd = box_counting_fd
        self.box_counting_curve = box_counting_curve
        self.circle = _circle_contour()

    def test_returns_float(self):
        fd = self.box_counting_fd(self.circle)
        self.assertIsInstance(fd, float)

    def test_range_1_to_2(self):
        fd = self.box_counting_fd(self.circle)
        self.assertGreaterEqual(fd, 1.0)
        self.assertLessEqual(fd, 2.0)

    def test_smooth_circle_near_1(self):
        """A perfectly smooth circle should have FD close to 1."""
        fd = self.box_counting_fd(self.circle, n_scales=8)
        self.assertLess(fd, 1.5)

    def test_noisy_contour_higher_fd(self):
        """A noisier contour should have higher FD than a smooth one."""
        smooth = _circle_contour(n=500)
        noisy = _noisy_circle_contour(n=500, noise=5.0)
        fd_smooth = self.box_counting_fd(smooth)
        fd_noisy = self.box_counting_fd(noisy)
        self.assertGreaterEqual(fd_noisy, fd_smooth - 0.05)

    def test_tiny_contour_returns_1(self):
        """Contour with fewer than 4 points returns 1.0."""
        tiny = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        fd = self.box_counting_fd(tiny)
        self.assertEqual(fd, 1.0)

    def test_degenerate_all_same_point(self):
        """All-same-coordinate contour returns 1.0 (zero span)."""
        pts = np.ones((50, 2)) * 5.0
        fd = self.box_counting_fd(pts)
        self.assertEqual(fd, 1.0)

    def test_different_n_scales(self):
        """Different n_scales should all return valid FD."""
        for n in [4, 6, 8, 10]:
            fd = self.box_counting_fd(self.circle, n_scales=n)
            self.assertGreaterEqual(fd, 1.0)
            self.assertLessEqual(fd, 2.0)

    def test_box_counting_curve_lengths(self):
        log_r, log_N = self.box_counting_curve(self.circle, n_scales=6)
        self.assertEqual(len(log_r), 6)
        self.assertEqual(len(log_N), 6)

    def test_box_counting_curve_monotone(self):
        """log(N) should increase as log(1/r) increases (more bins → more occupied)."""
        log_r, log_N = self.box_counting_curve(self.circle, n_scales=8)
        self.assertTrue(np.all(np.diff(log_r) > 0))

    def test_square_contour(self):
        sq = _square_contour()
        fd = self.box_counting_fd(sq)
        self.assertGreaterEqual(fd, 1.0)
        self.assertLessEqual(fd, 2.0)


# ===========================================================================
# 2. css (Curvature Scale Space)
# ===========================================================================

class TestCSS(unittest.TestCase):
    """Tests for fractal.css functions."""

    def setUp(self):
        from puzzle_reconstruction.algorithms.fractal.css import (
            curvature_scale_space,
            css_to_feature_vector,
            freeman_chain_code,
        )
        self.css_fn = curvature_scale_space
        self.css_vec = css_to_feature_vector
        self.fcc = freeman_chain_code
        self.circle = _circle_contour()

    def test_curvature_scale_space_type(self):
        css = self.css_fn(self.circle)
        self.assertIsInstance(css, list)

    def test_curvature_scale_space_default_length(self):
        css = self.css_fn(self.circle)
        self.assertEqual(len(css), 7)

    def test_curvature_scale_space_tuples(self):
        css = self.css_fn(self.circle)
        for sigma, zc in css:
            self.assertIsInstance(sigma, float)
            self.assertIsInstance(zc, np.ndarray)

    def test_css_sigmas_increasing(self):
        css = self.css_fn(self.circle)
        sigmas = [s for s, _ in css]
        self.assertEqual(sigmas, sorted(sigmas))

    def test_css_to_feature_vector_shape(self):
        css = self.css_fn(self.circle, n_sigmas=5)
        vec = self.css_vec(css, n_bins=32)
        self.assertEqual(vec.shape, (5 * 32,))

    def test_css_to_feature_vector_unit_norm(self):
        """Feature vector should be L2-normalized."""
        css = self.css_fn(self.circle, n_sigmas=5)
        vec = self.css_vec(css, n_bins=32)
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_css_to_feature_vector_empty(self):
        vec = self.css_vec([])
        self.assertEqual(len(vec), 64)  # fallback n_bins

    def test_css_custom_sigma_range(self):
        sigmas = [1.0, 4.0, 16.0]
        css = self.css_fn(self.circle, sigma_range=sigmas)
        self.assertEqual(len(css), 3)
        self.assertAlmostEqual(css[0][0], 1.0)
        self.assertAlmostEqual(css[2][0], 16.0)

    def test_freeman_chain_code_string(self):
        code = self.fcc(self.circle)
        self.assertIsInstance(code, str)

    def test_freeman_chain_code_valid_digits(self):
        code = self.fcc(self.circle)
        for ch in code:
            self.assertIn(ch, "01234567")

    def test_freeman_chain_code_short_contour(self):
        """Contour with < 2 points returns empty string."""
        code = self.fcc(np.array([[5, 5]]))
        self.assertEqual(code, "")

    def test_freeman_chain_code_nonempty(self):
        """Normal contour should produce nonempty chain code."""
        code = self.fcc(self.circle)
        self.assertGreater(len(code), 0)


# ===========================================================================
# 3. divider
# ===========================================================================

class TestDividerFD(unittest.TestCase):
    """Tests for fractal.divider.divider_fd and divider_curve."""

    def setUp(self):
        from puzzle_reconstruction.algorithms.fractal.divider import (
            divider_fd, divider_curve,
        )
        self.divider_fd = divider_fd
        self.divider_curve = divider_curve
        self.circle = _circle_contour()

    def test_returns_float(self):
        fd = self.divider_fd(self.circle)
        self.assertIsInstance(fd, float)

    def test_range_1_to_2(self):
        fd = self.divider_fd(self.circle)
        self.assertGreaterEqual(fd, 1.0)
        self.assertLessEqual(fd, 2.0)

    def test_smooth_circle_close_to_1(self):
        fd = self.divider_fd(self.circle)
        self.assertLess(fd, 1.6)

    def test_zero_length_contour_returns_1(self):
        pts = np.ones((50, 2))
        fd = self.divider_fd(pts)
        self.assertEqual(fd, 1.0)

    def test_divider_curve_returns_arrays(self):
        log_s, log_L = self.divider_curve(self.circle)
        self.assertIsInstance(log_s, np.ndarray)
        self.assertIsInstance(log_L, np.ndarray)
        self.assertEqual(len(log_s), len(log_L))

    def test_divider_curve_positive_lengths(self):
        log_s, log_L = self.divider_curve(self.circle)
        self.assertGreater(len(log_s), 0)

    def test_noisy_higher_fd(self):
        noisy = _noisy_circle_contour(noise=4.0)
        smooth = _circle_contour()
        fd_smooth = self.divider_fd(smooth)
        fd_noisy = self.divider_fd(noisy)
        # noisy should not be much less than smooth
        self.assertGreaterEqual(fd_noisy + 0.1, fd_smooth)

    def test_different_n_scales(self):
        for n in [4, 6, 8]:
            fd = self.divider_fd(self.circle, n_scales=n)
            self.assertGreaterEqual(fd, 1.0)
            self.assertLessEqual(fd, 2.0)


# ===========================================================================
# 4. ifs (Iterated Function System)
# ===========================================================================

class TestIFS(unittest.TestCase):
    """Tests for fractal.ifs.fit_ifs_coefficients."""

    def setUp(self):
        from puzzle_reconstruction.algorithms.fractal.ifs import (
            fit_ifs_coefficients, reconstruct_from_ifs, ifs_distance,
        )
        self.fit_ifs = fit_ifs_coefficients
        self.reconstruct = reconstruct_from_ifs
        self.ifs_distance = ifs_distance
        self.circle = _circle_contour()

    def test_returns_array(self):
        coeffs = self.fit_ifs(self.circle)
        self.assertIsInstance(coeffs, np.ndarray)

    def test_default_n_transforms(self):
        coeffs = self.fit_ifs(self.circle, n_transforms=8)
        self.assertEqual(len(coeffs), 8)

    def test_coefficients_bounded(self):
        """IFS coefficients must satisfy |d| < 1 for convergence."""
        coeffs = self.fit_ifs(self.circle, n_transforms=8)
        self.assertTrue(np.all(np.abs(coeffs) <= 0.95))

    def test_small_curve(self):
        """Very small curve should still return coefficients."""
        tiny = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.5], [3.0, 0.0]])
        coeffs = self.fit_ifs(tiny, n_transforms=2)
        self.assertGreater(len(coeffs), 0)

    def test_different_n_transforms(self):
        for n in [4, 6, 8, 12]:
            coeffs = self.fit_ifs(self.circle, n_transforms=n)
            self.assertEqual(len(coeffs), n)

    def test_reconstruct_shape(self):
        coeffs = self.fit_ifs(self.circle, n_transforms=4)
        profile = self.reconstruct(coeffs, n_points=128)
        self.assertEqual(profile.shape, (128,))

    def test_ifs_distance_same(self):
        """Distance of array to itself should be zero."""
        coeffs = self.fit_ifs(self.circle, n_transforms=8)
        d = self.ifs_distance(coeffs, coeffs)
        self.assertAlmostEqual(d, 0.0, places=10)

    def test_ifs_distance_different(self):
        rng = np.random.RandomState(0)
        c1 = rng.uniform(-0.5, 0.5, 8)
        c2 = rng.uniform(-0.5, 0.5, 8)
        d = self.ifs_distance(c1, c2)
        self.assertGreater(d, 0.0)

    def test_ifs_distance_different_lengths(self):
        """Distance with different-length arrays should still work."""
        c1 = np.array([0.1, 0.2, 0.3, 0.4])
        c2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        d = self.ifs_distance(c1, c2)
        self.assertIsInstance(d, float)
        self.assertGreaterEqual(d, 0.0)


# ===========================================================================
# 5. tangram.classifier
# ===========================================================================

class TestTangramClassifier(unittest.TestCase):
    """Tests for tangram.classifier functions."""

    def setUp(self):
        from puzzle_reconstruction.algorithms.tangram.classifier import (
            classify_shape, compute_interior_angles,
        )
        from puzzle_reconstruction.models import ShapeClass
        self.classify_shape = classify_shape
        self.compute_angles = compute_interior_angles
        self.ShapeClass = ShapeClass

    def test_triangle_classification(self):
        tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        cls = self.classify_shape(tri)
        self.assertEqual(cls, self.ShapeClass.TRIANGLE)

    def test_rectangle_classification(self):
        rect = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
        cls = self.classify_shape(rect)
        self.assertEqual(cls, self.ShapeClass.RECTANGLE)

    def test_pentagon_classification(self):
        pts = np.array([
            [1.0, 0.0], [2.0, 0.7], [1.6, 1.8], [0.4, 1.8], [0.0, 0.7]
        ])
        cls = self.classify_shape(pts)
        self.assertEqual(cls, self.ShapeClass.PENTAGON)

    def test_hexagon_classification(self):
        # Regular hexagon: exactly 6 vertices
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        cls = self.classify_shape(pts)
        self.assertEqual(cls, self.ShapeClass.HEXAGON)

    def test_degenerate_polygon(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        cls = self.classify_shape(pts)
        self.assertEqual(cls, self.ShapeClass.POLYGON)

    def test_compute_interior_angles_triangle(self):
        # Equilateral triangle: all angles ~60 degrees
        pts = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3) / 2],
        ])
        angles = self.compute_angles(pts)
        self.assertEqual(len(angles), 3)
        for a in angles:
            self.assertAlmostEqual(np.degrees(a), 60.0, delta=1.0)

    def test_compute_interior_angles_rectangle(self):
        rect = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        angles = self.compute_angles(rect)
        self.assertEqual(len(angles), 4)
        for a in angles:
            self.assertAlmostEqual(np.degrees(a), 90.0, delta=1.0)

    def test_polygon_7_vertices(self):
        pts = np.array([[np.cos(2*np.pi*k/7), np.sin(2*np.pi*k/7)] for k in range(7)])
        cls = self.classify_shape(pts)
        self.assertEqual(cls, self.ShapeClass.POLYGON)


# ===========================================================================
# 6. tangram.hull
# ===========================================================================

class TestTangramHull(unittest.TestCase):
    """Tests for tangram.hull functions."""

    def setUp(self):
        from puzzle_reconstruction.algorithms.tangram.hull import (
            convex_hull, rdp_simplify, normalize_polygon,
        )
        self.convex_hull = convex_hull
        self.rdp_simplify = rdp_simplify
        self.normalize_polygon = normalize_polygon
        self.circle = _circle_contour().astype(np.float32)

    def test_convex_hull_returns_array(self):
        hull = self.convex_hull(self.circle)
        self.assertIsInstance(hull, np.ndarray)

    def test_convex_hull_fewer_points(self):
        hull = self.convex_hull(self.circle)
        self.assertLessEqual(len(hull), len(self.circle))

    def test_convex_hull_dtype(self):
        hull = self.convex_hull(self.circle)
        self.assertEqual(hull.dtype, np.float32)

    def test_rdp_simplify_fewer_points(self):
        simplified = self.rdp_simplify(self.circle)
        self.assertLessEqual(len(simplified), len(self.circle))

    def test_rdp_simplify_returns_float32(self):
        simplified = self.rdp_simplify(self.circle)
        self.assertEqual(simplified.dtype, np.float32)

    def test_rdp_simplify_zero_epsilon(self):
        """Zero epsilon should return the same points."""
        simplified = self.rdp_simplify(self.circle, epsilon_ratio=0.0)
        self.assertGreaterEqual(len(simplified), 2)

    def test_normalize_polygon_returns_tuple(self):
        hull = self.convex_hull(self.circle)
        result = self.normalize_polygon(hull)
        self.assertEqual(len(result), 4)

    def test_normalize_polygon_centroid_near_zero(self):
        hull = self.convex_hull(self.circle)
        normalized, centroid, scale, angle = self.normalize_polygon(hull)
        mean = normalized.mean(axis=0)
        np.testing.assert_allclose(mean, [0.0, 0.0], atol=1e-6)

    def test_normalize_polygon_scale(self):
        hull = self.convex_hull(self.circle)
        normalized, centroid, scale, angle = self.normalize_polygon(hull)
        self.assertGreater(scale, 0.0)


# ===========================================================================
# 7. tangram.inscriber (extract_tangram_edge via TangramSignature)
# ===========================================================================

class TestTangramInscriber(unittest.TestCase):
    """Tests for tangram.inscriber.extract_tangram_edge and fit_tangram."""

    def setUp(self):
        from puzzle_reconstruction.algorithms.tangram.inscriber import (
            extract_tangram_edge, fit_tangram,
        )
        from puzzle_reconstruction.models import TangramSignature, ShapeClass
        self.extract_edge = extract_tangram_edge
        self.fit_tangram = fit_tangram
        self.TangramSignature = TangramSignature
        self.ShapeClass = ShapeClass
        # Create a simple TangramSignature manually for edge extraction
        poly = np.array([
            [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]
        ], dtype=np.float64)
        self.sig = TangramSignature(
            polygon=poly,
            shape_class=ShapeClass.RECTANGLE,
            centroid=np.array([100.0, 100.0]),
            angle=0.0,
            scale=1.0,
            area=1.0,
        )

    def test_extract_edge_shape(self):
        edge = self.extract_edge(self.sig, edge_index=0, n_points=64)
        self.assertEqual(edge.shape, (64, 2))

    def test_extract_edge_default_n_points(self):
        edge = self.extract_edge(self.sig, edge_index=0)
        self.assertEqual(edge.shape, (128, 2))

    def test_extract_edge_is_straight_line(self):
        """For a rectangular polygon edge should be a straight line."""
        edge = self.extract_edge(self.sig, edge_index=0, n_points=10)
        # First point and last point differ; interior points are collinear
        p0 = edge[0]
        p_last = edge[-1]
        for pt in edge[1:-1]:
            # Cross product of (pt-p0) and (p_last-p0) should be ~0
            v1 = pt - p0
            v2 = p_last - p0
            cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
            self.assertAlmostEqual(cross, 0.0, places=10)

    def test_extract_edge_wraps_index(self):
        """Edge index wraps around modulo number of polygon vertices."""
        n_verts = len(self.sig.polygon)
        edge0 = self.extract_edge(self.sig, edge_index=0, n_points=10)
        edge_wrapped = self.extract_edge(self.sig, edge_index=n_verts, n_points=10)
        np.testing.assert_array_almost_equal(edge0, edge_wrapped)

    def test_fit_tangram_returns_signature(self):
        circle = _circle_contour().astype(np.float32)
        sig = self.fit_tangram(circle)
        self.assertIsInstance(sig, self.TangramSignature)

    def test_fit_tangram_polygon_not_empty(self):
        circle = _circle_contour().astype(np.float32)
        sig = self.fit_tangram(circle)
        self.assertGreater(len(sig.polygon), 0)

    def test_fit_tangram_scale_positive(self):
        circle = _circle_contour().astype(np.float32)
        sig = self.fit_tangram(circle)
        self.assertGreater(sig.scale, 0.0)

    def test_fit_tangram_area_nonneg(self):
        circle = _circle_contour().astype(np.float32)
        sig = self.fit_tangram(circle)
        self.assertGreaterEqual(sig.area, 0.0)

    def test_fit_tangram_shape_class_valid(self):
        circle = _circle_contour().astype(np.float32)
        sig = self.fit_tangram(circle)
        self.assertIn(sig.shape_class, list(self.ShapeClass))


# ===========================================================================
# 8. gradient_flow
# ===========================================================================

class TestGradientFlow(unittest.TestCase):
    """Tests for algorithms.gradient_flow module."""

    def setUp(self):
        from puzzle_reconstruction.algorithms.gradient_flow import (
            GradientField,
            compute_gradient,
            compute_magnitude,
            compute_orientation,
            compute_divergence,
            compare_gradient_fields,
            GradientStats,
            compute_gradient_stats,
        )
        self.GradientField = GradientField
        self.compute_gradient = compute_gradient
        self.compute_magnitude = compute_magnitude
        self.compute_orientation = compute_orientation
        self.compute_divergence = compute_divergence
        self.compare_gradient_fields = compare_gradient_fields
        self.GradientStats = GradientStats
        self.compute_gradient_stats = compute_gradient_stats
        self.img = _grey_image()

    def test_gradient_field_type(self):
        field = self.compute_gradient(self.img)
        self.assertIsInstance(field, self.GradientField)

    def test_gradient_field_shape(self):
        field = self.compute_gradient(self.img)
        h, w = self.img.shape
        self.assertEqual(field.gx.shape, (h, w))
        self.assertEqual(field.gy.shape, (h, w))

    def test_gradient_field_mismatch_raises(self):
        gx = np.zeros((10, 10), dtype=np.float32)
        gy = np.zeros((10, 12), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.GradientField(gx=gx, gy=gy)

    def test_compute_magnitude_shape(self):
        field = self.compute_gradient(self.img)
        mag = self.compute_magnitude(field)
        self.assertEqual(mag.shape, self.img.shape)

    def test_compute_magnitude_nonneg(self):
        field = self.compute_gradient(self.img)
        mag = self.compute_magnitude(field)
        self.assertTrue(np.all(mag >= 0))

    def test_compute_orientation_shape(self):
        field = self.compute_gradient(self.img)
        orient = self.compute_orientation(field)
        self.assertEqual(orient.shape, self.img.shape)

    def test_compute_orientation_range(self):
        field = self.compute_gradient(self.img)
        orient = self.compute_orientation(field)
        self.assertTrue(np.all(orient >= -np.pi - 1e-6))
        self.assertTrue(np.all(orient <= np.pi + 1e-6))

    def test_compute_divergence_shape(self):
        field = self.compute_gradient(self.img)
        div = self.compute_divergence(field)
        self.assertEqual(div.shape, self.img.shape)

    def test_compare_gradient_fields_identical(self):
        field = self.compute_gradient(self.img)
        sim = self.compare_gradient_fields(field, field)
        # Cosine similarity of a field with itself should be very close to 1
        # (slightly below due to floating-point epsilon in denominator)
        self.assertGreater(sim, 0.99)

    def test_compare_gradient_fields_range(self):
        rng = np.random.RandomState(1)
        img1 = rng.randint(0, 256, (64, 64), dtype=np.uint8)
        img2 = rng.randint(0, 256, (64, 64), dtype=np.uint8)
        f1 = self.compute_gradient(img1)
        f2 = self.compute_gradient(img2)
        sim = self.compare_gradient_fields(f1, f2)
        self.assertGreaterEqual(sim, -1.0)
        self.assertLessEqual(sim, 1.0)

    def test_compare_gradient_fields_shape_mismatch_raises(self):
        f1 = self.compute_gradient(np.zeros((32, 32), dtype=np.uint8))
        f2 = self.compute_gradient(np.zeros((64, 64), dtype=np.uint8))
        with self.assertRaises(ValueError):
            self.compare_gradient_fields(f1, f2)

    def test_compute_gradient_stats_type(self):
        field = self.compute_gradient(self.img)
        stats = self.compute_gradient_stats(field)
        self.assertIsInstance(stats, self.GradientStats)

    def test_compute_gradient_stats_mean_nonneg(self):
        field = self.compute_gradient(self.img)
        stats = self.compute_gradient_stats(field)
        self.assertGreaterEqual(stats.mean_magnitude, 0.0)

    def test_compute_gradient_stats_edge_density_range(self):
        field = self.compute_gradient(self.img)
        stats = self.compute_gradient_stats(field)
        self.assertGreaterEqual(stats.edge_density, 0.0)
        self.assertLessEqual(stats.edge_density, 1.0)

    def test_compute_gradient_stats_invalid_threshold_raises(self):
        field = self.compute_gradient(self.img)
        with self.assertRaises(ValueError):
            self.compute_gradient_stats(field, threshold=-1.0)

    def test_compute_gradient_invalid_ksize_raises(self):
        with self.assertRaises(ValueError):
            self.compute_gradient(self.img, ksize=4)

    def test_gradient_params_stored(self):
        field = self.compute_gradient(self.img, ksize=5)
        self.assertEqual(field.params.get("ksize"), 5)

    def test_gradient_normalize(self):
        field = self.compute_gradient(self.img, normalize=True)
        mag = self.compute_magnitude(field)
        self.assertLessEqual(float(mag.max()), 1.0 + 1e-5)


# ===========================================================================
# 9. synthesis
# ===========================================================================

class TestSynthesis(unittest.TestCase):
    """Tests for algorithms.synthesis module."""

    def setUp(self):
        from puzzle_reconstruction.algorithms.synthesis import (
            compute_fractal_signature,
            build_edge_signatures,
        )
        from puzzle_reconstruction.models import (
            Fragment, FractalSignature, TangramSignature, ShapeClass,
        )
        from puzzle_reconstruction.algorithms.tangram.inscriber import fit_tangram
        self.compute_fractal_signature = compute_fractal_signature
        self.build_edge_signatures = build_edge_signatures
        self.Fragment = Fragment
        self.FractalSignature = FractalSignature
        self.fit_tangram = fit_tangram
        self.ShapeClass = ShapeClass

        # Build a fragment for edge signature tests
        rng = np.random.RandomState(42)
        contour = _circle_contour(n=200).astype(np.float32)
        image = rng.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        self.contour = contour
        self.fragment = Fragment(
            fragment_id=0,
            image=image,
            contour=contour,
        )

    def test_compute_fractal_signature_type(self):
        sig = self.compute_fractal_signature(self.contour)
        self.assertIsInstance(sig, self.FractalSignature)

    def test_compute_fractal_signature_fd_box_range(self):
        sig = self.compute_fractal_signature(self.contour)
        self.assertGreaterEqual(sig.fd_box, 1.0)
        self.assertLessEqual(sig.fd_box, 2.0)

    def test_compute_fractal_signature_fd_divider_range(self):
        sig = self.compute_fractal_signature(self.contour)
        self.assertGreaterEqual(sig.fd_divider, 1.0)
        self.assertLessEqual(sig.fd_divider, 2.0)

    def test_compute_fractal_signature_ifs_coeffs(self):
        sig = self.compute_fractal_signature(self.contour)
        self.assertIsInstance(sig.ifs_coeffs, np.ndarray)
        self.assertEqual(len(sig.ifs_coeffs), 8)

    def test_compute_fractal_signature_css_image(self):
        sig = self.compute_fractal_signature(self.contour)
        self.assertIsInstance(sig.css_image, list)
        self.assertGreater(len(sig.css_image), 0)

    def test_compute_fractal_signature_chain_code(self):
        sig = self.compute_fractal_signature(self.contour)
        self.assertIsInstance(sig.chain_code, str)

    def test_compute_fractal_signature_curve_shape(self):
        sig = self.compute_fractal_signature(self.contour)
        self.assertEqual(sig.curve.shape, (256, 2))

    def test_build_edge_signatures_requires_tangram_and_fractal(self):
        """build_edge_signatures should raise AssertionError if tangram/fractal missing."""
        frag = self.Fragment(
            fragment_id=1,
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            contour=self.contour,
        )
        with self.assertRaises((AssertionError, AttributeError)):
            self.build_edge_signatures(frag)

    def test_build_edge_signatures_returns_list(self):
        frag = self.fragment
        frag.fractal = self.compute_fractal_signature(self.contour)
        frag.tangram = self.fit_tangram(self.contour)
        sigs = self.build_edge_signatures(frag, n_sides=4, n_points=64)
        self.assertIsInstance(sigs, list)

    def test_build_edge_signatures_count(self):
        frag = self.fragment
        frac = self.compute_fractal_signature(self.contour)
        tang = self.fit_tangram(self.contour)
        frag.fractal = frac
        frag.tangram = tang
        sigs = self.build_edge_signatures(frag, n_sides=4, n_points=64)
        self.assertEqual(len(sigs), 4)


# ===========================================================================
# 10. texture_descriptor
# ===========================================================================

class TestTextureDescriptor(unittest.TestCase):
    """Tests for algorithms.texture_descriptor module."""

    def setUp(self):
        from puzzle_reconstruction.algorithms.texture_descriptor import (
            TextureDescriptor,
            compute_lbp,
            compute_glcm_features,
            compute_stats_descriptor,
            compute_texture_descriptor,
            normalize_descriptor,
            descriptor_distance,
            batch_compute_descriptors,
        )
        self.TextureDescriptor = TextureDescriptor
        self.compute_lbp = compute_lbp
        self.compute_glcm = compute_glcm_features
        self.compute_stats = compute_stats_descriptor
        self.compute_desc = compute_texture_descriptor
        self.normalize = normalize_descriptor
        self.distance = descriptor_distance
        self.batch = batch_compute_descriptors
        self.img = _grey_image()

    def test_compute_lbp_shape(self):
        hist = self.compute_lbp(self.img, n_bins=64)
        self.assertEqual(hist.shape, (64,))

    def test_compute_lbp_normalized(self):
        hist = self.compute_lbp(self.img, n_bins=64)
        self.assertAlmostEqual(float(hist.sum()), 1.0, places=5)

    def test_compute_lbp_invalid_bins(self):
        with self.assertRaises(ValueError):
            self.compute_lbp(self.img, n_bins=1)

    def test_compute_glcm_features_shape(self):
        feats = self.compute_glcm(self.img)
        self.assertEqual(feats.shape, (4,))

    def test_compute_glcm_energy_nonneg(self):
        feats = self.compute_glcm(self.img)
        # energy (index 2) must be >= 0
        self.assertGreaterEqual(float(feats[2]), 0.0)

    def test_compute_glcm_invalid_levels(self):
        with self.assertRaises(ValueError):
            self.compute_glcm(self.img, levels=1)

    def test_compute_stats_descriptor_grayscale(self):
        vec = self.compute_stats(self.img)
        # For grayscale: [mean, std] → len 2
        self.assertEqual(len(vec), 2)

    def test_compute_stats_descriptor_rgb(self):
        rng = np.random.RandomState(0)
        rgb = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        vec = self.compute_stats(rgb)
        self.assertEqual(len(vec), 6)  # mean+std per channel

    def test_compute_texture_descriptor_lbp(self):
        desc = self.compute_desc(self.img, method="lbp", lbp_bins=32)
        self.assertIsInstance(desc, self.TextureDescriptor)
        self.assertEqual(desc.method, "lbp")

    def test_compute_texture_descriptor_glcm(self):
        desc = self.compute_desc(self.img, method="glcm")
        self.assertEqual(len(desc.vector), 4)

    def test_compute_texture_descriptor_combined(self):
        desc = self.compute_desc(self.img, method="combined", lbp_bins=32)
        self.assertGreater(len(desc.vector), 4)

    def test_compute_texture_descriptor_invalid_method(self):
        with self.assertRaises(ValueError):
            self.compute_desc(self.img, method="invalid")

    def test_normalize_descriptor_unit_norm(self):
        desc = self.compute_desc(self.img, method="lbp", lbp_bins=32)
        norm_desc = self.normalize(desc)
        norm = float(np.linalg.norm(norm_desc.vector))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_descriptor_distance_zero_self(self):
        desc = self.compute_desc(self.img, method="lbp", lbp_bins=32)
        d = self.distance(desc, desc)
        self.assertAlmostEqual(d, 0.0, places=8)

    def test_descriptor_distance_nonneg(self):
        rng = np.random.RandomState(1)
        img2 = rng.randint(0, 256, (64, 64), dtype=np.uint8)
        d1 = self.compute_desc(self.img, method="lbp", lbp_bins=32)
        d2 = self.compute_desc(img2, method="lbp", lbp_bins=32)
        dist = self.distance(d1, d2)
        self.assertGreaterEqual(dist, 0.0)

    def test_descriptor_distance_length_mismatch_raises(self):
        d1 = self.TextureDescriptor(vector=np.ones(10, dtype=np.float32))
        d2 = self.TextureDescriptor(vector=np.ones(20, dtype=np.float32))
        with self.assertRaises(ValueError):
            self.distance(d1, d2)

    def test_batch_compute_descriptors(self):
        rng = np.random.RandomState(5)
        imgs = [rng.randint(0, 256, (32, 32), dtype=np.uint8) for _ in range(4)]
        descs = self.batch(imgs, method="glcm")
        self.assertEqual(len(descs), 4)
        for i, desc in enumerate(descs):
            self.assertEqual(desc.image_id, i)

    def test_texture_descriptor_image_id_negative_raises(self):
        with self.assertRaises(ValueError):
            self.TextureDescriptor(vector=np.ones(4, dtype=np.float32), image_id=-1)

    def test_texture_descriptor_len(self):
        td = self.TextureDescriptor(vector=np.ones(16, dtype=np.float32))
        self.assertEqual(len(td), 16)


# ===========================================================================
# 11. word_segmentation
# ===========================================================================

class TestWordSegmentation(unittest.TestCase):
    """Tests for algorithms.word_segmentation module."""

    def setUp(self):
        from puzzle_reconstruction.algorithms.word_segmentation import (
            WordBox,
            LineSegment,
            WordSegmentationResult,
            binarize,
            segment_words,
            merge_line_words,
            segment_lines,
            segment_document,
        )
        self.WordBox = WordBox
        self.LineSegment = LineSegment
        self.WordSegmentationResult = WordSegmentationResult
        self.binarize = binarize
        self.segment_words = segment_words
        self.merge_line_words = merge_line_words
        self.segment_lines = segment_lines
        self.segment_document = segment_document
        self.text_img = _text_image()

    def test_binarize_otsu(self):
        bw = self.binarize(self.text_img, method="otsu")
        self.assertEqual(bw.shape, self.text_img.shape)
        unique = set(np.unique(bw))
        self.assertTrue(unique.issubset({0, 255}))

    def test_binarize_adaptive(self):
        bw = self.binarize(self.text_img, method="adaptive")
        self.assertEqual(bw.shape, self.text_img.shape)

    def test_binarize_invalid_method(self):
        with self.assertRaises(ValueError):
            self.binarize(self.text_img, method="unknown")

    def test_segment_words_returns_list(self):
        words = self.segment_words(self.text_img)
        self.assertIsInstance(words, list)

    def test_segment_words_wordbox_type(self):
        words = self.segment_words(self.text_img)
        for wb in words:
            self.assertIsInstance(wb, self.WordBox)

    def test_wordbox_properties(self):
        wb = self.WordBox(x=10, y=20, w=50, h=15)
        self.assertEqual(wb.x2, 60)
        self.assertEqual(wb.y2, 35)
        self.assertAlmostEqual(wb.cx, 35.0)
        self.assertAlmostEqual(wb.cy, 27.5)
        self.assertEqual(wb.area, 750)

    def test_wordbox_iou_self(self):
        wb = self.WordBox(x=0, y=0, w=10, h=10)
        self.assertAlmostEqual(wb.iou(wb), 1.0)

    def test_wordbox_iou_no_overlap(self):
        wb1 = self.WordBox(x=0, y=0, w=5, h=5)
        wb2 = self.WordBox(x=100, y=100, w=5, h=5)
        self.assertAlmostEqual(wb1.iou(wb2), 0.0)

    def test_merge_line_words_empty(self):
        lines = self.merge_line_words([])
        self.assertEqual(lines, [])

    def test_merge_line_words_returns_linesegments(self):
        words = [
            self.WordBox(x=10, y=20, w=30, h=10),
            self.WordBox(x=50, y=21, w=30, h=10),
            self.WordBox(x=10, y=80, w=30, h=10),
        ]
        lines = self.merge_line_words(words)
        self.assertGreater(len(lines), 0)
        for ln in lines:
            self.assertIsInstance(ln, self.LineSegment)

    def test_segment_document_returns_result(self):
        result = self.segment_document(self.text_img)
        self.assertIsInstance(result, self.WordSegmentationResult)

    def test_segment_document_image_shape(self):
        result = self.segment_document(self.text_img)
        h, w = self.text_img.shape
        self.assertEqual(result.image_shape, (h, w))

    def test_segment_document_n_words_nonneg(self):
        result = self.segment_document(self.text_img)
        self.assertGreaterEqual(result.n_words, 0)

    def test_segment_document_blank_image(self):
        """Blank image should produce 0 or very few words."""
        blank = np.full((100, 200), 255, dtype=np.uint8)
        result = self.segment_document(blank)
        self.assertGreaterEqual(result.n_words, 0)

    def test_linesegment_n_words(self):
        words = [self.WordBox(x=0, y=0, w=10, h=10)]
        ln = self.LineSegment(line_idx=0, words=words, bbox=(0, 0, 10, 10))
        self.assertEqual(ln.n_words, 1)

    def test_wordbox_aspect_ratio(self):
        wb = self.WordBox(x=0, y=0, w=20, h=10)
        self.assertAlmostEqual(wb.aspect_ratio, 2.0)


if __name__ == "__main__":
    unittest.main()
