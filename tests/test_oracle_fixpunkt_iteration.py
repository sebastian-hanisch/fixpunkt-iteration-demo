"""Orakel-Tests: die Kernroutinen gegen einen anderen Rechenweg (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §1 und §5).

Die Demo rechnet ‖A^k‖ in geschlossener Form und die 2×2-Norm aus Frobeniusnorm und Determinante; die Orakel rechnen mit `numpy.linalg.matrix_power` und der
Singulärwertzerlegung (`numpy.linalg.norm(·, 2)`), mit exakten Brüchen (`fractions`) und mit `numpy.linalg.eigvals`. Der Fixpunkt von cos kommt gegen
`scipy.optimize.brentq` (fehlt scipy, überspringt sich dieser Test). Zufall nur über SplitMix64."""

import math
from fractions import Fraction

import numpy as np
import pytest

import fpi_algorithm as A
import fpi_constants as C
from fpi_scenario import BANACH_MAPS


def _svd_norm_of_power(M, k):
    return float(np.linalg.norm(np.linalg.matrix_power(M, k), 2))


# ---------------------------------------------------------------- ‖A^k‖: geschlossene Form gegen Matrixpotenz + SVD
def test_jordan_norms_equal_matrix_power_and_svd():
    for a in (0.0, 0.3, 0.9, 0.9 + 1e-9, 0.99, 1.0, 1.2):      # a + 1e-9: kein Dualbruch-Sonderfall (Rechner-Artefakt der Linie)
        for b in (0.0, 0.1, 1.0, 5.0, 10.0):
            M = np.array([[a, b], [0.0, a]])
            norms = A.jordan_power_norms(a, b, 150)
            for k in (0, 1, 2, 3, 9, 10, 37, 100, 150):
                ref = _svd_norm_of_power(M, k) if k else 1.0
                assert norms[k] == pytest.approx(ref, rel=1e-9, abs=1e-12), (a, b, k)


def test_jordan_norm_satisfies_the_singular_value_equation_in_exact_fractions():
    """s = ‖A^k‖ ist der größte Singulärwert: s⁴ − F·s² + D² = 0 mit F = 2p² + q², D = p² (p, q als exakte Brüche, a = 9/10, b = 5)."""
    a, b = Fraction(9, 10), Fraction(5)
    norms = A.jordan_power_norms(0.9, 5.0, 120)
    for k in (1, 2, 9, 30, 55, 100, 120):
        p = a ** k
        q = k * a ** (k - 1) * b
        F, D = 2 * p * p + q * q, p * p
        s = Fraction(float(norms[k]))
        residual = abs(s ** 4 - F * s ** 2 + D * D) / (F * F)
        assert float(residual) < 1e-13, k
        s2 = D / s                       # der kleinere Singulärwert; der Prüfling liefert den größeren
        assert s >= s2


def test_peak_summary_equals_matrix_power_scan():
    for a, b in ((0.9, 5.0), (0.6, 3.0), (0.99, 1.0), (0.7, 0.0), (0.0, 4.0), (1.0, 2.0)):
        M = np.array([[a, b], [0.0, a]])
        ref = [1.0] + [_svd_norm_of_power(M, k) for k in range(1, 400)]
        s = A.buckel_summary(a, b)
        assert s.peak == pytest.approx(max(ref), rel=1e-9)
        assert s.k_peak == int(np.argmax(ref)), (a, b)
        after = [k for k in range(s.k_peak, 400) if ref[k] < 1.0]
        assert s.first_below_one == (after[0] if after else None), (a, b)


# ---------------------------------------------------------------- 2×2-Norm und Heavy-Ball-Matrix
def test_norm2_2x2_matches_svd_on_200_random_matrices(zufall):
    rng = zufall(11)
    for _ in range(200):
        P = np.array([[rng.uniform(-3, 3) for _ in range(2)] for _ in range(2)])
        assert float(A.norm2_2x2(P[0, 0], P[0, 1], P[1, 0], P[1, 1])) == pytest.approx(float(np.linalg.norm(P, 2)), rel=1e-12)


def test_heavy_ball_norms_equal_matrix_power_on_200_random_points(zufall):
    rng = zufall(12)
    for _ in range(200):
        beta = rng.uniform(0.02, 0.98)
        h = rng.uniform(0.001, 2 * (1 + beta) - 0.001)
        M = A.heavy_ball_matrix(beta, h)
        norms = A.heavy_ball_norms(beta, h, 40)
        for k in (1, 2, 7, 25, 40):
            assert norms[k] == pytest.approx(_svd_norm_of_power(M, k), rel=1e-9, abs=1e-14), (beta, h, k)


def test_heavy_ball_spectral_radius_equals_eigvals_on_200_random_points(zufall):
    rng = zufall(13)
    for _ in range(200):
        beta = rng.uniform(0.0, 0.98)
        h = rng.uniform(0.001, 4.0)
        ref = float(np.max(np.abs(np.linalg.eigvals(A.heavy_ball_matrix(beta, h)))))
        assert A.heavy_ball_spectral_radius(beta, h) == pytest.approx(ref, rel=1e-9, abs=1e-12), (beta, h)


def test_heavy_ball_stability_region_is_0_to_2_times_1_plus_beta(zufall):
    rng = zufall(14)
    for _ in range(100):
        beta = rng.uniform(0.0, 0.98)
        edge = 2 * (1 + beta)
        assert A.heavy_ball_spectral_radius(beta, 0.999 * edge) < 1.0
        assert A.heavy_ball_spectral_radius(beta, 1.001 * edge) > 1.0
        assert A.heavy_ball_spectral_radius(beta, 0.0) == pytest.approx(1.0, abs=1e-12)   # h → 0: Eigenwert 1


def test_raster_peaks_equal_the_scalar_reference_on_a_subsample():
    """Das vektorisierte Raster gegen die Schleife mit Matrixpotenz (wie in der Vormessung, mit Abbruch bei Norm < 10⁻⁶·Spitze) an 152 Punkten (jeder 26. Rasterpunkt und der Punkt der größten Spitze)."""
    B, H = A.heavy_ball_raster()
    peaks, _ = A.heavy_ball_peaks(B, H, C.RASTER_KMAX)
    flat_idx = list(range(0, B.size, 26)) + [int(np.argmax(peaks))]
    assert len(flat_idx) == 152
    for idx in flat_idx:
        beta, h = float(B.ravel()[idx]), float(H.ravel()[idx])
        M = A.heavy_ball_matrix(beta, h)
        P = np.eye(2)
        best = 1.0
        for _ in range(C.RASTER_KMAX):
            P = P @ M
            n = float(np.linalg.norm(P, 2))
            best = max(best, n)
            if n < 1e-6 * best:
                break
        assert peaks.ravel()[idx] == pytest.approx(best, rel=1e-8), (beta, h)


# ---------------------------------------------------------------- Gradientenabstieg
def test_gd_best_step_and_rate_equal_a_scan_over_20000_steps():
    for kappa in (10.0, 100.0):
        etas = np.linspace(1e-4, 2.2 / kappa, 20000)
        rates = np.max(np.abs(1 - np.outer(etas, [1.0, kappa])), axis=1)
        eta_best, rate_best = A.gd_best(kappa)
        assert float(etas[int(np.argmin(rates))]) == pytest.approx(eta_best, abs=1.2e-5)   # Rasterweite des Scans 2,2/κ/20000 = 1,1·10⁻⁵ (κ = 10)
        assert float(rates.min()) == pytest.approx(rate_best, abs=5e-5)
        assert np.all(A.gd_rate(kappa, etas) >= rate_best - 1e-12)      # kein Schritt ist besser als der beste


def test_gd_rate_depends_only_on_the_extreme_curvatures_on_200_random_spectra(zufall):
    """Zufallsspektren mit 3 bis 8 Eigenwerten in [λmin, λmax]: die Rate max|1 − ηλ| über alle Eigenwerte ist die der beiden Extremwerte (η skaliert mit λmin)."""
    rng = zufall(21)
    for _ in range(200):
        lmin = rng.uniform(0.3, 3.0)
        kappa = rng.uniform(2.0, 100.0)
        inner = [rng.uniform(lmin, lmin * kappa) for _ in range(1 + int(rng.random() * 6))]
        lams = np.array([lmin, lmin * kappa] + inner)
        eta = rng.uniform(0.0, 1.2 * 2.0 / (lmin * kappa))
        assert float(np.max(np.abs(1 - eta * lams))) == pytest.approx(A.gd_rate(kappa, eta * lmin), rel=1e-12)


def test_gd_norm_path_equals_repeated_multiplication():
    for kappa, frac in ((10.0, 0.9), (37.0, 0.55), (100.0, 1.01)):
        eta = A.gd_eta_from_fraction(kappa, frac)
        x = np.array([1.0, 1.0])
        ref = [float(np.linalg.norm(x))]
        for _ in range(300):
            x = x - eta * np.array([1.0, kappa]) * x
            ref.append(float(np.linalg.norm(x)))
        assert A.gd_norm_path(kappa, eta, 300) == pytest.approx(ref, rel=1e-9)


# ---------------------------------------------------------------- Banach
def test_banach_fixed_point_of_cos_equals_brentq():
    optimize = pytest.importorskip("scipy.optimize")
    for key in BANACH_MAPS:
        m = BANACH_MAPS[key]
        ref = optimize.brentq(lambda x: float(m.f(np.array(x))) - x, m.lo, m.hi, xtol=1e-15, rtol=1e-15)
        assert A.banach_fixed_point(m) == pytest.approx(ref, abs=3e-15)   # f(x) − x hat Rundungsrauschen von etwa 4 Einheiten der Maschinengenauigkeit


def test_banach_constants_equal_the_hand_derivation():
    assert A.banach_fixed_point(BANACH_MAPS["affin"]) == pytest.approx(2.5, abs=1e-14)                 # 0,6·x + 1 = x  ⇒  x = 2,5
    assert A.banach_fixed_point(BANACH_MAPS["wurzel"]) == pytest.approx((1 + math.sqrt(7)) / 2, abs=1e-14)   # x² = x + 1,5
    for key, m in BANACH_MAPS.items():
        assert A.banach_lipschitz(m) == pytest.approx(m.L_exact, rel=1e-12), key
        assert A.banach_is_self_map(m), key


def test_banach_self_map_check_rejects_a_map_that_leaves_the_interval():
    from fpi_scenario import BanachMap
    bad = BanachMap("x", "x", "2x", lambda x: 2.0 * x, lambda x: np.full_like(np.asarray(x, dtype=float), 2.0), 0.0, 1.0, 2.0)
    assert not A.banach_is_self_map(bad)
    assert A.banach_lipschitz(bad) == 2.0


def test_banach_bound_holds_for_all_maps_and_all_starts(zufall):
    rng = zufall(31)
    for key, m in BANACH_MAPS.items():
        xstar = A.banach_fixed_point(m)
        L = A.banach_lipschitz(m)
        for _ in range(50):
            pos = rng.random()
            x0 = A.banach_start(m, pos)
            xs = A.banach_orbit(m, x0, 120)
            bounds = A.banach_bound(L, abs(xs[1] - xs[0]), np.arange(121))
            assert np.all(A.banach_bound_holds(np.abs(xs - xstar), bounds, xstar)), (key, pos)
