"""Kern gegen Handrechnung (DEMO-PLAYBOOK §1b und §12): die drei Handbeispiele H1 bis H3 des Plans und die Heavy-Ball-Matrix mit N² = 0.

Die Erwartungswerte stehen als Rechnung im Test; sie stammen aus der Herleitung von Hand (Norm aus Singulärwerten, Faktoren ±9/11, Lipschitz-Konstante sin 1),
nicht aus einem Lauf des Codes. Kein Parameter ist eine Zweierpotenz (a = 0,9; a + 1e-9)."""

import math

import numpy as np
import pytest

import fpi_algorithm as A
from fpi_scenario import BANACH_MAPS


def _norm_by_hand(a, b, k):
    """Norm von A^k aus den Singulärwerten: s1·s2 = p², s1² + s2² = 2p² + q²  ⇒  s1 = (√(q² + 4p²) + q)/2."""
    p = a ** k
    q = k * a ** (k - 1) * b
    return (math.sqrt(q * q + 4 * p * p) + q) / 2


# ---------------------------------------------------------------- H1: Buckel
def test_h1_second_power_by_hand():
    """A² = [[0,81; 9], [0; 0,81]] (9 = 2·0,9·5); die geschlossene Form A^k = [[p, q], [0, p]] stimmt mit der Matrixpotenz überein."""
    A2 = np.linalg.matrix_power(np.array([[0.9, 5.0], [0.0, 0.9]]), 2)
    assert A2 == pytest.approx(np.array([[0.81, 9.0], [0.0, 0.81]]))


@pytest.mark.parametrize("k, expected", [(1, 5.1571), (2, 9.0723), (3, 12.1936), (5, 16.4237), (9, 19.3788), (10, 19.3773), (20, 13.5096), (50, 1.4316), (100, 0.0148)])
def test_h1_norm_values(k, expected):
    norms = A.jordan_power_norms(0.9, 5.0, 399)
    assert norms[k] == pytest.approx(expected, abs=6e-5)
    assert norms[k] == pytest.approx(_norm_by_hand(0.9, 5.0, k), rel=1e-12)


def test_h1_singular_value_check_at_k9():
    """Probe der Herleitung: s1·s2 = p² = 0,9^18 = 0,150095 bei k = 9."""
    s = A.jordan_power_norms(0.9, 5.0, 20)[9]
    p = 0.9 ** 9
    q = 9 * 0.9 ** 8 * 5.0
    s2 = (math.sqrt(q * q + 4 * p * p) - q) / 2
    assert s * s2 == pytest.approx(0.9 ** 18, rel=1e-10) and 0.9 ** 18 == pytest.approx(0.150095, abs=5e-7)


def test_h1_summary():
    s = A.buckel_summary(0.9, 5.0)
    assert (s.k_peak, s.first_below_one) == (9, 55)
    assert s.peak == pytest.approx(19.3788, abs=5e-5)
    assert s.rho == 0.9 and s.has_buckel
    assert -1 / math.log(0.9) == pytest.approx(9.49, abs=0.005)          # Näherung des Maximums
    assert s.ratio == pytest.approx(0.0147564 / 0.9 ** 100, rel=1e-4)   # ‖A^100‖ = 1,476·10⁻², 0,9^100 = 2,656·10⁻⁵


def test_h1_without_coupling_the_norm_falls_from_the_first_step():
    s = A.buckel_summary(0.9, 0.0)
    assert (s.k_peak, s.peak, s.first_below_one, s.has_buckel) == (0, 1.0, 1, False)
    assert A.jordan_power_norms(0.9, 0.0, 30) == pytest.approx(0.9 ** np.arange(31), rel=1e-13)


# ---------------------------------------------------------------- H2: Gradientenabstieg
def test_h2_best_step_has_factors_plus_minus_9_over_11():
    eta, rate = A.gd_best(10.0)
    assert eta == pytest.approx(2 / 11) and rate == pytest.approx(9 / 11)
    f = A.gd_factors(10.0, 2 / 11)
    assert f == pytest.approx([9 / 11, -9 / 11])


def test_h2_steps_to_1e6_at_the_best_step():
    """‖x_k‖ = √2·(9/11)^k < 10⁻⁶  ⇒  k ≥ ln(√2·10⁶)/ln(11/9) = 70,57  ⇒  k = 71."""
    assert math.log(math.sqrt(2) * 1e6) / math.log(11 / 9) == pytest.approx(70.57, abs=0.005)
    assert A.gd_steps_to_tol(10.0, 2 / 11) == 71


def test_h2_step_one_over_lmax_has_rate_09_and_132_steps():
    f = A.gd_factors(10.0, 0.1)
    assert f == pytest.approx([0.9, 0.0], abs=1e-15)
    assert A.gd_steps_to_tol(10.0, 0.1) == 132


def test_h2_threshold_and_beyond():
    assert A.gd_factors(10.0, 0.2) == pytest.approx([0.8, -1.0])        # an der Schwelle: eine Richtung mit Faktor −1
    assert A.gd_threshold_state(10.0, 0.2) == "an"
    f = A.gd_factors(10.0, 0.21)
    assert f == pytest.approx([0.79, -1.1])
    assert A.gd_norm_path(10.0, 0.21, 400)[-1] == pytest.approx(1.1 ** 400, rel=1e-9) and 1.1 ** 400 == pytest.approx(3.6e16, rel=0.02)


def test_h2_the_iteration_matrix_is_symmetric_and_the_norm_has_no_hump():
    M = np.diag([1 - 2 / 11, 1 - 10 * 2 / 11])
    assert np.linalg.norm(np.linalg.matrix_power(M, 20), 2) == pytest.approx((9 / 11) ** 20, rel=1e-12)
    assert (9 / 11) ** 20 == pytest.approx(1.807160e-2, rel=1e-6)
    path = A.gd_norm_path(10.0, 2 / 11, 200)
    assert np.all(np.diff(path) <= 0)


# ---------------------------------------------------------------- H3: Banach
def test_h3_cos_constants_by_hand():
    m = BANACH_MAPS["cos"]
    assert m.lo == pytest.approx(0.5403, abs=5e-5) and m.hi == 1.0           # cos bildet [0, 1] in [cos 1, 1] ab
    assert A.banach_lipschitz(m) == pytest.approx(math.sin(1.0), rel=1e-12) and math.sin(1.0) == pytest.approx(0.841471, abs=5e-7)
    assert A.banach_is_self_map(m)


def test_h3_bound_holds_in_all_ten_steps_and_the_step_counts():
    m = BANACH_MAPS["cos"]
    xstar = 0.7390851332151607                                                  # Fixpunkt von cos (Dottie-Zahl)
    assert A.banach_fixed_point(m) == pytest.approx(xstar, abs=1e-15)
    L = math.sin(1.0)
    xs = A.banach_orbit(m, 1.0, 10)
    d01 = abs(xs[1] - xs[0])
    assert xs[1] == pytest.approx(0.540302, abs=5e-7) and d01 == pytest.approx(0.459698, abs=5e-7)
    bounds = A.banach_bound(L, d01, np.arange(11))
    errors = np.abs(xs - xstar)
    assert np.all(errors <= bounds)
    assert errors[10] == pytest.approx(5.15e-3, abs=5e-6) and bounds[10] == pytest.approx(0.516, abs=5e-4)
    assert A.banach_guaranteed_steps(L, d01) == 87
    assert A.banach_actual_steps(m, 1.0, xstar) == 32


# ---------------------------------------------------------------- Heavy Ball
def test_heavy_ball_critical_point_is_nilpotent():
    """β = 0,81, h = (1 − 0,9)² = 0,01: doppelter Eigenwert 0,9, N = M − 0,9·I erfüllt N² = 0, und M^k = ρ^k·I + k·ρ^(k−1)·N."""
    beta, rho = 0.81, 0.9
    h = A.heavy_ball_critical_h(beta)
    assert h == pytest.approx(0.01, abs=1e-15)
    M = A.heavy_ball_matrix(beta, h)
    assert M == pytest.approx(np.array([[1.8, -0.81], [1.0, 0.0]]))
    N = M - rho * np.eye(2)
    assert np.abs(N @ N).max() < 1e-15
    for k in (1, 9, 50, 399):
        assert np.linalg.matrix_power(M, k) == pytest.approx(rho ** k * np.eye(2) + k * rho ** (k - 1) * N, rel=1e-9, abs=1e-300)


def test_heavy_ball_peak_at_critical_damping():
    s = A.momentum_summary(0.81, 0.01)
    assert s.kind == "doppelt" and s.rho == pytest.approx(0.9, abs=1e-15)
    assert (s.k_peak, s.peak) == (9, pytest.approx(7.034, abs=5e-4))
    assert s.ratio == pytest.approx(201, abs=0.5)


def test_without_momentum_the_state_is_a_scalar_and_the_norm_falls_monotonically():
    for h in (0.01, 0.7, 1.9):
        n = A.heavy_ball_norms(0.0, h, 60)
        assert n == pytest.approx(abs(1 - h) ** np.arange(61), rel=1e-12)
        assert np.all(np.diff(n) <= 0)


def test_critical_h_is_the_double_root():
    for beta in (0.3, 0.7, 0.9):
        h = A.heavy_ball_critical_h(beta)
        assert (1 + beta - h) ** 2 - 4 * beta == pytest.approx(0, abs=1e-14)
        assert A.heavy_ball_kind(beta, h) == "doppelt"


def test_function_value_with_standard_start_is_monotone_even_though_the_state_norm_is_not():
    """Die Aussage „Momentum macht den Verlust nicht monoton“ ist hier NICHT belegt: bei β = 0,81, h = 0,01 steigt f nie, der Buckel steckt im Zustand."""
    f = A.heavy_ball_function_values(0.81, 0.01, 200)
    assert int(np.sum(f[1:] > f[:-1])) == 0
    assert A.heavy_ball_worst_start_peak(0.81, 0.01)[0] > 4.9


# ---------------------------------------------------------------- Experiment 1 (Zahl)
def test_affine_fixed_point_and_orbit_by_hand():
    """a = 0,8, c = 1,5: x* = 1,5/0,2 = 7,5; Start 3,7: x₁ = 0,8·3,7 + 1,5 = 4,46."""
    assert A.fixed_point_affine(0.8, 1.5) == pytest.approx(7.5)
    xs = A.affine_orbit(0.8, 1.5, 3.7, 3)
    assert xs[:3] == pytest.approx([3.7, 4.46, 5.068])
    assert abs(xs[3] - 7.5) == pytest.approx(0.8 ** 3 * abs(3.7 - 7.5), rel=1e-12)
    assert A.fixed_point_affine(1.0, 1.5) is None
