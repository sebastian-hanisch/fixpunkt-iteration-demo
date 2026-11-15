"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): ρ = 1, b = 0, nilpotent (a = 0), Schritt genau an der Schwelle (Float-Reste), κ = 1,
Start im Fixpunkt, Regler-Ränder, Momentum an der Stabilitätsgrenze und ohne Momentum."""

import math

import numpy as np
import pytest

import fpi_algorithm as A
import fpi_constants as C
import fpi_evaluation as E
from fpi_scenario import BANACH_MAPS


# ---------------------------------------------------------------- Experiment 2
def test_rho_one_without_coupling_the_norm_stays_one():
    n = A.jordan_power_norms(1.0, 0.0, 399)
    assert n == pytest.approx(np.ones(400), abs=1e-13)
    s = A.buckel_summary(1.0, 0.0)
    assert s.first_below_one is None and not s.has_buckel and s.k_peak == 0


def test_rho_one_with_coupling_grows_linearly_like_k_times_b():
    """A^k = [[1, k·b], [0, 1]]: die Norm liegt zwischen k·b und k·b + 1/(k·b) (aus s1 = (√(q² + 4) + q)/2)."""
    for b in (0.1, 1.0, 5.0):
        n = A.jordan_power_norms(1.0, b, 399)
        for k in (10, 100, 399):
            q = k * b
            assert q <= n[k] <= q + 1.0 / q + 1e-12
    assert A.buckel_summary(1.0, 5.0).first_below_one is None


def test_nilpotent_matrix_is_zero_from_k_equal_2():
    """a = 0: A = [[0, b], [0, 0]], A² = 0; die Norm ist b bei k = 1 und danach null (kein Fehler durch 0^0 oder 0^(−1))."""
    n = A.jordan_power_norms(0.0, 5.0, 10)
    assert n[0] == 1.0 and n[1] == pytest.approx(5.0) and np.all(n[2:] == 0.0)
    s = A.buckel_summary(0.0, 5.0)
    assert (s.k_peak, s.first_below_one, s.ratio) == (1, 2, None) and s.has_buckel
    z = A.buckel_summary(0.0, 0.0)          # Nullmatrix: schon A¹ hat die Norm 0
    assert (z.k_peak, z.peak, z.first_below_one) == (0, 1.0, 1)


def test_small_coupling_gives_no_hump_even_though_a_is_not_normal():
    """a = 0,6, b = 0,6: ‖A‖ = (√(q² + 4p²) + q)/2 mit p = q = 0,6 gibt 0,971 < 1 – nicht normal, aber kein Buckel."""
    assert (math.sqrt(0.6 ** 2 + 4 * 0.6 ** 2) + 0.6) / 2 == pytest.approx(0.971, abs=5e-4)
    s = A.buckel_summary(0.6, 0.6)
    assert not s.has_buckel and s.k_peak == 0


def test_window_shorter_than_the_ratio_step_gives_no_ratio():
    s = A.buckel_summary(0.9, 5.0, kmax=50)
    assert s.ratio is None and s.k_peak == 9 and s.first_below_one is None     # bis k = 50 liegt die Norm noch über 1


def test_slider_extremes_do_not_overflow():
    for a in (C.E2_A_MIN, C.E2_A_MAX):
        for b in (C.E2_B_MIN, C.E2_B_MAX):
            n = A.jordan_power_norms(a, b, C.KMAX_ANALYSIS)
            assert np.all(np.isfinite(n)) and np.all(n >= 0)


# ---------------------------------------------------------------- Experiment 3
def test_step_exactly_at_the_threshold_is_recognised_in_spite_of_float_residue():
    for kappa in (3.0, 7.0, 10.0, 37.0, 100.0):
        eta = A.gd_eta_from_fraction(kappa, 1.0)
        assert A.gd_threshold_state(kappa, eta) == "an", kappa
        assert A.gd_threshold_state(kappa, eta * 0.999) == "unter"
        assert A.gd_threshold_state(kappa, eta * 1.001) == "über"
        assert A.gd_steps_to_tol(kappa, eta) is None          # keine Konvergenz an der Schwelle
        assert A.gd_norm_path(kappa, eta, 2000)[-1] == pytest.approx(1.0, abs=1e-6)    # die steile Komponente springt zwischen +1 und −1


def test_condition_number_one_converges_in_one_step():
    eta, rate = A.gd_best(1.0)
    assert (eta, rate) == (1.0, 0.0)
    assert A.gd_steps_to_tol(1.0, 1.0) == 1


def test_slider_extremes_of_the_step_fraction():
    for kappa in (C.E3_KAPPA_MIN, C.E3_KAPPA_MAX):
        for frac in (C.E3_FRAC_MIN, C.E3_FRAC_MAX):
            g = E.analyse_gd(kappa, frac)
            assert np.isfinite(g.rate)
            assert g.state == ("über" if frac > 1.0 else "unter")
    assert E.analyse_gd(100.0, 0.05).steps is None or E.analyse_gd(100.0, 0.05).steps > 5000   # sehr kleiner Schritt: langsam, die Suche bricht bei 20 000 ab


# ---------------------------------------------------------------- Experiment 4
def test_start_in_the_fixed_point_has_zero_distance():
    m = BANACH_MAPS["affin"]
    xstar = A.banach_fixed_point(m)
    pos = (xstar - m.lo) / (m.hi - m.lo)
    b = E.analyse_banach("affin", pos, 30)
    assert b.d01 == pytest.approx(0.0, abs=1e-14)
    assert A.banach_guaranteed_steps(b.L, 0.0) == 0
    kind, text = E.verdict_banach(E.analyse_banach("affin", 0.5, 30))          # Lage 0,5: x₀ = 2,5 = Fixpunkt
    assert kind == "info" and "selbst der Fixpunkt" in text


def test_start_at_both_interval_ends_is_inside_the_interval_for_every_map():
    for key, m in BANACH_MAPS.items():
        for pos in (0.0, 1.0):
            b = E.analyse_banach(key, pos, 60)
            assert m.lo <= b.x0 <= m.hi
            assert int(b.holds[1:].sum()) == 60


def test_the_bound_without_rounding_floor_fails_at_large_k():
    """Die Schranke fällt unter jede Maschinengenauigkeit, der gemessene Fehler nicht (hier √(x + 1,5): Fehler bleibt bei 2·10⁻¹⁶ bis 4·10⁻¹⁶): ohne das Rundungsrauschen
    gälte sie in 400 Schritten nicht überall (gemessen: in 38 bis 41 von 400), mit ihm in allen 400, für jede Abbildung und jede Lage des Starts."""
    for pos in (0.0, 0.2, 0.5, 1.0):
        b = E.analyse_banach("wurzel", pos, 400)
        assert int((b.errors[1:] <= b.bounds[1:]).sum()) < 400
        assert int(b.holds[1:].sum()) == 400
    for key in BANACH_MAPS:
        for pos in (0.0, 0.2, 0.5, 0.77, 1.0):
            assert int(E.analyse_banach(key, pos, 400).holds[1:].sum()) == 400, (key, pos)


# ---------------------------------------------------------------- Experiment 5
def test_momentum_at_the_stability_edge_is_unstable_by_definition():
    for beta in (0.0, 0.5 + 1e-9, 0.9):
        h = 2 * (1 + beta)
        assert A.heavy_ball_spectral_radius(beta, h) == pytest.approx(1.0, abs=1e-12)
        assert A.heavy_ball_kind(beta, h) == "instabil"


def test_momentum_beta_zero_and_critical_h_is_one():
    """β = 0: der Zustand ist ein Skalar, h = (1 − √0)² = 1 gibt den Faktor 0 (ein Schritt ans Ziel)."""
    assert A.heavy_ball_critical_h(0.0) == 1.0
    n = A.heavy_ball_norms(0.0, 1.0, 5)
    assert n[1] == 0.0 and n[0] == 1.0


def test_momentum_summary_for_unstable_point_has_no_ratio():
    s = A.momentum_summary(0.5 + 1e-9, 5.0)
    assert s.kind == "instabil" and s.ratio is None and s.rho > 1.0


def test_critical_h_button_value_is_clamped_to_the_slider_range():
    import fpi_presets as P
    assert P.critical_h_for_slider(0.81) == pytest.approx(0.01, abs=1e-15)
    assert P.critical_h_for_slider(0.98) == C.E5_H_MIN         # (1 − √0,98)² = 10⁻⁴ liegt unter dem kleinsten Reglerwert
    assert P.critical_h_for_slider(0.0) == 1.0
    for beta in np.linspace(0.0, 0.98, 50):
        assert C.E5_H_MIN <= P.critical_h_for_slider(float(beta)) <= C.E5_H_MAX


# ---------------------------------------------------------------- Experiment 1
def test_affine_boundary_cases():
    assert E.analyse_affine(1.0, 0.0, 3.7, 10).kind == "grenzfall"
    assert E.analyse_affine(-1.0, 1.5, 3.7, 10).kind == "grenzfall"
    assert E.analyse_affine(1.0 + 1e-9, 1.5, 3.7, 10).kind == "divergiert"
    assert E.analyse_affine(1.0 - 1e-9, 1.5, 3.7, 10).kind == "konvergiert"
    xs = E.analyse_affine(-1.0, 1.5, 3.7, 4).xs
    assert xs == pytest.approx([3.7, -2.2, 3.7, -2.2, 3.7])         # Zyklus der Länge 2: x₀ und c − x₀


def test_cobweb_stops_when_the_path_leaves_the_window():
    xs, ys = A.cobweb_points(1.5, 1.0, 3.7, 25, -5.0, 5.0)
    assert len(xs) == len(ys) and all(-5.0 <= v <= 5.0 for v in xs + ys)
    assert len(xs) < 1 + 2 * 25
