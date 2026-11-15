"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für die Verfahren gelten müssen, ohne dass man das Ergebnis kennt.

Je Eigenschaft 50 Zufallsinstanzen mit festem Zufalls-Seed (SplitMix64), dazu Randfälle in test_edge_cases.py. Jede Eigenschaft nennt ihre Voraussetzung."""

import math

import numpy as np
import pytest

import fpi_algorithm as A
import fpi_evaluation as E


def _random_ab(rng):
    return rng.uniform(0.0, 1.2), rng.uniform(0.0, 10.0)


def test_norm_is_never_below_the_spectral_radius_power(zufall):
    """‖A^k‖₂ ≥ ρ(A)^k für jede Matrix (Eigenwert-Eigenvektor-Paar): hier A = [[a, b], [0, a]], 50 zufällige (a, b), k = 0 bis 399 (Rundung: relative Toleranz 10⁻¹²)."""
    rng = zufall(41)
    ks = np.arange(400)
    for _ in range(50):
        a, b = _random_ab(rng)
        n = A.jordan_power_norms(a, b, 399)
        assert np.all(n >= np.power(a, ks) * (1 - 1e-12)), (a, b)


def test_scaling_the_matrix_scales_the_norm_of_the_powers_with_c_to_the_k(zufall):
    """A → c·A (c > 0): ‖(cA)^k‖ = c^k·‖A^k‖; in der Parametrisierung (a, b) → (c·a, c·b). Gerechnet mit k ≤ 60, damit c^k nicht überläuft."""
    rng = zufall(42)
    ks = np.arange(61)
    for _ in range(50):
        a, b = rng.uniform(0.1, 1.0), rng.uniform(0.0, 5.0)
        c = rng.uniform(0.5, 1.5)
        left = A.jordan_power_norms(c * a, c * b, 60)
        right = np.power(c, ks) * A.jordan_power_norms(a, b, 60)
        assert left == pytest.approx(right, rel=1e-9), (a, b, c)


def test_a_normal_matrix_has_no_hump_the_norm_equals_the_eigenvalue_power(zufall):
    """b = 0 (diagonal, normal): ‖A^k‖ = |a|^k. Nur für normale Matrizen; für b ≠ 0 gilt es nicht (nächster Test)."""
    rng = zufall(43)
    ks = np.arange(300)
    for _ in range(50):
        a = rng.uniform(0.05, 1.2)
        assert A.jordan_power_norms(a, 0.0, 299) == pytest.approx(np.power(a, ks), rel=1e-12)
        assert not A.buckel_summary(a, 0.0).has_buckel or a > 1.0


def test_the_hump_grows_with_the_coupling(zufall):
    """Für festes a in (0, 1) steigt die Spitze von ‖A^k‖ streng mit b (b > 0): 50 zufällige a, b₁ < b₂."""
    rng = zufall(44)
    for _ in range(50):
        a = rng.uniform(0.3, 0.99)
        b1 = rng.uniform(0.5, 4.0)
        b2 = b1 + rng.uniform(0.5, 4.0)
        assert A.buckel_summary(a, b1).peak < A.buckel_summary(a, b2).peak, (a, b1, b2)


def test_the_two_by_two_norm_is_invariant_under_rotations_and_homogeneous(zufall):
    rng = zufall(45)
    for _ in range(50):
        P = np.array([[rng.uniform(-2, 2) for _ in range(2)] for _ in range(2)])
        t, u = rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)
        Q1 = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        Q2 = np.array([[math.cos(u), -math.sin(u)], [math.sin(u), math.cos(u)]])
        R = Q1 @ P @ Q2
        base = float(A.norm2_2x2(*P.ravel()))
        assert float(A.norm2_2x2(*R.ravel())) == pytest.approx(base, rel=1e-10)
        assert float(A.norm2_2x2(*(3.0 * P).ravel())) == pytest.approx(3.0 * base, rel=1e-12)


def test_affine_orbit_distance_to_the_fixed_point_is_linear_in_the_start(zufall):
    """x_k − x* = a^k·(x₀ − x*): der Abstand skaliert linear mit dem Startabstand (|a| ≠ 1; 50 Zufallsfälle, Toleranz für Rundung relativ 10⁻⁹ des Startabstands)."""
    rng = zufall(46)
    for _ in range(50):
        a, c = rng.uniform(-0.99, 0.99), rng.uniform(-3, 3)
        xstar = A.fixed_point_affine(a, c)
        x0 = rng.uniform(-10, 10)
        lam = rng.uniform(0.2, 3.0)
        d1 = A.affine_orbit(a, c, xstar + (x0 - xstar), 20) - xstar
        d2 = A.affine_orbit(a, c, xstar + lam * (x0 - xstar), 20) - xstar
        assert d2 == pytest.approx(lam * d1, abs=1e-9 * max(1.0, abs(x0 - xstar) * lam))


def test_gd_rate_is_symmetric_around_the_best_step_in_the_two_directions(zufall):
    """Unter dem besten Schritt begrenzt die flache Richtung (Faktor 1 − η), darüber die steile (|1 − ηκ|); beim besten Schritt sind beide gleich groß (50 Zufalls-κ)."""
    rng = zufall(47)
    for _ in range(50):
        kappa = rng.uniform(2.0, 100.0)
        eta_best, rate_best = A.gd_best(kappa)
        assert abs(1 - eta_best) == pytest.approx(abs(1 - eta_best * kappa), rel=1e-12)
        assert abs(1 - eta_best) == pytest.approx(rate_best, rel=1e-12)
        lo, hi = 0.9 * eta_best, 1.1 * eta_best
        assert abs(1 - lo) > abs(1 - lo * kappa) and abs(1 - hi * kappa) > abs(1 - hi)
        assert A.gd_rate(kappa, lo) > rate_best and A.gd_rate(kappa, hi) > rate_best


def test_gd_iteration_matrix_is_symmetric_so_the_norm_never_rises_below_the_threshold(zufall):
    """Unter der Schwelle (η < 2/λmax) fällt ‖x_k‖ in jedem Schritt (symmetrische Matrix, ohne Momentum): 50 Zufallsfälle, Start (1, 1)."""
    rng = zufall(48)
    for _ in range(50):
        kappa = rng.uniform(2.0, 100.0)
        frac = rng.uniform(0.05, 0.99)
        path = A.gd_norm_path(kappa, A.gd_eta_from_fraction(kappa, frac), 200)
        assert np.all(np.diff(path) <= 1e-15), (kappa, frac)


def test_banach_error_decreases_by_at_least_the_factor_L_per_step(zufall):
    """|x_(k+1) − x*| ≤ L·|x_k − x*| für jede Selbstabbildung mit Lipschitz-Konstante L auf dem Intervall (50 Zufallsstarts je Abbildung; Rundungsrauschen 4 Maschinengenauigkeiten)."""
    from fpi_scenario import BANACH_MAPS
    rng = zufall(49)
    for key, m in BANACH_MAPS.items():
        xstar, L = A.banach_fixed_point(m), A.banach_lipschitz(m)
        floor = A.banach_rounding_floor(xstar)
        for _ in range(50):
            xs = A.banach_orbit(m, A.banach_start(m, rng.random()), 60)
            err = np.abs(xs - xstar)
            assert np.all(err[1:] <= L * err[:-1] + floor), key


def test_without_momentum_the_state_norm_is_monotone_and_with_momentum_the_peak_exists_in_stable_complex_cases(zufall):
    """β = 0: für jedes h in (0, 2) fällt die Norm monoton. β > 0 mit komplexen Eigenwerten: mindestens ein Fall mit Spitze über 1 unter 50 Zufallspunkten (Vorzeichen der Aussage: es gibt Buckel, nicht „immer“)."""
    rng = zufall(50)
    for _ in range(50):
        h = rng.uniform(0.01, 1.99)
        assert np.all(np.diff(A.heavy_ball_norms(0.0, h, 100)) <= 1e-15)
    humps = 0
    for _ in range(50):
        beta = rng.uniform(0.3, 0.98)
        h = rng.uniform(0.01, 2 * (1 + beta) * 0.99)
        if A.heavy_ball_spectral_radius(beta, h) < 1 and A.momentum_summary(beta, h).peak > 1.0 + 1e-12:
            humps += 1
    assert humps >= 1


def test_same_input_gives_the_same_output():
    s1, s2 = E.analyse_buckel(0.9, 5.0), E.analyse_buckel(0.9, 5.0)
    assert np.array_equal(s1.norms, s2.norms) and s1.summary == s2.summary
    assert A.heavy_ball_raster_count() == A.heavy_ball_raster_count()
