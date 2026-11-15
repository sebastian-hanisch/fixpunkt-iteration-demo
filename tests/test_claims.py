"""Claims-Tests: jede Zahl der Tabelle „Befunde“ in der README wird hier über dieselben Funktionen nachgerechnet, die die App aufruft (DEMO-PLAYBOOK §11a).

Die README-Tabelle „Zahl → Test“ nennt für jede Zahl den Test und die Art der Prüfung (Band oder exakt). Absolutaussagen („kein“, „nie“, „in allen“) bekommen ihre Gegenprobe
über das ganze Raster bzw. alle Zellen und ihre Bedingung (nur lineare Iteration, nur diese Matrizen, nur dieses Raster)."""

import math

import numpy as np
import pytest

import fpi_algorithm as A
import fpi_constants as C
import fpi_evaluation as E
from fpi_scenario import BANACH_MAPS


# ---------------------------------------------------------------- Experiment 2: Buckel
def test_claim_peak_19_379_at_k_9_and_first_below_one_55():
    s = A.buckel_summary(0.9, 5.0)
    assert s.peak == pytest.approx(19.379, abs=5e-4) and s.k_peak == 9 and s.first_below_one == 55


def test_claim_norm_at_k_100_is_556_times_the_eigenvalue_power():
    n = A.jordan_power_norms(0.9, 5.0, 399)[100]
    assert n == pytest.approx(1.476e-2, rel=1e-3) and 0.9 ** 100 == pytest.approx(2.656e-5, rel=1e-3)
    assert n / 0.9 ** 100 == pytest.approx(556, abs=0.5)


def test_claim_core_table_twelve_cells():
    """Spitze von ‖A^k‖ (k = 0 bis 399) für a ∈ {0,6; 0,9; 0,99} und b ∈ {0, 1, 5, 20}; in jeder Zelle ist ρ = a < 1."""
    expected = {
        0.6: [1.0, 1.3, 6.0, 24.0],
        0.9: [1.0, 3.9, 19.4, 77.5],
        0.99: [1.0, 37.0, 184.9, 739.5],
    }
    rows = E.core_table()
    assert [r["a"] for r in rows] == list(expected)
    for r in rows:
        assert r["a"] < 1.0
        assert r["peaks"] == pytest.approx(expected[r["a"]], abs=0.051)


def test_claim_core_table_peak_is_exactly_one_without_coupling_and_rises_with_b_in_all_rows():
    rows = E.core_table()
    for r in rows:
        assert r["peaks"][0] == 1.0
        assert all(r["peaks"][i] < r["peaks"][i + 1] for i in range(3))


def test_claim_first_k_below_one_for_the_core_cells():
    """a = 0,6: 4, 9, 12 für b = 1, 5, 20; a = 0,9: 35, 55, 70; a = 0,99: in keiner Zelle bis k = 399 (Peak bei k = 99)."""
    expected = {(0.6, 1.0): 4, (0.6, 5.0): 9, (0.6, 20.0): 12, (0.9, 1.0): 35, (0.9, 5.0): 55, (0.9, 20.0): 70,
                (0.99, 1.0): None, (0.99, 5.0): None, (0.99, 20.0): None}
    for (a, b), k in expected.items():
        assert A.buckel_summary(a, b).first_below_one == k, (a, b)
    for b in (1.0, 5.0, 20.0):
        assert A.buckel_summary(0.99, b).k_peak == 99
    assert A.jordan_power_norms(0.99, 1.0, 399)[399] > 1.0


# ---------------------------------------------------------------- Experiment 3: Gradientenabstieg
def test_claim_best_step_and_best_rate_for_kappa_10_and_100():
    for kappa, eta_5, rate_5 in ((10.0, 0.18182, 0.81818), (100.0, 0.01980, 0.98020)):
        eta, rate = A.gd_best(kappa)
        assert eta == pytest.approx(eta_5, abs=6e-6) and rate == pytest.approx(rate_5, abs=6e-6)
        etas = np.linspace(1e-4, 2.2 / kappa, 20000)
        scan = A.gd_rate(kappa, etas)
        assert float(scan.min()) == pytest.approx(rate, abs=5e-5)                       # Scan über 20 000 Schritte: 0,81819 bzw. 0,98020
        assert float(etas[int(np.argmin(scan))]) == pytest.approx(eta, abs=1.2e-5)


def test_claim_threshold_norms_after_2000_steps():
    """0,99 · 2/λmax: 2,8·10⁻¹⁸ (κ = 10) und 5,1·10⁻¹⁸ (κ = 100); genau 2/λmax: 1,000; 1,01 · 2/λmax: 1,6·10¹⁷ (beide κ); Start (1, 1)."""
    for kappa, small in ((10.0, 2.8e-18), (100.0, 5.1e-18)):
        for frac, expected in ((0.99, small), (1.0, 1.0), (1.01, 1.6e17)):
            n = A.gd_norm_path(kappa, A.gd_eta_from_fraction(kappa, frac), 2000)[-1]
            assert n == pytest.approx(expected, rel=0.03), (kappa, frac)
    assert A.gd_norm_path(10.0, A.gd_eta_from_fraction(10.0, 1.0), 2000)[-1] == pytest.approx(1.0, abs=5e-4)


def test_claim_steps_to_1e6_for_kappa_10():
    assert A.gd_steps_to_tol(10.0, 2 / 11) == 71 and A.gd_steps_to_tol(10.0, 0.1) == 132


def test_claim_no_step_beats_the_best_rate_over_200_random_condition_numbers(zufall):
    """„Die Rate (κ − 1)/(κ + 1) ist das Minimum“: Scan über 2000 Schritte je κ, 200 Zufalls-κ in [2, 100]; kein Scan-Wert liegt darunter."""
    rng = zufall(61)
    for _ in range(200):
        kappa = rng.uniform(2.0, 100.0)
        etas = np.linspace(1e-5, 2.2 / kappa, 2000)
        assert float(A.gd_rate(kappa, etas).min()) >= A.gd_best(kappa)[1] - 1e-12


# ---------------------------------------------------------------- Experiment 4: Banach
def test_claim_cos_constants_and_step_counts():
    b = E.analyse_banach("cos", 1.0, 120)
    assert b.x0 == 1.0 and b.xstar == pytest.approx(0.7390851332151607, abs=1e-15)
    assert b.L == pytest.approx(0.841471, abs=5e-7)
    assert (b.k_guaranteed, b.k_actual) == (87, 32)
    assert int(b.holds[1:].sum()) == 120


def test_claim_cos_error_rate_equals_abs_sin_xstar_to_five_digits():
    m = BANACH_MAPS["cos"]
    xs = A.banach_orbit(m, 1.0, 30)
    xstar = A.banach_fixed_point(m)
    err = np.abs(xs - xstar)
    assert err[30] / err[29] == pytest.approx(abs(math.sin(xstar)), abs=2e-5)
    assert abs(math.sin(xstar)) == pytest.approx(0.67361, abs=5e-6)


def test_claim_cos_at_ten_steps_error_and_bound():
    b = E.analyse_banach("cos", 1.0, 120)
    assert b.errors[10] == pytest.approx(5.15e-3, abs=5e-6) and b.bounds[10] == pytest.approx(0.516, abs=5e-4)
    assert b.d01 == pytest.approx(0.459698, abs=5e-7)


def test_claim_other_two_maps_step_counts():
    """0,6·x + 1, Start 1 (Lage 0,2): garantiert 28, tatsächlich 28, Rate 0,6 (= L, Schranke scharf); √(x + 1,5), Start 0,6: garantiert 16, tatsächlich 11, Rate 0,2743, L = 0,4082."""
    a = E.analyse_banach("affin", 0.2, 120)
    assert (a.k_guaranteed, a.k_actual) == (28, 28) and a.true_rate == pytest.approx(0.6) and a.L == pytest.approx(0.6)
    w = E.analyse_banach("wurzel", 0.2, 120)
    assert w.x0 == pytest.approx(0.6) and (w.k_guaranteed, w.k_actual) == (16, 11)
    assert w.true_rate == pytest.approx(0.2743, abs=5e-5) and w.L == pytest.approx(0.4082, abs=5e-5)


def test_claim_bound_without_rounding_floor_counts():
    """Ohne Rundungsrauschen gälte die Schranke bei √(x + 1,5) nur in 38 bis 41 von 400 Schritten (Lagen 0; 0,1; 0,2; 0,5; 0,77; 1), bei 0,6·x + 1 ab Lage 0,5 in keinem, bei cos in allen."""
    for pos in (0.0, 0.1, 0.2, 0.5, 0.77, 1.0):
        b = E.analyse_banach("wurzel", pos, 400)
        assert 38 <= int((b.errors[1:] <= b.bounds[1:]).sum()) <= 41, pos
    for pos in (0.5, 0.77, 1.0):
        b = E.analyse_banach("affin", pos, 400)
        assert int((b.errors[1:] <= b.bounds[1:]).sum()) == 0, pos
    for pos in (0.0, 0.2, 0.5, 1.0):
        b = E.analyse_banach("cos", pos, 400)
        assert int((b.errors[1:] <= b.bounds[1:]).sum()) == 400, pos


def test_claim_bound_holds_in_every_step_for_all_three_maps_and_five_starts():
    for key in BANACH_MAPS:
        for pos in (0.0, 0.2, 0.5, 0.77, 1.0):
            assert int(E.analyse_banach(key, pos, 400).holds[1:].sum()) == 400, (key, pos)


# ---------------------------------------------------------------- Experiment 5: Momentum
def test_claim_critical_damping_peak_7_034_at_k_9_and_factor_201():
    s = A.momentum_summary(0.81, 0.01)
    assert s.rho == pytest.approx(0.9, abs=1e-15) and s.peak == pytest.approx(7.034, abs=5e-4) and s.k_peak == 9
    assert s.ratio == pytest.approx(201, abs=0.5)
    assert np.abs((lambda N: N @ N)(A.heavy_ball_matrix(0.81, 0.01) - 0.9 * np.eye(2))).max() < 1e-15


def test_claim_raster_counts():
    """3920 stabile Punkte; 518 (13,2 %) mit Spitze über 2; keiner über 10; reell 46 von 496, komplex 472 von 3424; größte Spitze 8,3 bei β = 0,98, h = 3,9111."""
    rc = A.heavy_ball_raster_count()
    assert (rc.total, rc.above2, rc.above10) == (3920, 518, 0)
    assert 100.0 * rc.above2 / rc.total == pytest.approx(13.2, abs=0.05)
    assert (rc.real_total, rc.real_above2, rc.complex_total, rc.complex_above2) == (496, 46, 3424, 472)
    assert rc.real_total + rc.complex_total == rc.total and rc.real_above2 + rc.complex_above2 == rc.above2
    assert rc.max_peak == pytest.approx(8.3, abs=0.05) and rc.max_at == pytest.approx((0.98, 3.9111), abs=5e-5)


def test_claim_raster_count_does_not_depend_on_the_length_of_the_search():
    """Die Zählung ist stabil gegen die Länge der Spitzensuche: k bis 300, 600 und 2000 geben dieselben Zahlen."""
    counts = {kmax: A.heavy_ball_raster_count(kmax) for kmax in (300, 600, 2000)}
    assert counts[300] == counts[600] == counts[2000]


def test_claim_all_raster_points_are_stable():
    B, H = A.heavy_ball_raster()
    assert B.shape == H.shape == (49, 80)
    assert np.all(A.heavy_ball_spectral_radius(B, H) < 1.0)
    assert B[0, 0] == pytest.approx(0.02) and B[-1, 0] == pytest.approx(0.98)


def test_claim_function_value_is_monotone_and_the_worst_start_peak_is_4_986():
    f = A.heavy_ball_function_values(0.81, 0.01, 200)
    assert int(np.sum(f[1:] > f[:-1])) == 0
    value, k = A.heavy_ball_worst_start_peak(0.81, 0.01)
    assert value == pytest.approx(4.986, abs=5e-4) and k == 9


def test_claim_without_momentum_no_hump_for_every_h():
    """Ohne Momentum (β = 0) fällt die Norm für h = 0,01; 0,3; 0,7 und 1,9 monoton."""
    for h in (0.01, 0.3, 0.7, 1.9):
        assert np.all(np.diff(A.heavy_ball_norms(0.0, h, 60)) <= 0)


# ---------------------------------------------------------------- Zweierpotenz-Falle
def test_claim_power_of_two_parameters_change_nothing_for_the_hump():
    """a = 0,9 und a + 10⁻⁹ geben dieselbe Spitze bis auf die Verschiebung durch die Störung (relative Abweichung unter 10⁻⁶): kein Sonderfall durch exakte Dualbrüche."""
    s0, s1 = A.buckel_summary(0.9, 5.0), A.buckel_summary(0.9 + 1e-9, 5.0)
    assert (s0.k_peak, s0.first_below_one) == (s1.k_peak, s1.first_below_one)
    assert s1.peak == pytest.approx(s0.peak, rel=1e-6)


def test_claim_compute_budget_of_the_live_calculation():
    """Alle fünf Rechnungen zusammen und das Raster brauchen weit unter 2 Sekunden (Timeout statt Zeitmessung: 20 Wiederholungen des schwersten Falls, Grenze 20 s)."""
    import time
    start = time.perf_counter()
    for _ in range(20):
        E.analyse_buckel(0.99, 20.0)
        E.analyse_gd(100.0, 0.05)
        E.analyse_banach("cos", 1.0, C.K_STEPS_MAX)
        E.analyse_momentum(0.98, 3.9, C.K_STEPS_MAX)
    A.heavy_ball_raster_count()
    assert time.perf_counter() - start < 20.0
