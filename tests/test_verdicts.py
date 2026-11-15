"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) in `fpi_evaluation.py`; die App ruft st.<Art>(Text). Die Zahlen im Text stammen aus der Handrechnung
(x* = c/(1 − a), Faktoren 1 − ηλ, Norm aus Singulärwerten), nicht aus einem Lauf des Verdikts."""

import dataclasses


import fpi_evaluation as E


# ---------------------------------------------------------------- Experiment 1
def test_affine_converging_names_the_fixed_point_and_the_factor():
    kind, text = E.verdict_affine(E.analyse_affine(0.8, 1.5, 3.7, 30))            # x* = 1,5/0,2 = 7,5
    assert kind == "success"
    assert "|a| = 0,80 < 1" in text and "x* = c/(1 − a) = 7,50" in text and "Faktor 0,80" in text and "wechselndem" not in text


def test_affine_negative_factor_mentions_the_sign_change():
    kind, text = E.verdict_affine(E.analyse_affine(-0.8, 1.5, 3.7, 30))           # x* = 1,5/1,8 = 0,8333
    assert kind == "success" and "mit wechselndem Vorzeichen" in text and "x* = c/(1 − a) = 0,83" in text


def test_affine_start_in_the_fixed_point_stays():
    kind, text = E.verdict_affine(E.analyse_affine(0.8, 1.5, 7.5, 30))
    assert kind == "success" and "selbst der Ruhepunkt" in text and "7,50" in text


def test_affine_boundary_cases_are_warnings_with_their_own_text():
    k1, t1 = E.verdict_affine(E.analyse_affine(1.0, 0.0, 3.7, 30))
    assert k1 == "warning" and "Jeder Punkt ist Ruhepunkt" in t1 and "3,70" in t1
    k2, t2 = E.verdict_affine(E.analyse_affine(1.0, 1.5, 3.7, 30))
    assert k2 == "warning" and "um c = 1,50 je Schritt" in t2
    k3, t3 = E.verdict_affine(E.analyse_affine(-1.0, 1.5, 3.7, 30))               # Zyklus 3,7 ↔ c − x₀ = −2,2; c/2 = 0,75
    assert k3 == "warning" and "3,70 und c − x₀ = −2,20" in t3 and "0,75" in t3


def test_affine_diverging_names_the_factor_and_the_repelling_fixed_point():
    kind, text = E.verdict_affine(E.analyse_affine(1.2, 1.5, 3.7, 30))            # x* = 1,5/(−0,2) = −7,5
    assert kind == "warning" and "|a| = 1,20 > 1" in text and "x* = −7,50" in text and "Faktor 1,20" in text
    kind, text = E.verdict_affine(E.analyse_affine(1.2, 1.5, -7.5, 30))
    assert kind == "warning" and "nur der Start genau auf x*" in text


# ---------------------------------------------------------------- Experiment 2
def test_buckel_verdict_names_peak_k_first_below_one_and_ratio():
    kind, text = E.verdict_buckel(E.analyse_buckel(0.9, 5.0))
    assert kind == "warning"
    assert "ρ = 0,90 < 1" in text and "auf 19,38 bei k = 9" in text and "ab k = 55" in text and "das 556-Fache von ρ^100" in text


def test_buckel_verdict_when_the_norm_never_returns_below_one():
    kind, text = E.verdict_buckel(E.analyse_buckel(0.99, 1.0))
    assert kind == "warning" and "bei k = 99" in text and "bis k = 399 liegt sie nicht wieder unter 1" in text


def test_buckel_verdict_without_coupling_and_with_small_coupling():
    kind, text = E.verdict_buckel(E.analyse_buckel(0.9, 0.0))
    assert kind == "success" and "diagonal, also normal" in text and "0,90^k" in text
    kind, text = E.verdict_buckel(E.analyse_buckel(0.6, 0.6))                      # nicht normal, aber Spitze nur bei k = 0
    assert kind == "success" and "ρ = 0,60 < 1 und b = 0,60 ≠ 0" in text and "Spitze bei k = 0" in text


def test_buckel_verdict_at_and_above_rho_one():
    k, t = E.verdict_buckel(E.analyse_buckel(1.0, 0.0))
    assert k == "warning" and "Einheitsmatrix" in t
    k, t = E.verdict_buckel(E.analyse_buckel(1.0, 5.0))
    assert k == "warning" and "wächst auf lange Sicht linear (ungefähr k·b)" in t and "b = 5,00" in t
    k, t = E.verdict_buckel(E.analyse_buckel(1.1, 5.0))
    assert k == "warning" and "ρ = 1,10 > 1" in t


def test_buckel_verdict_for_the_nilpotent_matrix_has_no_ratio_sentence():
    kind, text = E.verdict_buckel(E.analyse_buckel(0.0, 5.0))                      # ‖A¹‖ = 5, ab k = 2 null
    assert kind == "warning" and "auf 5,00 bei k = 1" in text and "ab k = 2" in text and "-Fache" not in text


# ---------------------------------------------------------------- Experiment 3
def test_gd_below_threshold_names_rate_steps_and_best_step():
    g = E.analyse_gd(10.0, 10 / 11)                                              # bester Schritt: Faktoren ±9/11, 71 Schritte (Handrechnung H2)
    kind, text = E.verdict_gd(g)
    assert kind == "success"
    assert "Rate max|1 − ηλ| = 0,8182" in text and "nach 71 Schritten" in text and "Bester Schritt 2/(λmin + λmax) = 0,1818" in text and "(κ − 1)/(κ + 1) = 0,8182" in text
    assert "(0,909 der Schwelle)" in text


def test_gd_limiting_direction_switches_at_the_best_step():
    assert "flachen Richtung" in E.verdict_gd(E.analyse_gd(10.0, 0.9))[1]           # Faktoren 0,82 und −0,80: der flache Faktor ist größer
    assert "steilen Richtung (Krümmung 10)" in E.verdict_gd(E.analyse_gd(10.0, 0.95))[1]    # 0,81 und −0,90


def test_gd_at_and_above_the_threshold_are_warnings():
    kind, text = E.verdict_gd(E.analyse_gd(10.0, 1.0))
    assert kind == "warning" and "genau an der Schwelle 2/λmax = 0,2000" in text and "Faktor −1" in text
    kind, text = E.verdict_gd(E.analyse_gd(10.0, 1.05))                            # η = 0,21: Faktor 1 − 2,1 = −1,1
    assert kind == "warning" and "η = 0,2100 liegt über der Schwelle 2/λmax = 0,2000" in text and "−1,100" in text and "Faktor 1,100" in text


def test_gd_without_convergence_in_the_search_window_says_so():
    g = dataclasses.replace(E.analyse_gd(100.0, 0.05), steps=None)
    kind, text = E.verdict_gd(g)
    assert kind == "success" and "mehr als 20000 Schritten" in text


# ---------------------------------------------------------------- Experiment 4
def test_banach_cos_verdict_names_all_numbers():
    kind, text = E.verdict_banach(E.analyse_banach("cos", 1.0, 120))                # Handrechnung H3: L = sin 1, 87 garantiert, 32 tatsächlich
    assert kind == "success"
    assert "L = 0,8415" in text and "in 120 von 120 Schritten" in text and "garantiert die Schranke 87 Schritte" in text and "tatsächlich genügen 32" in text
    assert "|f′(x*)| = 0,6736 liegt unter L" in text and "[0,540; 1,000]" in text


def test_banach_affine_bound_is_sharp():
    kind, text = E.verdict_banach(E.analyse_banach("affin", 0.2, 120))              # Start 1: d01 = 0,6, Schranke 0,6^k·1,5 = Fehler
    assert kind == "success" and "ist gleich L: bei dieser Abbildung ist die Schranke scharf" in text and "garantiert die Schranke 28 Schritte, tatsächlich genügen 28" in text


def test_banach_verdict_refuses_without_the_assumptions():
    b = E.analyse_banach("cos", 1.0, 30)
    kind, text = E.verdict_banach(dataclasses.replace(b, self_map=False))
    assert kind == "warning" and "nicht erfüllt" in text
    assert E.verdict_banach(dataclasses.replace(b, L=1.0))[0] == "warning"


def test_banach_verdict_without_convergence_in_the_window():
    b = dataclasses.replace(E.analyse_banach("cos", 1.0, 30), k_actual=None)
    assert "mehr als 10000" in E.verdict_banach(b)[1]


# ---------------------------------------------------------------- Experiment 5
def test_momentum_critical_damping_verdict():
    kind, text = E.verdict_momentum(E.analyse_momentum(0.81, 0.01, 120))
    assert kind == "warning"
    assert "ρ = 0,900 < 1" in text and "doppelter Eigenwert (kritisch gedämpft)" in text and "auf 7,03 bei k = 9" in text and "das 201-Fache von ρ^100" in text
    assert "nicht den Funktionswert" in text


def test_momentum_without_momentum_has_no_hump():
    kind, text = E.verdict_momentum(E.analyse_momentum(0.0, 0.4, 120))              # Skalar 1 − h = 0,6
    assert kind == "success" and "ρ = 0,600 < 1" in text and "kein Buckel" in text


def test_momentum_unstable_names_the_stability_interval():
    kind, text = E.verdict_momentum(E.analyse_momentum(0.6, 3.5, 120))              # 2(1 + β) = 3,2
    assert kind == "warning" and "(0; 3,20)" in text and "konvergiert nicht" in text


def test_momentum_eigenvalue_types_are_named():
    assert "zwei reelle Eigenwerte" in E.verdict_momentum(E.analyse_momentum(0.6, 0.01, 120))[1]            # Diskriminante 1,59² − 2,4 = 0,128 > 0
    assert "ein komplexes Eigenwertpaar" in E.verdict_momentum(E.analyse_momentum(0.6, 0.1, 120))[1]            # Diskriminante 1,5² − 2,4 < 0


# ---------------------------------------------------------------- Kernfrage
def test_core_verdict_counts_the_rows():
    rows = [{"a": 0.6, "peaks": [1.0, 2.0, 3.0, 4.0]}, {"a": 0.9, "peaks": [1.0, 5.0, 9.0, 77.5]}, {"a": 0.99, "peaks": [1.0, 36.0, 184.0, 739.4]}]
    kind, text = E.verdict_core(rows)
    assert kind == "info"
    assert "In allen 12 Zellen" in text and "(3 von 3 Zeilen) und steigt mit b (3 von 3 Zeilen streng): bis 739,4 bei a = 0,99, b = 20" in text


def test_core_verdict_names_a_row_that_does_not_rise():
    rows = [{"a": 0.6, "peaks": [1.0, 2.0, 3.0, 4.0]}, {"a": 0.9, "peaks": [1.0, 5.0, 5.0, 6.0]}, {"a": 0.99, "peaks": [1.1, 36.0, 184.0, 739.4]}]
    text = E.verdict_core(rows)[1]
    assert "genau bei 1 (2 von 3 Zeilen)" in text and "steigt mit b (2 von 3 Zeilen streng)" in text      # Zeile 1 hat Spitze 1,1 bei b = 0, Zeile 2 steigt nicht streng


# ---------------------------------------------------------------- Zahlenformat
def test_number_format_uses_decimal_comma_and_minus_sign():
    assert E.de(1234.5678, 2) == "1234,57" and E.de(-0.25, 2) == "−0,25"
    assert E.de_sig(555.557, 3) == "556" and E.de_sig(0.0147564, 3) == "0,0148" and E.de_sig(1.586e17, 3) == "1,59·10^17" and E.de_sig(0, 3) == "0"
