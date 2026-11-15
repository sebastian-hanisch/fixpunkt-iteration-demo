"""Kennzahlen, Verdikt-Logik und Tabellen (DEMO-PLAYBOOK §12: jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text)).

Zahlen in Texten haben ein Dezimalkomma (`de`). Jede Zahl, die ein Verdikt nennt, kommt aus der Kennzahl-Struktur, die der Kern liefert."""

import math
from dataclasses import dataclass

import numpy as np

import fpi_algorithm as A
import fpi_constants as C
from fpi_scenario import BANACH_MAPS

TOL_ONE = 1e-12   # |a| gilt als 1, wenn es sich um weniger als diese Toleranz unterscheidet (der Regler stellt 1,00 exakt ein)


def de(x: float, nd: int = 2) -> str:
    """Zahl mit Dezimalkomma und nd Nachkommastellen (Minus als „−“)."""
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def de_sig(x: float, sig: int = 3) -> str:
    """Zahl mit sig gültigen Stellen, bei sehr großen oder kleinen Beträgen in Zehnerpotenz-Schreibweise."""
    if x == 0:
        return "0"
    if abs(x) >= 1e5 or abs(x) < 1e-3:
        mant, exp = f"{x:.{sig - 1}e}".split("e")
        return f"{mant.replace('.', ',')}·10^{int(exp)}".replace("-", "−")
    digits = max(sig - 1 - int(math.floor(math.log10(abs(x)))), 0)
    return de(x, digits)


@dataclass(frozen=True)
class Settings:
    k_steps: int = C.DEFAULT_K_STEPS
    e1_a: float = C.DEFAULT_E1_A
    e1_c: float = C.DEFAULT_E1_C
    e1_x0: float = C.DEFAULT_E1_X0
    e2_a: float = C.DEFAULT_E2_A
    e2_b: float = C.DEFAULT_E2_B
    e3_kappa: float = C.DEFAULT_E3_KAPPA
    e3_frac: float = C.DEFAULT_E3_FRAC
    e4_map: str = C.DEFAULT_E4_MAP
    e4_pos: float = C.DEFAULT_E4_POS
    e5_beta: float = C.DEFAULT_E5_BETA
    e5_h: float = C.DEFAULT_E5_H


# ============================================================ Experiment 1 =================================================================
@dataclass(frozen=True)
class AffineAnalysis:
    a: float
    c: float
    x0: float
    xs: np.ndarray             # x_0 … x_K
    xstar: float | None        # c/(1 − a); None bei a = 1
    kind: str                  # konvergiert | grenzfall | divergiert


def analyse_affine(a: float, c: float, x0: float, steps: int) -> AffineAnalysis:
    kind = "konvergiert" if abs(a) < 1.0 - TOL_ONE else ("grenzfall" if abs(a) <= 1.0 + TOL_ONE else "divergiert")
    return AffineAnalysis(a, c, x0, A.affine_orbit(a, c, x0, steps), A.fixed_point_affine(a, c), kind)


def verdict_affine(an: AffineAnalysis) -> tuple[str, str]:
    a, c, x0 = an.a, an.c, an.x0
    if an.kind == "konvergiert":
        vz = " mit wechselndem Vorzeichen" if a < 0 else ""
        if abs(x0 - an.xstar) < 1e-12:
            return "success", f"|a| = {de(abs(a))} < 1, und der Start x₀ = {de(x0)} ist selbst der Ruhepunkt x* = c/(1 − a) = {de(an.xstar)}: die Bahn bleibt dort."
        return "success", (f"|a| = {de(abs(a))} < 1: Die Bahn geht für jeden Startwert gegen den Ruhepunkt x* = c/(1 − a) = {de(an.xstar)}{vz}; "
                           f"der Abstand zu x* schrumpft je Schritt um den Faktor {de(abs(a))}.")
    if an.kind == "grenzfall":
        if a > 0:
            if abs(c) < 1e-12:
                return "warning", f"|a| = 1 mit c = 0: Jeder Punkt ist Ruhepunkt, die Bahn bleibt bei x₀ = {de(x0)}; es gibt keinen von x₀ unabhängigen Grenzwert."
            return "warning", f"a = 1 mit c ≠ 0: Es gibt keinen Ruhepunkt, die Bahn wandert um c = {de(c)} je Schritt (linear), ohne Grenzwert."
        return "warning", (f"a = −1: Die Bahn springt zwischen x₀ = {de(x0)} und c − x₀ = {de(c - x0)} hin und her (Zyklus der Länge 2, außer x₀ = c/2 = {de(c / 2)}); "
                           "kein Grenzwert.")
    if abs(x0 - an.xstar) < 1e-12:
        return "warning", f"|a| = {de(abs(a))} > 1: Der Ruhepunkt x* = {de(an.xstar)} ist abstoßend; nur der Start genau auf x* bleibt dort (hier x₀ = x*), jede Abweichung wächst."
    return "warning", (f"|a| = {de(abs(a))} > 1: Der Abstand zum Ruhepunkt x* = {de(an.xstar)} wächst je Schritt um den Faktor {de(abs(a))} (der Ruhepunkt stößt die Bahn ab); "
                       "es gibt keinen Grenzwert.")


# ============================================================ Experiment 2 =================================================================
@dataclass(frozen=True)
class BuckelAnalysis:
    a: float
    b: float
    norms: np.ndarray          # ‖A^k‖, k = 0 … KMAX_ANALYSIS
    summary: A.BuckelSummary


def analyse_buckel(a: float, b: float) -> BuckelAnalysis:
    return BuckelAnalysis(a, b, A.jordan_power_norms(a, b, C.KMAX_ANALYSIS), A.buckel_summary(a, b))


def verdict_buckel(an: BuckelAnalysis) -> tuple[str, str]:
    s, b = an.summary, an.b
    if s.rho > 1.0 + TOL_ONE:
        return "warning", f"ρ = {de(s.rho)} > 1: Die Norm von A^k wächst (Spitze im Fenster bis k = {C.KMAX_ANALYSIS}: {de_sig(s.peak)}); der Eigenwert gibt die Wachstumsrate an; bei b ≠ 0 kommt noch ein Faktor von etwa k·b/a hinzu."
    if s.rho >= 1.0 - TOL_ONE:
        if b == 0:
            return "warning", "ρ = 1 mit b = 0: A ist die Einheitsmatrix, die Norm bleibt 1; die Bahn bleibt bei ihrem Start, ein Grenzwert ohne Abstand zum Start gibt es nicht."
        return "warning", f"ρ = 1 mit b = {de(b)} ≠ 0: Die Norm von A^k wächst auf lange Sicht linear (ungefähr k·b), obwohl beide Eigenwerte den Betrag 1 haben; die Bahn konvergiert nicht."
    if s.has_buckel:
        tail = f" Bei k = {C.RATIO_K} ist sie noch das {de_sig(s.ratio, 3)}-Fache von ρ^{C.RATIO_K}." if s.ratio is not None else ""
        back = f"; erst ab k = {s.first_below_one} liegt sie wieder unter 1" if s.first_below_one is not None else f"; bis k = {C.KMAX_ANALYSIS} liegt sie nicht wieder unter 1"
        return "warning", (f"ρ = {de(s.rho)} < 1, aber die Norm von A^k steigt zuerst auf {de(s.peak)} bei k = {s.k_peak}{back}.{tail} "
                           "Die Eigenwerte sagen den Grenzwert (alle Bahnen gehen gegen 0), nicht den Weg dorthin.")
    if b == 0:
        return "success", f"b = 0: A ist diagonal, also normal, und es gibt keinen Buckel: Die Norm fällt vom ersten Schritt an, ‖A^k‖ = {de(s.rho)}^k."
    return "success", (f"ρ = {de(s.rho)} < 1 und b = {de(b)} ≠ 0 (A ist nicht normal), aber die Norm von A^k steigt hier nie über 1 (Spitze bei k = {s.k_peak}); "
                       "ein Buckel hängt an a und b zugleich.")


# ============================================================ Experiment 3 =================================================================
@dataclass(frozen=True)
class GDAnalysis:
    kappa: float
    frac: float
    eta: float
    factors: tuple[float, float]
    rate: float
    state: str                 # unter | an | über
    steps: int | None          # Schritte bis ‖x‖ < GD_TOL
    best_eta: float
    best_rate: float
    best_frac: float


def analyse_gd(kappa: float, frac: float) -> GDAnalysis:
    eta = A.gd_eta_from_fraction(kappa, frac)
    f = A.gd_factors(kappa, eta)
    best_eta, best_rate = A.gd_best(kappa)
    return GDAnalysis(kappa, frac, eta, (float(f[0]), float(f[1])), A.gd_rate(kappa, eta), A.gd_threshold_state(kappa, eta),
                      A.gd_steps_to_tol(kappa, eta), best_eta, best_rate, kappa / (kappa + 1.0))


def verdict_gd(g: GDAnalysis) -> tuple[str, str]:
    thr = 2.0 / g.kappa
    if g.state == "über":
        return "warning", (f"η = {de(g.eta, 4)} liegt über der Schwelle 2/λmax = {de(thr, 4)}: Der Faktor der steilen Richtung ist {de(g.factors[1], 3)} (Betrag über 1), "
                           f"die Norm wächst je Schritt um den Faktor {de(g.rate, 3)}.")
    if g.state == "an":
        return "warning", (f"η = {de(g.eta, 4)} liegt genau an der Schwelle 2/λmax = {de(thr, 4)}: Die steile Richtung hat den Faktor −1, die Norm sinkt nicht unter 1; "
                           "es gibt keinen Grenzwert, solange diese Komponente nicht null ist.")
    limit = "flachen Richtung (Krümmung 1)" if abs(g.factors[0]) >= abs(g.factors[1]) else f"steilen Richtung (Krümmung {de(g.kappa, 0)})"
    steps = f"{g.steps} Schritten" if g.steps is not None else f"mehr als {C.GD_KMAX} Schritten"
    return "success", (f"Rate max|1 − ηλ| = {de(g.rate, 4)}, begrenzt von der {limit}; ‖x_k‖ < 10⁻⁶ nach {steps} (Start (1, 1)). "
                       f"Bester Schritt 2/(λmin + λmax) = {de(g.best_eta, 4)} ({de(g.best_frac, 3)} der Schwelle) mit Rate (κ − 1)/(κ + 1) = {de(g.best_rate, 4)}.")


# ============================================================ Experiment 4 =================================================================
@dataclass(frozen=True)
class BanachAnalysis:
    map_key: str
    lo: float                  # Intervall der Abbildung
    hi: float
    pos: float
    x0: float
    xstar: float
    L: float
    self_map: bool
    d01: float
    steps_shown: int
    errors: np.ndarray         # |x_k − x*|, k = 0 … steps_shown
    bounds: np.ndarray         # L^k/(1 − L)·|x₁ − x₀|
    holds: np.ndarray          # Fehler ≤ Schranke + Rundungsrauschen, k = 0 … steps_shown
    k_guaranteed: int
    k_actual: int | None
    true_rate: float           # |f'(x*)|


def analyse_banach(map_key: str, pos: float, steps: int) -> BanachAnalysis:
    m = BANACH_MAPS[map_key]
    x0 = A.banach_start(m, pos)
    xs = A.banach_orbit(m, x0, steps)
    xstar = A.banach_fixed_point(m)
    L = A.banach_lipschitz(m)
    d01 = abs(xs[1] - xs[0])
    ks = np.arange(steps + 1)
    errors = np.abs(xs - xstar)
    bounds = A.banach_bound(L, d01, ks)
    return BanachAnalysis(map_key, m.lo, m.hi, pos, x0, xstar, L, A.banach_is_self_map(m), d01, steps, errors, bounds, A.banach_bound_holds(errors, bounds, xstar),
                          A.banach_guaranteed_steps(L, d01), A.banach_actual_steps(m, x0, xstar), abs(float(m.df(np.array(xstar)))))


def verdict_banach(b: BanachAnalysis) -> tuple[str, str]:
    if not b.self_map or b.L >= 1.0:
        return "warning", "Die Voraussetzungen des Banachschen Satzes sind hier nicht erfüllt (Selbstabbildung des Intervalls und L < 1); die Schranke gilt nicht."
    if b.d01 == 0.0:
        return "info", f"Der Start x₀ = {de(b.x0, 4)} ist selbst der Fixpunkt: Fehler und Schranke sind null."
    n = b.steps_shown
    ok = int(b.holds[1:].sum())
    actual = f"{b.k_actual}" if b.k_actual is not None else f"mehr als {C.BANACH_KMAX}"
    kind = "success" if ok == n else "warning"
    return kind, (f"Auf [{de(b.lo, 3)}; {de(b.hi, 3)}] ist f eine Selbstabbildung mit L = {de(b.L, 4)} < 1. Die Schranke gilt in {ok} von {n} Schritten (k = 1 bis {n}, "
                  f"mit Rundungsrauschen von 4 Maschinengenauigkeiten). Bis zum Fehler 10⁻⁶ garantiert die Schranke {b.k_guaranteed} Schritte, tatsächlich genügen {actual}; "
                  f"die wirkliche Rate |f′(x*)| = {de(b.true_rate, 4)} " + ("liegt unter L: die Schranke ist hier nicht scharf." if b.true_rate < b.L - 1e-9 else "ist gleich L: bei dieser Abbildung ist die Schranke scharf."))


# ============================================================ Experiment 5 =================================================================
@dataclass(frozen=True)
class MomentumAnalysis:
    summary: A.MomentumSummary
    norms: np.ndarray          # ‖M^k‖, k = 0 … K
    critical_h: float
    eigenvalues: tuple[complex, complex]


def analyse_momentum(beta: float, h: float, steps: int) -> MomentumAnalysis:
    tr, det = 1.0 + beta - h, beta
    d = tr * tr - 4.0 * det
    r = complex(math.sqrt(d)) if d >= 0 else complex(0.0, math.sqrt(-d))
    return MomentumAnalysis(A.momentum_summary(beta, h), A.heavy_ball_norms(beta, h, steps), A.heavy_ball_critical_h(beta),
                            ((tr + r) / 2.0, (tr - r) / 2.0))


def verdict_momentum(m: MomentumAnalysis) -> tuple[str, str]:
    s = m.summary
    if s.kind == "instabil":
        return "warning", f"Spektralradius ρ = {de(s.rho, 3)} ≥ 1 (h liegt nicht in (0; 2(1 + β)) = (0; {de(2 * (1 + s.beta), 2)})): Die Bahn konvergiert nicht."
    art = {"doppelt": "doppelter Eigenwert (kritisch gedämpft)", "reell": "zwei reelle Eigenwerte", "komplex": "ein komplexes Eigenwertpaar"}[s.kind]
    if s.peak > 1.0 + 1e-12:
        tail = f" Bei k = {C.RATIO_K} ist die Norm noch das {de_sig(s.ratio, 3)}-Fache von ρ^{C.RATIO_K}." if s.ratio is not None else ""
        return "warning", (f"Spektralradius ρ = {de(s.rho, 3)} < 1 ({art}), aber ‖M^k‖ steigt zuerst auf {de(s.peak, 2)} bei k = {s.k_peak}.{tail} "
                           "Das beschreibt den Zustand (x_k, x_(k−1)), nicht den Funktionswert.")
    return "success", f"Spektralradius ρ = {de(s.rho, 3)} < 1 ({art}); die Norm von M^k fällt vom Start an, kein Buckel (Spitze bei k = {s.k_peak})."


# ============================================================ Kernfrage ====================================================================
CORE_A = (0.6, 0.9, 0.99)
CORE_B = (0.0, 1.0, 5.0, 20.0)


def core_table() -> list[dict]:
    """Je a in CORE_A eine Zeile: Spitze von ‖A^k‖ (k = 0 bis 399) für jedes b in CORE_B."""
    return [{"a": a, "peaks": [A.buckel_summary(a, b).peak for b in CORE_B]} for a in CORE_A]


def verdict_core(rows: list[dict]) -> tuple[str, str]:
    zeros_one = sum(1 for r in rows if abs(r["peaks"][0] - 1.0) < 1e-12)
    rising = sum(1 for r in rows if all(r["peaks"][i] < r["peaks"][i + 1] for i in range(len(r["peaks"]) - 1)))
    biggest = max((p, r["a"], b) for r in rows for p, b in zip(r["peaks"], CORE_B))
    cells = sum(len(r["peaks"]) for r in rows)
    return "info", (f"In allen {cells} Zellen ist der Spektralradius kleiner als 1 (a = {', '.join(de(a) for a in CORE_A)}), und doch liegt die Spitze der Norm bei b = 0 genau bei 1 "
                    f"({zeros_one} von {len(rows)} Zeilen) und steigt mit b ({rising} von {len(rows)} Zeilen streng): bis {de(biggest[0], 1)} bei a = {de(biggest[1])}, b = {de(biggest[2], 0)}. "
                    "Der Eigenwert allein sagt also den Grenzwert; ob die Norm vorher steigt, hängt an der Nicht-Normalität der Matrix.")
