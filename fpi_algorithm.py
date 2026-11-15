"""Kern der Demo (nur numpy/Standardbibliothek): lineare Iterationen x → A·x und die Banach-Schranke.

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b):

* `affine_orbit`, `fixed_point_affine`      Experiment 1: x → a·x + c
* `jordan_power_norms`, `buckel_summary`      Experiment 2: ‖A^k‖₂ für A = [[a, b], [0, a]] in geschlossener Form
* `gd_*`                                      Experiment 3: Gradientenabstieg auf der Quadratik mit den Krümmungen 1 und κ
* `banach_*`                                  Experiment 4: A-priori-Schranke L^k/(1 − L)·|x₁ − x₀|
* `heavy_ball_*`                              Experiment 5: Gradientenabstieg mit Momentum, Zustand (x_k, x_(k−1))

Gerechnet wird in Float64. Wo die Rundung den Befund berührt, steht es an der Funktion (Banach-Schranke: Rundungsrauschen)."""

import math
from dataclasses import dataclass

import numpy as np

import fpi_constants as C
from fpi_scenario import BanachMap

EPS = float(np.finfo(float).eps)


# ============================================================ Experiment 1: eine Zahl iterieren ============================================
def fixed_point_affine(a: float, c: float):
    """Ruhepunkt x* = c/(1 − a) von x → a·x + c; für a = 1 gibt es ihn nur bei c = 0 (dann ist jeder Punkt Ruhepunkt)."""
    if a == 1.0:
        return None
    return c / (1.0 - a)


def affine_orbit(a: float, c: float, x0: float, steps: int) -> np.ndarray:
    """x_0 … x_steps der Folge x_(k+1) = a·x_k + c."""
    xs = np.empty(steps + 1)
    xs[0] = x0
    for k in range(steps):
        xs[k + 1] = a * xs[k] + c
    return xs


def cobweb_points(a: float, c: float, x0: float, steps: int, lo: float, hi: float):
    """Linienzug des Spinnwebs: von (x_k, x_k) senkrecht zu (x_k, f(x_k)), dann waagerecht zu (x_(k+1), x_(k+1)); endet, sobald ein Punkt das Fenster [lo, hi] verlässt."""
    xs, ys = [x0], [x0]
    x = x0
    for _ in range(steps):
        fx = a * x + c
        if not (lo <= fx <= hi):
            break
        xs += [x, fx]
        ys += [fx, fx]
        x = fx
    return xs, ys


# ============================================================ Experiment 2: Eigenwert gegen Weg ============================================
def norm2_upper_triangular(p, q):
    """Spektralnorm von [[p, q], [0, p]]: Aus s₁·s₂ = |det| = p² und s₁² + s₂² = 2p² + q² folgt s₁ = (√(q² + 4p²) + |q|)/2."""
    q = np.abs(q)
    return (np.sqrt(q * q + 4.0 * p * p) + q) / 2.0


def jordan_power_norms(a: float, b: float, kmax: int) -> np.ndarray:
    """‖A^k‖₂ für k = 0 … kmax, A = [[a, b], [0, a]]; A^k = [[a^k, k·a^(k−1)·b], [0, a^k]] (geschlossene Form, ohne Matrixpotenz)."""
    ks = np.arange(kmax + 1)
    p = np.power(float(a), ks)
    q = np.where(ks == 0, 0.0, ks * np.power(float(a), np.maximum(ks - 1, 0)) * float(b))
    return norm2_upper_triangular(p, q)


@dataclass(frozen=True)
class BuckelSummary:
    rho: float                 # Spektralradius |a|
    peak: float                # max_k ‖A^k‖ über k = 0 … kmax (k = 0 gibt 1)
    k_peak: int                # kleinstes k mit diesem Maximum
    first_below_one: int | None   # erstes k nach dem Maximum mit ‖A^k‖ < 1, sonst None
    ratio: float | None        # ‖A^RATIO_K‖ / ρ^RATIO_K (None bei ρ = 0)
    has_buckel: bool           # Norm steigt über 1


def buckel_summary(a: float, b: float, kmax: int = C.KMAX_ANALYSIS) -> BuckelSummary:
    norms = jordan_power_norms(a, b, kmax)
    k_peak = int(np.argmax(norms))
    peak = float(norms[k_peak])
    after = np.nonzero(norms[k_peak:] < 1.0)[0]
    first = int(k_peak + after[0]) if len(after) else None
    rho = abs(float(a))
    ratio = None
    if rho > 0 and kmax >= C.RATIO_K:
        ratio = float(norms[C.RATIO_K] / rho ** C.RATIO_K)
    return BuckelSummary(rho, peak, k_peak, first, ratio, peak > 1.0 + 1e-12)


# ============================================================ Experiment 3: Gradientenabstieg auf der Quadratik ============================
def gd_factors(kappa: float, eta: float) -> np.ndarray:
    """Faktoren 1 − η·λ der beiden Eigenrichtungen λ = 1 und λ = κ."""
    return 1.0 - eta * np.array([1.0, float(kappa)])


def gd_eta_from_fraction(kappa: float, frac: float) -> float:
    """Schritt η als Anteil der Schwelle 2/λmax = 2/κ."""
    return frac * 2.0 / float(kappa)


def gd_rate(kappa: float, eta) -> np.ndarray | float:
    """Rate max |1 − η·λ| über λ ∈ {1, κ}; gilt für Zahl und Feld."""
    eta = np.asarray(eta, dtype=float)
    out = np.maximum(np.abs(1.0 - eta), np.abs(1.0 - eta * kappa))
    return float(out) if out.ndim == 0 else out


def gd_best(kappa: float) -> tuple[float, float]:
    """Bester Schritt 2/(λmin + λmax) und die zugehörige Rate (κ − 1)/(κ + 1)."""
    return 2.0 / (1.0 + kappa), (kappa - 1.0) / (kappa + 1.0)


def gd_norm_path(kappa: float, eta: float, steps: int, x0=(1.0, 1.0)) -> np.ndarray:
    """‖x_k‖ für k = 0 … steps; jede Eigenrichtung wird einzeln mit ihrem Faktor potenziert."""
    ks = np.arange(steps + 1)
    f = gd_factors(kappa, eta)
    with np.errstate(over="ignore"):      # über der Schwelle wächst die Norm bis über die Gleitkomma-Grenze (dann inf, die Suche bricht ab)
        return np.sqrt((np.power(f[0], ks) * x0[0]) ** 2 + (np.power(f[1], ks) * x0[1]) ** 2)


def gd_steps_to_tol(kappa: float, eta: float, tol: float = C.GD_TOL, kmax: int = C.GD_KMAX, x0=(1.0, 1.0)):
    """Kleinstes k mit ‖x_k‖ < tol (Start x0), None, wenn es bis kmax nicht erreicht wird."""
    path = gd_norm_path(kappa, eta, kmax, x0)
    hit = np.nonzero(path < tol)[0]
    return int(hit[0]) if len(hit) else None


def gd_threshold_state(kappa: float, eta: float) -> str:
    """„unter“, „an“ oder „über“ der Schwelle 2/λmax (Vergleich mit Gleitkomma-Toleranz)."""
    thr = 2.0 / kappa
    if abs(eta - thr) <= 1e-12 * thr:
        return "an"
    return "unter" if eta < thr else "über"


# ============================================================ Experiment 4: Banach =========================================================
def banach_grid(m: BanachMap, points: int = C.GRID_POINTS) -> np.ndarray:
    return np.linspace(m.lo, m.hi, points)


def banach_lipschitz(m: BanachMap, points: int = C.GRID_POINTS) -> float:
    """L = max |f'| auf dem Intervall, an dicht liegenden Stützstellen (die Ableitungen der drei Abbildungen sind monoton, der Rand ist Stützstelle)."""
    return float(np.max(np.abs(m.df(banach_grid(m, points)))))


def banach_is_self_map(m: BanachMap, points: int = C.GRID_POINTS) -> bool:
    """f bildet das Intervall in sich ab (an den Stützstellen geprüft, mit Gleitkomma-Toleranz)."""
    y = m.f(banach_grid(m, points))
    tol = 4 * EPS * max(abs(m.lo), abs(m.hi), 1.0)
    return bool(y.min() >= m.lo - tol and y.max() <= m.hi + tol)


def banach_fixed_point(m: BanachMap) -> float:
    """Fixpunkt durch Bisektion von f(x) − x auf dem Intervall (nicht durch Iterieren: unabhängig vom geprüften Verfahren)."""
    lo, hi = m.lo, m.hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:      # Gleitkomma-Auflösung erreicht
            break
        if float(m.f(np.array(mid))) - mid > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def banach_start(m: BanachMap, pos: float) -> float:
    return m.lo + pos * (m.hi - m.lo)


def banach_orbit(m: BanachMap, x0: float, steps: int) -> np.ndarray:
    xs = np.empty(steps + 1)
    xs[0] = x0
    for k in range(steps):
        xs[k + 1] = float(m.f(np.array(xs[k])))
    return xs


def banach_bound(L: float, d01: float, ks) -> np.ndarray:
    """A-priori-Schranke |x_k − x*| ≤ L^k/(1 − L)·|x₁ − x₀|."""
    return np.power(L, np.asarray(ks, dtype=float)) / (1.0 - L) * d01


def banach_guaranteed_steps(L: float, d01: float, tol: float = C.BANACH_TOL) -> int:
    """Kleinstes k, für das die Schranke unter tol liegt (0, wenn der Start schon Fixpunkt ist)."""
    if d01 <= 0.0:
        return 0
    k = math.ceil(math.log(tol * (1.0 - L) / d01) / math.log(L))
    return max(k, 0)


def banach_actual_steps(m: BanachMap, x0: float, xstar: float, tol: float = C.BANACH_TOL, kmax: int = C.BANACH_KMAX):
    """Kleinstes k mit |x_k − x*| < tol im Gleitkomma-Lauf (None, wenn es bis kmax nicht erreicht wird)."""
    x = x0
    for k in range(kmax + 1):
        if abs(x - xstar) < tol:
            return k
        x = float(m.f(np.array(x)))
    return None


def banach_rounding_floor(xstar: float) -> float:
    """Rundungsrauschen der Fehlermessung: Die Schranke fällt unter jede Maschinengenauigkeit, der gemessene Fehler nicht (4 Einheiten der Maschinengenauigkeit)."""
    return 4.0 * EPS * max(abs(xstar), 1.0)


def banach_bound_holds(errors: np.ndarray, bounds: np.ndarray, xstar: float) -> np.ndarray:
    """Je Schritt: Fehler ≤ Schranke + Rundungsrauschen."""
    return errors <= bounds + banach_rounding_floor(xstar)


# ============================================================ Experiment 5: Momentum =======================================================
def heavy_ball_matrix(beta: float, h: float) -> np.ndarray:
    """Zustand (x_k, x_(k−1)): x_(k+1) = (1 + β − h)·x_k − β·x_(k−1) auf einer Krümmung λ mit h = η·λ."""
    return np.array([[1.0 + beta - h, -beta], [1.0, 0.0]])


def heavy_ball_critical_h(beta: float) -> float:
    """Kritisch gedämpft (doppelter Eigenwert √β): h = (1 − √β)²."""
    return (1.0 - math.sqrt(beta)) ** 2


def heavy_ball_spectral_radius(beta, h):
    """Spektralradius aus Spur und Determinante (z² − (1 + β − h)·z + β = 0), für Zahl und Feld: reelle Wurzeln (|t| + √d)/2, komplexe √β."""
    t = 1.0 + np.asarray(beta, dtype=float) - np.asarray(h, dtype=float)
    d = t * t - 4.0 * np.asarray(beta, dtype=float)
    out = np.where(d >= 0, (np.abs(t) + np.sqrt(np.maximum(d, 0.0))) / 2.0, np.sqrt(np.asarray(beta, dtype=float)))
    return float(out) if np.ndim(out) == 0 else out


def heavy_ball_kind(beta: float, h: float) -> str:
    """„instabil“ (Spektralradius ≥ 1, mit Toleranz: auf dem Rand h = 2(1 + β) ist ρ = 1 und rundet knapp darunter), „doppelt“ (doppelter Eigenwert), „reell“ oder „komplex“."""
    if heavy_ball_spectral_radius(beta, h) >= 1.0 - 1e-12:
        return "instabil"
    d = (1.0 + beta - h) ** 2 - 4.0 * beta
    if abs(d) <= 1e-12:
        return "doppelt"
    return "reell" if d > 0 else "komplex"


def norm2_2x2(p11, p12, p21, p22):
    """Spektralnorm einer 2×2-Matrix aus Frobeniusnorm F und Determinante D: s₁² = (F + √(F² − 4D²))/2 (elementweise für Felder).

    Die Einträge werden vorher durch ihren größten Betrag geteilt und die Norm danach wieder damit multipliziert: Bei instabilen Parametern wachsen die Potenzen bis 10^238,
    und F² ginge sonst über die Grenze der Gleitkommazahlen."""
    scale = np.maximum(np.maximum(np.abs(p11), np.abs(p12)), np.maximum(np.abs(p21), np.abs(p22)))
    sc = np.where(scale == 0.0, 1.0, scale)
    a11, a12, a21, a22 = p11 / sc, p12 / sc, p21 / sc, p22 / sc
    f = a11 * a11 + a12 * a12 + a21 * a21 + a22 * a22
    d = a11 * a22 - a12 * a21
    return sc * np.sqrt((f + np.sqrt(np.maximum(f * f - 4.0 * d * d, 0.0))) / 2.0)


def _heavy_ball_iter(beta, h, kmax: int):
    """Erzeugt (k, ‖M^k‖₂) für k = 1 … kmax durch fortlaufendes Multiplizieren; `beta` und `h` Zahlen oder Felder gleicher Form."""
    beta = np.asarray(beta, dtype=float)
    h = np.asarray(h, dtype=float)
    m = 1.0 + beta - h
    p11 = np.ones_like(m)
    p12 = np.zeros_like(m)
    p21 = np.zeros_like(m)
    p22 = np.ones_like(m)
    for k in range(1, kmax + 1):
        # P ← P·M mit M = [[m, −β], [1, 0]]
        p11, p12, p21, p22 = p11 * m + p12, -beta * p11, p21 * m + p22, -beta * p21
        n = norm2_2x2(p11, p12, p21, p22)
        # Ohne Momentum (β = 0) besteht der Zustand nur aus x_k: die Matrix ist der Skalar 1 − h (normal); die verzögerte Koordinate wäre nur Buchführung.
        with np.errstate(over="ignore"):      # der Skalar-Zweig wird für β ≠ 0 mitgerechnet und verworfen
            scalar = np.abs(m) ** k
        yield k, np.where(beta == 0.0, scalar, n)


def heavy_ball_norms(beta, h, kmax: int):
    """‖M^k‖₂ für k = 0 … kmax; Feld der Form (kmax + 1, …). Für große Felder `heavy_ball_peaks` nehmen (hält nur das laufende Maximum)."""
    m = np.asarray(beta, dtype=float) + np.asarray(h, dtype=float)
    out = np.empty((kmax + 1,) + m.shape)
    out[0] = 1.0
    for k, n in _heavy_ball_iter(beta, h, kmax):
        out[k] = n
    return out


def heavy_ball_peaks(beta, h, kmax: int):
    """(max_k ‖M^k‖, kleinstes k mit diesem Maximum) über k = 0 … kmax, elementweise; k = 0 gibt 1."""
    shape = np.shape(np.asarray(beta, dtype=float) + np.asarray(h, dtype=float))
    peak = np.ones(shape)
    arg = np.zeros(shape, dtype=int)
    for k, n in _heavy_ball_iter(beta, h, kmax):
        better = n > peak
        peak = np.where(better, n, peak)
        arg = np.where(better, k, arg)
    return peak, arg


@dataclass(frozen=True)
class MomentumSummary:
    beta: float
    h: float
    rho: float
    kind: str                  # instabil | doppelt | reell | komplex
    peak: float                # max_k ‖M^k‖ (k = 0 gibt 1)
    k_peak: int
    ratio: float | None        # ‖M^100‖ / ρ^100


def momentum_summary(beta: float, h: float, kmax: int = C.KMAX_ANALYSIS) -> MomentumSummary:
    norms = heavy_ball_norms(beta, h, kmax)
    k_peak = int(np.argmax(norms))
    rho = float(heavy_ball_spectral_radius(beta, h))
    ratio = float(norms[C.RATIO_K] / rho ** C.RATIO_K) if 0 < rho < 1 else None
    return MomentumSummary(float(beta), float(h), rho, heavy_ball_kind(beta, h), float(norms[k_peak]), k_peak, ratio)


def heavy_ball_raster():
    """Das Raster der Vormessung: 49 Werte β von 0,02 bis 0,98, je 80 Werte h im Innern von (0, 2(1 + β)); gibt (β-Feld, h-Feld) der Form (49, 80)."""
    lo, hi, n = C.RASTER_BETAS
    betas = np.linspace(lo, hi, n)
    hs = np.array([np.linspace(0.0, 2.0 * (1.0 + b), C.RASTER_H + 2)[1:-1] for b in betas])
    return np.repeat(betas[:, None], C.RASTER_H, axis=1), hs


@dataclass(frozen=True)
class RasterCount:
    total: int
    above2: int
    above10: int
    real_total: int
    real_above2: int
    complex_total: int
    complex_above2: int
    max_peak: float
    max_at: tuple[float, float]    # (β, h) der größten Spitze


def heavy_ball_raster_count(kmax: int = C.RASTER_KMAX) -> RasterCount:
    """Zählt über das Raster die stabilen Punkte (Spektralradius < 1) und darunter die mit Spitze > 2 bzw. > 10 (getrennt nach reellen und komplexen Eigenwerten)."""
    B, H = heavy_ball_raster()
    stable = heavy_ball_spectral_radius(B, H) < 1.0
    peaks, _ = heavy_ball_peaks(B, H, kmax)
    disc = (1.0 + B - H) ** 2 - 4.0 * B
    real = stable & (disc >= 0)
    cplx = stable & (disc < 0)
    big = stable & (peaks > C.RASTER_SPITZE)
    flat = np.where(stable, peaks, -1.0)
    i = np.unravel_index(int(np.argmax(flat)), flat.shape)
    return RasterCount(int(stable.sum()), int(big.sum()), int((stable & (peaks > 10.0)).sum()), int(real.sum()), int((real & big).sum()),
                       int(cplx.sum()), int((cplx & big).sum()), float(flat[i]), (float(B[i]), float(H[i])))


def heavy_ball_worst_start_peak(beta: float, h: float, kmax: int = 300):
    """Größtes max_k |x_k| über alle Einheits-Startvektoren (x₀, x₋₁): die Norm der ersten Zeile von M^k, k = 1 … kmax-1; gibt (Wert, k)."""
    M = heavy_ball_matrix(beta, h)
    P = np.eye(2)
    best, kb = 0.0, 0
    for k in range(1, kmax):
        P = P @ M
        z = float(np.linalg.norm(P[0]))
        if z > best:
            best, kb = z, k
    return best, kb


def heavy_ball_function_values(beta: float, h: float, steps: int):
    """Funktionswerte f_k = x_k²/2 (λ = 1) mit Standardstart x₀ = x₋₁ = 1; ein Hinweis auf die Monotonie des Verlustes."""
    xp, x = 1.0, 1.0
    f = [0.5 * x * x]
    for _ in range(steps):
        xp, x = x, (1.0 + beta - h) * x - beta * xp
        f.append(0.5 * x * x)
    return np.array(f)
