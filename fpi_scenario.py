"""Szenario: die drei Abbildungen des Experiments „Banach-Schranke“ mit ihrem Intervall.

Jede Abbildung f hat ein Intervall [lo, hi], das sie in sich abbildet (Selbstabbildung), und eine Ableitung `df`, aus der die Lipschitz-Konstante
L = max |f'| auf dem Intervall folgt. Keine Abbildung hat eine Zweierpotenz als Konstante (kein ½·x + 1): Dualbrüche sind in Gleitkomma exakt."""

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

import fpi_constants as C


@dataclass(frozen=True)
class BanachMap:
    key: str
    label: str
    formula: str                       # für Texte
    f: Callable[[np.ndarray], np.ndarray]
    df: Callable[[np.ndarray], np.ndarray]
    lo: float
    hi: float
    L_exact: float                     # Lipschitz-Konstante auf [lo, hi], von Hand hergeleitet (Orakel in den Tests)


BANACH_MAPS = {
    # cos bildet [0, 1] in [cos 1, 1] ab; auf [cos 1, 1] ist |cos'| = sin x ≤ sin 1 (x ≤ 1 < π/2), und cos([cos 1, 1]) = [cos 1, cos(cos 1)] liegt darin.
    "cos": BanachMap("cos", C.E4_MAP_LABELS["cos"], "cos x", np.cos, lambda x: -np.sin(x), math.cos(1.0), 1.0, math.sin(1.0)),
    # 0,6·x + 1 bildet [0, 5] in [1, 4] ab; L = 0,6; Fixpunkt 2,5.
    "affin": BanachMap("affin", C.E4_MAP_LABELS["affin"], "0,6·x + 1", lambda x: 0.6 * x + 1.0, lambda x: np.full_like(np.asarray(x, dtype=float), 0.6), 0.0, 5.0, 0.6),
    # √(x + 1,5) bildet [0, 3] in [√1,5; √4,5] ≈ [1,22; 2,12] ab; |f'| = 1/(2·√(x + 1,5)) ist am linken Rand am größten: L = 1/(2·√1,5).
    "wurzel": BanachMap("wurzel", C.E4_MAP_LABELS["wurzel"], "√(x + 1,5)", lambda x: np.sqrt(x + 1.5), lambda x: 0.5 / np.sqrt(x + 1.5), 0.0, 3.0, 0.5 / math.sqrt(1.5)),
}
