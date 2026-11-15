"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Schnellstart-Szenarien.

Keine Voreinstellung ist eine Zweierpotenz (0,5; 1; 2; 0,25 …): Dualbrüche sind in Gleitkomma exakt und erzeugen Sonderfälle, die mit dem
Eigenwert nichts zu tun haben (Messung der Linie, Stück 3b). Ausnahmen stehen mit Grund an Ort und Stelle (Schwelle η = 2/λmax, Start 1 bei cos)."""

# ---- allgemeine Einstellung (Seitenleiste) ------------------------------------------------------------------------------------------------
K_STEPS_MIN, K_STEPS_MAX, DEFAULT_K_STEPS = 40, 400, 120   # Schritte, die die Diagramme zeigen (k = 0 bis K)

# ---- Experiment 1: eine Zahl iterieren, x -> a x + c -----------------------------------------------------------------------------------
E1_A_MIN, E1_A_MAX, DEFAULT_E1_A = -1.5, 1.5, 0.8
E1_C_MIN, E1_C_MAX, DEFAULT_E1_C = -3.0, 3.0, 1.3
E1_X0_MIN, E1_X0_MAX, DEFAULT_E1_X0 = -10.0, 10.0, 3.7
WEB_STEPS = 25                      # so viele Schritte zeichnet das Spinnweb höchstens

# ---- Experiment 2: Eigenwert gegen Weg, A = [[a, b], [0, a]] ----------------------------------------------------------------------------
E2_A_MIN, E2_A_MAX, DEFAULT_E2_A = 0.0, 1.2, 0.9
E2_B_MIN, E2_B_MAX, DEFAULT_E2_B = 0.0, 10.0, 5.0
KMAX_ANALYSIS = 399                 # Maximum und „erstes k unter 1“ gelten für k = 0 bis 399, unabhängig vom Anzeigefenster
RATIO_K = 100                       # Vergleichsschritt für das Verhältnis ‖A^k‖ / ρ^k

# ---- Experiment 3: Gradientenabstieg auf der Quadratik f = (x² + κ y²)/2 -----------------------------------------------------------------
E3_KAPPA_MIN, E3_KAPPA_MAX, DEFAULT_E3_KAPPA = 2.0, 100.0, 10.0
E3_FRAC_MIN, E3_FRAC_MAX, DEFAULT_E3_FRAC = 0.05, 1.10, 0.9      # η als Anteil der Schwelle 2/λmax
GD_TOL = 1e-6                       # Schwelle für „Schritte bis ‖x‖ < GD_TOL“, Start (1, 1)
GD_KMAX = 20000                     # so weit sucht die Schrittzahl

# ---- Experiment 4: Banach-Schranke ----------------------------------------------------------------------------------------------------
E4_MAP_KEYS = ("cos", "affin", "wurzel")
E4_MAP_LABELS = {"cos": "cos x auf [cos 1, 1]", "affin": "0,6·x + 1 auf [0, 5]", "wurzel": "√(x + 1,5) auf [0, 3]"}
DEFAULT_E4_MAP = "affin"
E4_POS_MIN, E4_POS_MAX, DEFAULT_E4_POS = 0.0, 1.0, 0.2          # Start als Lage im Intervall (0 = linker, 1 = rechter Rand)
BANACH_TOL = 1e-6
BANACH_KMAX = 10000
GRID_POINTS = 20001                 # Stützstellen für Lipschitz-Konstante und Selbstabbildung

# ---- Experiment 5: Gradientenabstieg mit Momentum (Heavy Ball) -------------------------------------------------------------------------
E5_BETA_MIN, E5_BETA_MAX, DEFAULT_E5_BETA = 0.0, 0.98, 0.7
E5_H_MIN, E5_H_MAX, DEFAULT_E5_H = 0.001, 4.0, 0.3              # h = η·λ
RASTER_BETAS = (0.02, 0.98, 49)     # linspace: von, bis, Anzahl
RASTER_H = 80                       # h-Werte je β im Innern von (0, 2(1 + β))
RASTER_KMAX = 2000
RASTER_SPITZE = 2.0                 # „Buckel“ im Raster: max_k ‖M^k‖ > 2

# ---- Farben --------------------------------------------------------------------------------------------------------------------------
COLOR_MAIN = "#2F6B65"
COLOR_REF = "#c0562a"
COLOR_GREY = "rgba(150,150,150,0.75)"
COLOR_BAND = "rgba(76,120,168,0.18)"
COLOR_LINE = "#4c78a8"

# ---- Schnellstart: je Szenario die Werte der Regler, die es setzt; alle übrigen behalten ihre Voreinstellung ------------------------------
# Schlüssel = Schlüssel der Regler in fpi_presets.SETTING_SPECS. 10/11 ist der beste Schritt für κ = 10 (Anteil κ/(κ + 1) der Schwelle).
PRESETS = {
    "Buckel (a 0,9, b 5)": {},
    "Normal (b = 0)": {"e2_b": 0.0},
    "GD bester Schritt (κ 10)": {"e3_kappa": 10.0, "e3_frac": 0.909},
    "GD an der Schwelle": {"e3_kappa": 10.0, "e3_frac": 1.0},
    "Momentum kritisch gedämpft": {"e5_beta": 0.81, "e5_h": 0.01},
    "cos (Banach)": {"e4_map": "cos", "e4_pos": 1.0},
}
PRESET_HELP = {
    "Buckel (a 0,9, b 5)": "Experiment 2: A = [[0,9; 5], [0; 0,9]] hat den Spektralradius 0,9, doch die Norm von A^k steigt zuerst auf 19,4. Alle anderen Regler stehen auf ihrer Voreinstellung.",
    "Normal (b = 0)": "Experiment 2 mit b = 0: die Matrix ist ein Vielfaches der Einheitsmatrix, die Norm fällt vom ersten Schritt an wie 0,9^k.",
    "GD bester Schritt (κ 10)": "Experiment 3: Krümmungen 1 und 10, Schritt 0,909 der Schwelle 2/λmax (bester Schritt 2/(λmin + λmax) = 10/11 der Schwelle, gerundet auf drei Stellen).",
    "GD an der Schwelle": "Experiment 3 genau an der Schwelle η = 2/λmax: eine Richtung hat den Faktor −1, die Norm sinkt nicht unter 1.",
    "Momentum kritisch gedämpft": "Experiment 5: β = 0,81 und h = 0,01 geben den doppelten Eigenwert 0,9 (kritisch gedämpft); die Norm von M^k steigt zuerst auf 7,03.",
    "cos (Banach)": "Experiment 4: x → cos x, Start am rechten Rand 1 des Intervalls [cos 1, 1]; garantierte und tatsächliche Schrittzahl bis 10⁻⁶.",
}
DEFAULT_PRESET = "Buckel (a 0,9, b 5)"
