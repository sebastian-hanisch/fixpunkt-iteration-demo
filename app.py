"""Iterieren und Fixpunkt – Eigenwerte sagen den Grenzwert, nicht den Weg - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 1 der Linie „Dynamische Systeme“ (die Wurzel): fünf Experimente zur Iteration x → A·x. Aufbau nach DEMO-PLAYBOOK §2: fester Kopf, Schnellstart und allgemeine Einstellung
in der Seitenleiste, die Regler der Experimente im Hauptbereich beim Experiment, Kernfragen vor den Annahmen, Formeln im Expander am Ende.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import fpi_algorithm as A
import fpi_constants as C
import fpi_evaluation as E
from fpi_evaluation import de
from fpi_layout import dunkles_design, kopf_css
from fpi_presets import (
    SETTING_SPECS,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    set_best_step,
    set_critical_h,
    sidebar_presets,
    sync_query_params,
)
from fpi_visualization import (
    build_affine_series,
    build_banach,
    build_buckel,
    build_cobweb,
    build_gd_path,
    build_gd_rate,
    build_momentum_curve,
    build_raster,
)

st.set_page_config(page_title="Iterieren und Fixpunkt – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _affine(a, c, x0, steps):
    return E.analyse_affine(a, c, x0, steps)


@st.cache_data(show_spinner=False)
def _buckel(a, b):
    return E.analyse_buckel(a, b)


@st.cache_data(show_spinner=False)
def _gd(kappa, frac):
    return E.analyse_gd(kappa, frac)


@st.cache_data(show_spinner=False)
def _banach(map_key, pos, steps):
    return E.analyse_banach(map_key, pos, steps)


@st.cache_data(show_spinner=False)
def _momentum(beta, h, steps):
    return E.analyse_momentum(beta, h, steps)


@st.cache_data(show_spinner=False)
def _raster_count():
    return A.heavy_ball_raster_count()


@st.cache_data(show_spinner=False)
def _core_rows():
    return E.core_table()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in fpi_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🔁 Iterieren und Fixpunkt – Eigenwerte sagen den Grenzwert, nicht den Weg")
st.markdown(
    """
Wiederholt man **x → A·x**, entscheidet der **Spektralradius ρ(A)**, ob die Bahn für jeden Startwert gegen den Ruhepunkt geht (genau dann, wenn ρ < 1). Er sagt aber nichts über den **Weg**: Bei einer nicht-normalen Matrix kann die
Norm zuerst stark wachsen, obwohl alle Eigenwerte klein sind. Dieses Stück ist die Wurzel der Linie „Dynamische Systeme“; es bleibt bei **linearen** Iterationen, die Nichtlinearität kommt im nächsten Stück.

Fünf Experimente: **eine Zahl iterieren**, **Eigenwert gegen Weg**, **Gradientenabstieg** auf einer Quadratik, die **Banach-Schranke** und **Momentum** mit denselben Eigenwerten und anderem Weg. Danach die Kernfrage, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert das Iterieren", expanded=True):
    st.markdown(
        """
1. **Regel wiederholen:** Aus dem Zustand x_k entsteht x_(k+1) = A·x_k (bei einer Zahl: a·x_k + c). Die Bahn ist die Folge x₀, x₁, x₂, …
2. **Ruhepunkt:** Ein Punkt x*, der sich selbst abbildet. Bei der linearen Regel geht jede Bahn gegen ihn, wenn alle Eigenwerte von A betragsmäßig unter 1 liegen (ρ < 1); liegt einer über 1, wächst fast jede Bahn.
3. **Eigenwert gegen Weg:** Der Eigenwert gibt die Rate auf lange Sicht (ρ^k). Wie die Norm von A^k auf dem Weg dorthin aussieht, hängt zusätzlich davon ab, ob A **normal** ist (zu orthogonalen Achsen diagonalisierbar). Bei einer nicht-normalen Matrix kann ‖A^k‖ erst steigen und dann fallen.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Szenario ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Anzeige**")
    k_steps = st.slider("Schritte im Diagramm K", *bounds("k_steps"), key="k_steps", help="Die Diagramme zeigen k = 0 bis K. Spitze und „erstes k unter 1“ in Experiment 2 und 5 gelten immer für k = 0 bis 399, unabhängig von K.")
    st.caption("Die Regler der Experimente stehen unten bei den Experimenten.")

# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 1: eine Zahl iterieren
# ---------------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Experiment 1: Eine Zahl iterieren (Spinnweb)")
st.caption("Die Regel x → a·x + c. Der Ruhepunkt ist x* = c/(1 − a); ob die Bahn ihn erreicht, entscheidet allein |a|.")
c1, c2, c3 = st.columns(3)
e1_a = c1.slider("Faktor a", *bounds("e1_a"), step=0.01, key="e1_a", help="Der Betrag von a entscheidet: unter 1 Grenzwert, bei genau 1 im Allgemeinen kein Grenzwert, über 1 Wachstum (außer im Ruhepunkt selbst).")
e1_c = c2.slider("Konstante c", *bounds("e1_c"), step=0.1, key="e1_c")
e1_x0 = c3.slider("Start x₀", *bounds("e1_x0"), step=0.1, key="e1_x0")
an1 = _affine(float(e1_a), float(e1_c), float(e1_x0), int(k_steps))
col_a, col_b = st.columns(2)
col_a.plotly_chart(build_cobweb(an1), width="stretch", key="e1_web")
col_b.plotly_chart(build_affine_series(an1), width="stretch", key="e1_series")
kind, text = E.verdict_affine(an1)
getattr(st, kind)(text)

# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: Eigenwert gegen Weg
# ---------------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Experiment 2: Eigenwert gegen Weg (nicht-normale Matrix)")
st.caption("A = [[a, b], [0, a]]: beide Eigenwerte sind a, der Spektralradius ist ρ = |a|. Die Norm von A^k steht in geschlossener Form (A^k = [[a^k, k·a^(k−1)·b], [0, a^k]]) und wird mit 0,9^k verglichen.")
c1, c2 = st.columns(2)
e2_a = c1.slider("Eigenwert a", *bounds("e2_a"), step=0.01, key="e2_a", help="Beide Eigenwerte der Matrix. Für a < 1 gehen alle Bahnen gegen 0.")
e2_b = c2.slider("Kopplung b (Eintrag rechts oben)", *bounds("e2_b"), step=0.1, key="e2_b", help="Bei b = 0 ist die Matrix diagonal (normal), mit wachsendem b immer weniger normal.")
an2 = _buckel(float(e2_a), float(e2_b))
s2 = an2.summary
m1, m2, m3, m4 = st.columns(4)
m1.metric("Spektralradius ρ", de(s2.rho, 2))
m2.metric("Spitze von ‖A^k‖", de(s2.peak, 3), delta=f"bei k = {s2.k_peak}", delta_color="off")
m3.metric("Erstes k nach der Spitze unter 1", "–" if s2.first_below_one is None else str(s2.first_below_one), help="Bei ρ ≥ 1 gibt es keins; bei ρ < 1 ohne Buckel ist es 1.")
m4.metric(f"‖A^{C.RATIO_K}‖ gegen ρ^{C.RATIO_K}", "–" if s2.ratio is None else f"{E.de_sig(s2.ratio, 3)}-fach", help="Verhältnis der Norm zur Vorhersage durch den Eigenwert allein.")
st.plotly_chart(build_buckel(an2, int(k_steps)), width="stretch", key="e2_curve")
kind, text = E.verdict_buckel(an2)
getattr(st, kind)(text)

# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 3: Gradientenabstieg
# ---------------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Experiment 3: Gradientenabstieg auf der Quadratik")
st.caption(
    "f(x, y) = (x² + κ·y²)/2 mit den Krümmungen 1 und κ, Start (1, 1). Der Schritt x → x − η·∇f ist die lineare Iteration mit der **symmetrischen** Matrix I − ηH: "
    "hier reichen die Eigenwerte, der Weg ist monoton. Ausführlich zur Schwelle und zum Rückwärtsverfolgen: "
    "[gradient-descent-demo](https://github.com/sebastian-hanisch/gradient-descent-demo)."
)
c1, c2, c3 = st.columns([2, 2, 1])
e3_kappa = c1.slider("Konditionszahl κ = λmax/λmin", *bounds("e3_kappa"), step=1.0, key="e3_kappa", help="Verhältnis der größten zur kleinsten Krümmung.")
e3_frac = c2.slider("Schritt η als Anteil der Schwelle 2/λmax", *bounds("e3_frac"), step=0.001, format="%.3f", key="e3_frac",
                    help="1,000 ist genau die Schwelle. Das Minimum der Rate liegt bei κ/(κ + 1).")
c3.write("")
c3.button("Bester Schritt", on_click=set_best_step, key="btn_best_step", width="stretch", help="Setzt den Schritt auf 2/(λmin + λmax), das ist der Anteil κ/(κ + 1) der Schwelle.")
g3 = _gd(float(e3_kappa), float(e3_frac))
m1, m2, m3 = st.columns(3)
m1.metric("Rate max|1 − ηλ|", de(g3.rate, 4))
m2.metric("Schritte bis ‖x‖ < 10⁻⁶", "–" if g3.steps is None else str(g3.steps))
m3.metric("Faktoren der beiden Richtungen", f"{de(g3.factors[0], 3)} und {de(g3.factors[1], 3)}")
col_a, col_b = st.columns(2)
col_a.plotly_chart(build_gd_path(g3, int(k_steps)), width="stretch", key="e3_path")
col_b.plotly_chart(build_gd_rate(g3), width="stretch", key="e3_rate")
kind, text = E.verdict_gd(g3)
getattr(st, kind)(text)

# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 4: Banach-Schranke
# ---------------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Experiment 4: Die Banach-Schranke")
st.caption(
    "Bildet f ein Intervall in sich ab und ist |f′| ≤ L < 1 darauf, gibt es genau einen Fixpunkt x*, und es gilt |x_k − x*| ≤ L^k/(1 − L)·|x₁ − x₀|. "
    "L ist eine Eigenschaft **auf dem Intervall** (die Selbstabbildung gehört zur Voraussetzung): cos hat auf ganz ℝ keine Konstante unter 1."
)
c1, c2 = st.columns(2)
e4_map = c1.selectbox("Abbildung f", C.E4_MAP_KEYS, format_func=lambda key: C.E4_MAP_LABELS[key], key="e4_map")
e4_pos = c2.slider("Start x₀ im Intervall (0 = linker, 1 = rechter Rand)", *bounds("e4_pos"), step=0.01, key="e4_pos")
b4 = _banach(str(e4_map), float(e4_pos), int(k_steps))
m1, m2, m3, m4 = st.columns(4)
m1.metric("Start x₀", de(b4.x0, 4))
m2.metric("Lipschitz-Konstante L", de(b4.L, 4), help="Größtes |f′| auf dem Intervall, an 20001 Stützstellen bestimmt.")
m3.metric("Garantierte Schritte bis 10⁻⁶", str(b4.k_guaranteed))
m4.metric("Tatsächliche Schritte", "–" if b4.k_actual is None else str(b4.k_actual))
st.plotly_chart(build_banach(b4), width="stretch", key="e4_curve")
kind, text = E.verdict_banach(b4)
getattr(st, kind)(text)

# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 5: Momentum
# ---------------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Experiment 5: Gleiche Eigenwerte, anderer Weg (Momentum)")
st.caption(
    "Gradientenabstieg mit Momentum (Heavy Ball) auf einer Krümmung λ: x_(k+1) = x_k − η·λ·x_k + β·(x_k − x_(k−1)). Als Zustand (x_k, x_(k−1)) ist das eine lineare Iteration mit der "
    "**nicht-symmetrischen** Matrix M = [[1 + β − h, −β], [1, 0]], h = η·λ. Ohne Momentum (β = 0) ist der Zustand ein Skalar und die Norm fällt monoton."
)
c1, c2, c3 = st.columns([2, 2, 1])
e5_beta = c1.slider("Momentum β", *bounds("e5_beta"), step=0.01, key="e5_beta")
e5_h = c2.slider("h = η·λ", *bounds("e5_h"), step=0.001, format="%.3f", key="e5_h", help="Der Spektralradius ist unter 1 genau für 0 < h < 2(1 + β).")
c3.write("")
c3.button("Kritisch dämpfen", on_click=set_critical_h, key="btn_critical", width="stretch", help="Setzt h = (1 − √β)²: doppelter Eigenwert √β.")
m5 = _momentum(float(e5_beta), float(e5_h), int(k_steps))
ms = m5.summary
m1, m2, m3, m4 = st.columns(4)
m1.metric("Spektralradius ρ", de(ms.rho, 3))
m2.metric("Eigenwerte", " und ".join(f"{de(z.real, 3)}" + (f" {'+' if z.imag >= 0 else '−'} {de(abs(z.imag), 3)}i" if abs(z.imag) > 1e-12 else "") for z in m5.eigenvalues))
m3.metric("Spitze von ‖M^k‖", de(ms.peak, 3), delta=f"bei k = {ms.k_peak}", delta_color="off")
m4.metric("Kritisch gedämpft bei h =", de(m5.critical_h, 4))
st.plotly_chart(build_momentum_curve(m5, int(k_steps)), width="stretch", key="e5_curve")
kind, text = E.verdict_momentum(m5)
getattr(st, kind)(text)

rc = _raster_count()
st.markdown("**Wie häufig ist der Buckel?** Zählung über ein festes Raster stabiler Punkte (β von 0,02 bis 0,98 in 49 Werten, je 80 Werte h im Innern von (0; 2(1 + β)), Spitze über k = 0 bis 2000):")
r1, r2, r3 = st.columns(3)
r1.metric("Stabile Rasterpunkte", str(rc.total))
r2.metric("davon mit Spitze über 2", f"{rc.above2} ({de(100.0 * rc.above2 / rc.total, 1)} %)")
r3.metric("davon mit Spitze über 10", str(rc.above10))
st.caption(
    f"Reelle Eigenwerte: {rc.real_above2} von {rc.real_total} Punkten mit Spitze über 2; komplexe: {rc.complex_above2} von {rc.complex_total}. Größte Spitze {de(rc.max_peak, 1)} bei β = {de(rc.max_at[0], 2)}, h = {de(rc.max_at[1], 4)}. "
    "Das ist eine Zählung für **dieses Raster**, kein Anteil aller Parameter."
)
st.plotly_chart(build_raster(float(e5_beta), float(e5_h)), width="stretch", key="e5_raster")

# ---------------------------------------------------------------------------------------------------------------------------------------
# Kernfrage
# ---------------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.markdown("## ❓ Kernfrage: Reichen die Eigenwerte?")
rows = _core_rows()
st.markdown("Spitze von ‖A^k‖ (k = 0 bis 399) für A = [[a, b], [0, a]]; in jeder Zelle ist der Spektralradius a kleiner als 1:")
st.dataframe(
    {"a": [de(r["a"], 2) for r in rows], **{f"b = {de(b, 0)}": [de(r["peaks"][i], 2) for r in rows] for i, b in enumerate(E.CORE_B)}},
    hide_index=True,
    width="stretch",
)
kind, text = E.verdict_core(rows)
getattr(st, kind)(text)

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------------------------------
# Grenzen
# ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Iteration ist linear** | Bei einer nichtlinearen Regel gilt der Eigenwert nur nahe am Ruhepunkt; was der Buckel dort bewirkt (etwa ein kleineres Einzugsgebiet), ist hier **nicht gemessen**. | [logistische-abbildung-demo](https://github.com/sebastian-hanisch/logistische-abbildung-demo): nichtlinear, ein Zustand |
| **Die Matrix ist normal** (hier: symmetrisch, wie beim Gradientenabstieg ohne Momentum) | Sonst kann ‖A^k‖ steigen, obwohl ρ < 1 (Experiment 2 und 5); für kurze Zeiten reichen die Eigenwerte dann nicht. | – |
| **Eine feste Matrix, exakte Rechnung** | Die Zahlen gelten für diese Matrizen in Gleitkomma (Float64); Jacobi- und Gauß-Seidel-Verfahren auf nichtsymmetrischen Systemen sind **nicht gemessen**. | – |
| **Eine Krümmung je Richtung, quadratische Zielfunktion** | Bei Netzen und anderen nichtlinearen Problemen ist die Schwelle 2/η nur eine Näherung; dort setzt die Dynamik der Schrittweite an. | [schrittweite-dynamik-demo](https://github.com/sebastian-hanisch/schrittweite-dynamik-demo) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist eine Regel $x_{k+1} = F(x_k)$ und ein Startwert $x_0$; gesucht ist, ob und wie schnell die Folge $x_k$ gegen einen Ruhepunkt $x^*$ mit $F(x^*) = x^*$ geht. Dieses Stück betrachtet nur lineare Regeln $F(x) = Ax + c$
(und die Zahl als Sonderfall $A = (a)$), die Banach-Schranke für eine Abbildung $f$ einer Zahl und die Zustandsmatrix $M$ des Momentums. Es ist ein vollständiges Modell für diese fünf Fälle; Aussagen über nichtlineare Regeln folgen aus ihm nicht.

| Symbol | Bedeutung |
|---|---|
| $A$ | Iterationsmatrix; $\rho(A)$ ihr Spektralradius (größter Betrag eines Eigenwerts) |
| $e_k = x_k - x^*$ | Abstand der Bahn zum Ruhepunkt |
| $a, b$ | Einträge von $A = \begin{pmatrix} a & b \\ 0 & a \end{pmatrix}$ in Experiment 2 |
| $\lambda_{\min} = 1,\ \lambda_{\max} = \kappa$ | Krümmungen der Quadratik, $\kappa$ die Konditionszahl |
| $\eta$ | Schritt des Gradientenabstiegs |
| $L,\ [l, r]$ | Lipschitz-Konstante von $f$ auf dem Intervall $[l, r]$ |
| $\beta,\ h = \eta\lambda$ | Momentum und Schritt mal Krümmung |
| $M$ | Zustandsmatrix des Momentums für $(x_k, x_{k-1})$ |

**Lineare Iteration.** Für $x_{k+1} = Ax_k + c$ mit $I - A$ invertierbar ist $x^* = (I - A)^{-1}c$, und der Abstand erfüllt

$$e_k = A^k e_0 .$$

Es gilt $e_k \to 0$ für jeden Start genau dann, wenn $\rho(A) < 1$. Immer ist $\|A^k\|_2 \ge \rho(A)^k$; ist $A$ normal (etwa symmetrisch), gilt Gleichheit. Bei einer nicht-normalen Matrix kann $\|A^k\|_2$ für kleine $k$ weit über $\rho(A)^k$ liegen.

**Experiment 2.** Für $A = \begin{pmatrix} a & b \\ 0 & a \end{pmatrix}$ ist

$$A^k = \begin{pmatrix} a^k & k\,a^{k-1} b \\ 0 & a^k \end{pmatrix}, \qquad \|A^k\|_2 = \frac{\sqrt{q^2 + 4p^2} + |q|}{2}, \quad p = a^k,\ q = k\,a^{k-1} b .$$

Die Norm folgt aus den Singulärwerten $s_1 \ge s_2$ mit $s_1 s_2 = |\det| = p^2$ und $s_1^2 + s_2^2 = \|A^k\|_F^2 = 2p^2 + q^2$. Das Maximum über $k$ liegt etwa bei $k \approx -1/\ln a$ (Näherung), sobald $b$ so groß ist, dass überhaupt ein Buckel entsteht.

**Experiment 3.** Für $f(x) = \tfrac12 x^\top H x$ mit $H = \mathrm{diag}(\lambda_{\min}, \lambda_{\max})$ ist $x_{k+1} = (I - \eta H)\,x_k$. Die Matrix ist symmetrisch, die Rate also

$$r(\eta) = \max_{\lambda \in \{\lambda_{\min},\, \lambda_{\max}\}} |1 - \eta\lambda| ,$$

kleiner als 1 genau für $0 < \eta < 2/\lambda_{\max}$, am kleinsten bei

$$\eta^* = \frac{2}{\lambda_{\min} + \lambda_{\max}}, \qquad r(\eta^*) = \frac{\kappa - 1}{\kappa + 1}, \quad \kappa = \frac{\lambda_{\max}}{\lambda_{\min}} .$$

**Experiment 4 (Banach).** Bildet $f$ das Intervall $[l, r]$ in sich ab und gilt $|f'(x)| \le L < 1$ darauf, so hat $f$ dort genau einen Fixpunkt $x^*$, und für $x_{k+1} = f(x_k)$ gilt

$$|x_k - x^*| \le \frac{L^k}{1 - L}\,|x_1 - x_0| .$$

Die Zahl der garantierten Schritte bis zum Fehler $\varepsilon$ ist das kleinste $k$ mit $\frac{L^k}{1 - L}|x_1 - x_0| < \varepsilon$.

**Experiment 5 (Momentum).** Auf einer Krümmung $\lambda$ lautet der Schritt $x_{k+1} = x_k - \eta\lambda x_k + \beta (x_k - x_{k-1})$, also mit $h = \eta\lambda$

$$\begin{pmatrix} x_{k+1} \\ x_k \end{pmatrix} = M \begin{pmatrix} x_k \\ x_{k-1} \end{pmatrix}, \qquad M = \begin{pmatrix} 1 + \beta - h & -\beta \\ 1 & 0 \end{pmatrix} .$$

Die Eigenwerte lösen $z^2 - (1 + \beta - h)\,z + \beta = 0$; es gilt $\rho(M) < 1$ genau für $0 \le \beta < 1$ und $0 < h < 2(1 + \beta)$. Der doppelte Eigenwert $\sqrt\beta$ tritt bei $h = (1 - \sqrt\beta)^2$ auf; dann ist
$N = M - \sqrt\beta\, I$ nilpotent ($N^2 = 0$) und

$$M^k = \beta^{k/2} I + k\,\beta^{(k-1)/2} N .$$

**Literatur.** Banachscher Fixpunktsatz und Spektralradius sind Lehrbuchwissen (nicht anhand einer bestimmten Quelle geprüft); die Norm-Formel ist hier hergeleitet und in den Tests gegen numpy geprüft.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Dynamische Systeme: Ruhepunkt, Kreislauf, Chaos](https://sebastianhanisch.net/konzepte-dynamische-systeme.html)."
)

sync_query_params({key: st.session_state[key] for key in SETTING_SPECS})
