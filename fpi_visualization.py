"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.

Logarithmische Achsen bekommen nur positive Werte (Nullen werden zu Lücken), sonst zeichnet Plotly nichts oder einen falschen Bereich."""

import numpy as np
import plotly.graph_objects as go

import fpi_algorithm as A
import fpi_constants as C

FLOOR = 1e-300


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _positive(values):
    """Für logarithmische Achsen: nicht positive oder nicht endliche Werte werden zu Lücken."""
    v = np.asarray(values, dtype=float)
    return np.where(np.isfinite(v) & (v > 0), v, np.nan)


# ---------------------------------------------------------------- Experiment 1
def cobweb_window(an):
    """Fenster [lo, hi] für Spinnweb und Gerade: um den Ruhepunkt (bzw. den Start) mit dem 1,5-Fachen des Abstands, mindestens ±1,5."""
    center = an.xstar if an.xstar is not None else an.x0
    half = 1.5 * max(abs(an.x0 - center), 1.0)
    return center - half, center + half


def build_cobweb(an, web_steps=C.WEB_STEPS):
    lo, hi = cobweb_window(an)
    xs = np.array([lo, hi])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=xs, mode="lines", line=dict(color=C.COLOR_GREY, width=2, dash="dash"), name="y = x", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=an.a * xs + an.c, mode="lines", line=dict(color=C.COLOR_MAIN, width=3), name="y = a·x + c",
                             hovertemplate="x = %{x:.3f}: f(x) = %{y:.3f}<extra></extra>"))
    wx, wy = A.cobweb_points(an.a, an.c, an.x0, web_steps, lo, hi)
    fig.add_trace(go.Scatter(x=wx, y=wy, mode="lines+markers", line=dict(color=C.COLOR_REF, width=1.5), marker=dict(size=4), name=f"Spinnweb (bis {web_steps} Schritte)",
                             hoverinfo="skip"))
    if an.xstar is not None and lo <= an.xstar <= hi:
        fig.add_trace(go.Scatter(x=[an.xstar], y=[an.xstar], mode="markers", marker=dict(size=11, symbol="diamond", color="black"), name="Ruhepunkt x*",
                                 hovertemplate="x* = %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="x", range=[lo, hi])
    fig.update_yaxes(title_text="f(x)", range=[lo, hi])
    return _base(fig, 420)


def build_affine_series(an):
    ks = np.arange(len(an.xs))
    fig = go.Figure()
    if an.xstar is not None:
        dist = _positive(np.abs(an.xs - an.xstar))
        fig.add_trace(go.Scatter(x=ks, y=dist, mode="lines+markers", line=dict(color=C.COLOR_MAIN, width=2), marker=dict(size=4), name="Abstand |x_k − x*|",
                                 hovertemplate="k = %{x}: %{y:.3e}<extra></extra>"))
        fig.update_yaxes(title_text="Abstand zum Ruhepunkt (logarithmisch)", type="log")
    else:
        fig.add_trace(go.Scatter(x=ks, y=an.xs, mode="lines+markers", line=dict(color=C.COLOR_MAIN, width=2), marker=dict(size=4), name="x_k",
                                 hovertemplate="k = %{x}: %{y:.4f}<extra></extra>"))
        fig.update_yaxes(title_text="x_k")
    fig.update_xaxes(title_text="Schritt k")
    return _base(fig, 300)


# ---------------------------------------------------------------- Experiment 2 und 5 (Norm der Potenzen)
def build_norm_curve(norms, rho, steps, k_peak, peak, ref_label, norm_label):
    """Norm der Potenzen gegen ρ^k (logarithmisch), mit Linie bei 1 und Markierung der Spitze (liegt sie außerhalb des Fensters, entfällt die Markierung)."""
    steps = min(steps, len(norms) - 1)
    ks = np.arange(steps + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=_positive(norms[: steps + 1]), mode="lines", line=dict(color=C.COLOR_MAIN, width=3), name=norm_label,
                             hovertemplate="k = %{x}: %{y:.4g}<extra></extra>"))
    ref = np.power(float(rho), ks) if rho > 0 else np.zeros(len(ks))
    fig.add_trace(go.Scatter(x=ks, y=_positive(ref), mode="lines", line=dict(color=C.COLOR_REF, width=2, dash="dash"), name=ref_label,
                             hovertemplate="k = %{x}: %{y:.4g}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, steps], y=[1.0, 1.0], mode="lines", line=dict(color=C.COLOR_GREY, width=1), name="1", hoverinfo="skip"))
    if peak > 1.0 + 1e-12 and k_peak <= steps:
        fig.add_trace(go.Scatter(x=[k_peak], y=[peak], mode="markers+text", marker=dict(size=11, color="black", symbol="diamond"),
                                 text=[f"Spitze {peak:.4g} bei k = {k_peak}".replace(".", ",")], textposition="top right", name="Spitze",
                                 hovertemplate="k = %{x}: %{y:.4g}<extra></extra>"))
    fig.update_xaxes(title_text="Schritt k")
    fig.update_yaxes(title_text="Norm (logarithmisch)", type="log")
    return _base(fig)


def build_buckel(an, steps):
    s = an.summary
    return build_norm_curve(an.norms, s.rho, steps, s.k_peak, s.peak, "ρ^k (Eigenwert allein)", "‖A^k‖")


def build_momentum_curve(m, steps):
    s = m.summary
    return build_norm_curve(m.norms, s.rho, steps, s.k_peak, s.peak, "ρ^k (Eigenwert allein)", "‖M^k‖")


# ---------------------------------------------------------------- Experiment 3
def build_gd_path(g, steps):
    path = A.gd_norm_path(g.kappa, g.eta, steps)
    ks = np.arange(steps + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=_positive(path), mode="lines", line=dict(color=C.COLOR_MAIN, width=3), name="‖x_k‖",
                             hovertemplate="k = %{x}: %{y:.3e}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, steps], y=[C.GD_TOL, C.GD_TOL], mode="lines", line=dict(color=C.COLOR_GREY, width=1, dash="dash"), name="10⁻⁶", hoverinfo="skip"))
    fig.update_xaxes(title_text="Schritt k")
    fig.update_yaxes(title_text="‖x_k‖ (logarithmisch)", type="log")
    return _base(fig)


def build_gd_rate(g):
    """Rate gegen den Schritt (als Anteil der Schwelle 2/λmax), mit dem Minimum bei κ/(κ + 1) und dem gewählten Punkt."""
    fracs = np.linspace(C.E3_FRAC_MIN, C.E3_FRAC_MAX, 400)
    rates = A.gd_rate(g.kappa, A.gd_eta_from_fraction(g.kappa, fracs))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fracs, y=rates, mode="lines", line=dict(color=C.COLOR_LINE, width=3), name="Rate max|1 − ηλ|",
                             hovertemplate="η = %{x:.3f}·Schwelle: Rate %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[C.E3_FRAC_MIN, C.E3_FRAC_MAX], y=[1.0, 1.0], mode="lines", line=dict(color=C.COLOR_GREY, width=1, dash="dash"), name="Rate 1", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[g.best_frac], y=[g.best_rate], mode="markers", marker=dict(size=11, symbol="diamond", color="black"), name="Minimum (κ − 1)/(κ + 1)",
                             hovertemplate="bester Schritt: %{x:.4f}·Schwelle, Rate %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[g.frac], y=[min(g.rate, 1.5)], mode="markers", marker=dict(size=11, color=C.COLOR_REF), name="gewählter Schritt",
                             hovertemplate="gewählt: %{x:.4f}·Schwelle, Rate %{y:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Schritt η als Anteil der Schwelle 2/λmax")
    fig.update_yaxes(title_text="Rate", range=[0, 1.5])
    return _base(fig)


# ---------------------------------------------------------------- Experiment 4
def build_banach(b):
    ks = np.arange(b.steps_shown + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=_positive(b.bounds), mode="lines", line=dict(color=C.COLOR_REF, width=2, dash="dash"), name="A-priori-Schranke L^k/(1 − L)·|x₁ − x₀|",
                             hovertemplate="k = %{x}: %{y:.3e}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ks, y=_positive(b.errors), mode="lines+markers", line=dict(color=C.COLOR_MAIN, width=3), marker=dict(size=4), name="tatsächlicher Fehler |x_k − x*|",
                             hovertemplate="k = %{x}: %{y:.3e}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, b.steps_shown], y=[C.BANACH_TOL, C.BANACH_TOL], mode="lines", line=dict(color=C.COLOR_GREY, width=1), name="10⁻⁶", hoverinfo="skip"))
    fig.update_xaxes(title_text="Schritt k")
    fig.update_yaxes(title_text="Fehler (logarithmisch)", type="log")
    return _base(fig)


# ---------------------------------------------------------------- Experiment 5: Raster
def build_raster(beta, h):
    """Alle 3920 Rasterpunkte (β, h), gefärbt nach der Spitze max_k ‖M^k‖; der gewählte Punkt ist markiert. Die Spitzen werden einmal gerechnet (zwischengespeichert)."""
    B, H, peaks = raster_data()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=B.ravel(), y=H.ravel(), mode="markers", marker=dict(size=5, color=np.minimum(peaks.ravel(), 10.0), colorscale="Viridis", cmin=1.0, cmax=10.0,
                                                                                    colorbar=dict(title="Spitze", tickvals=[1, 2, 4, 6, 8, 10])),
                             name="Rasterpunkt", hovertemplate="β = %{x:.2f}, h = %{y:.3f}: Spitze %{marker.color:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[beta], y=[h], mode="markers", marker=dict(size=13, color="red", symbol="x"), name="gewählt", hoverinfo="skip"))
    fig.update_xaxes(title_text="Momentum β")
    fig.update_yaxes(title_text="h = η·λ")
    return _base(fig, 380)


_RASTER = {}


def raster_data():
    """(β-Feld, h-Feld, Spitzen-Feld) des Rasters; beim ersten Aufruf gerechnet (0,1 s), danach aus dem Speicher des Prozesses."""
    if not _RASTER:
        B, H = A.heavy_ball_raster()
        peaks, _ = A.heavy_ball_peaks(B, H, C.RASTER_KMAX)
        _RASTER.update(B=B, H=H, P=peaks)
    return _RASTER["B"], _RASTER["H"], _RASTER["P"]
