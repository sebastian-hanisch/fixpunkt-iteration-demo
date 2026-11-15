"""Permalink (Begrenzen), Schnellstart-Szenarien und die beiden Knöpfe „bester Schritt“ und „kritisch dämpfen“ (Standardmuster aus dem Demo-Portfolio).

Die Regler der Experimente stehen im Hauptbereich beim Experiment, die allgemeine Einstellung (Schritte im Diagramm) in der Seitenleiste; alle haben einen Eintrag in
SETTING_SPECS, damit der Schnellstart ein Szenario genau dann hervorhebt, wenn jeder Regler seinen Wert hat (tests/test_layout.py). Es gibt keinen Zufallsanteil
und deshalb keinen Zufalls-Seed."""

from dataclasses import dataclass

import streamlit as st

import fpi_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "k_steps": SettingSpec("k", int, C.DEFAULT_K_STEPS, C.K_STEPS_MIN, C.K_STEPS_MAX),
    "e1_a": SettingSpec("e1a", float, C.DEFAULT_E1_A, C.E1_A_MIN, C.E1_A_MAX),
    "e1_c": SettingSpec("e1c", float, C.DEFAULT_E1_C, C.E1_C_MIN, C.E1_C_MAX),
    "e1_x0": SettingSpec("e1x", float, C.DEFAULT_E1_X0, C.E1_X0_MIN, C.E1_X0_MAX),
    "e2_a": SettingSpec("e2a", float, C.DEFAULT_E2_A, C.E2_A_MIN, C.E2_A_MAX),
    "e2_b": SettingSpec("e2b", float, C.DEFAULT_E2_B, C.E2_B_MIN, C.E2_B_MAX),
    "e3_kappa": SettingSpec("e3k", float, C.DEFAULT_E3_KAPPA, C.E3_KAPPA_MIN, C.E3_KAPPA_MAX),
    "e3_frac": SettingSpec("e3f", float, C.DEFAULT_E3_FRAC, C.E3_FRAC_MIN, C.E3_FRAC_MAX),
    "e4_map": SettingSpec("e4m", str, C.DEFAULT_E4_MAP, choices=C.E4_MAP_KEYS),
    "e4_pos": SettingSpec("e4p", float, C.DEFAULT_E4_POS, C.E4_POS_MIN, C.E4_POS_MAX),
    "e5_beta": SettingSpec("e5b", float, C.DEFAULT_E5_BETA, C.E5_BETA_MIN, C.E5_BETA_MAX),
    "e5_h": SettingSpec("e5h", float, C.DEFAULT_E5_H, C.E5_H_MIN, C.E5_H_MAX),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _clean(spec, raw):
    """Wert aus der Adresszeile: umwandeln, auf die Reglergrenzen begrenzen, Auswahlwerte nur aus der erlaubten Liste; sonst None."""
    try:
        value = spec.caster(raw)
    except (ValueError, TypeError):
        return None
    if spec.choices is not None:
        return value if value in spec.choices else None
    if value != value:   # NaN
        return None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            value = _clean(spec, qp[spec.url_param])
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Setzt alle Regler auf das Szenario: die genannten auf ihren Wert, alle übrigen auf ihre Voreinstellung (sonst bliebe ein Rest des vorigen Szenarios stehen)."""
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = C.PRESETS[name].get(state_key, spec.default)


def aktives_preset():
    """Name des Szenarios, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Szenario ohne Eintrag für einen Regler meint dessen Voreinstellung) mit dem Session State."""
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(state_key, spec.default) == preset.get(state_key, spec.default) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Szenario ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden (es setzt alle Regler der Experimente):")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def set_best_step():
    """Knopf im Experiment 3: Schritt auf den besten Wert 2/(λmin + λmax), das ist der Anteil κ/(κ + 1) der Schwelle."""
    kappa = float(st.session_state.get("e3_kappa", C.DEFAULT_E3_KAPPA))
    st.session_state["e3_frac"] = kappa / (kappa + 1.0)


def critical_h_for_slider(beta: float) -> float:
    """Kritisch gedämpftes h = (1 − √β)², auf den Reglerbereich begrenzt (bei β = 0,98 liegt es mit 10⁻⁴ unter dem kleinsten Reglerwert)."""
    return min(max((1.0 - beta ** 0.5) ** 2, C.E5_H_MIN), C.E5_H_MAX)


def set_critical_h():
    """Knopf im Experiment 5: h auf den kritisch gedämpften Wert (1 − √β)²."""
    st.session_state["e5_h"] = critical_h_for_slider(float(st.session_state.get("e5_beta", C.DEFAULT_E5_BETA)))
