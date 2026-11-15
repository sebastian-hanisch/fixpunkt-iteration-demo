"""Schnellstart-Szenarien: Vollständigkeit, gültige Werte, das erste Szenario entspricht den Voreinstellungen, Adresszeile (Permalink) begrenzt Werte;
keine Voreinstellung ist eine Zweierpotenz."""

import math

import pytest

import fpi_constants as C
import fpi_evaluation as E
import fpi_presets as P


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.SETTING_SPECS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_choices():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.choices is not None:
                assert value in spec.choices, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_first_preset_is_the_default_state():
    first = next(iter(C.PRESETS))
    assert first == C.DEFAULT_PRESET and C.PRESETS[first] == {}
    assert E.Settings() == E.Settings(**{key: spec.default for key, spec in P.SETTING_SPECS.items()})


def test_presets_differ_pairwise_in_at_least_one_setting():
    full = {name: tuple(p.get(key, spec.default) for key, spec in P.SETTING_SPECS.items()) for name, p in C.PRESETS.items()}
    assert len(set(full.values())) == len(full)


def test_setting_specs_match_the_settings_dataclass():
    assert set(P.SETTING_SPECS) == set(E.Settings.__dataclass_fields__)


def test_no_default_is_a_power_of_two():
    """Dualbrüche (0,5; 1; 2; 0,25 …) sind in Gleitkomma exakt und erzeugen Sonderfälle (Rechner-Artefakt der Linie); Voreinstellungen sind keine."""
    for key, spec in P.SETTING_SPECS.items():
        v = spec.default
        if isinstance(v, str) or key == "k_steps":
            continue
        m, _ = math.frexp(abs(float(v)))
        assert m != 0.5, (key, v)


def test_the_deliberate_exceptions_are_documented_in_the_presets():
    """Zweierpotenzen kommen nur dort vor, wo sie die Sache selbst sind: die Schwelle (Anteil 1,0 von 2/λmax) und der Start 1 bei cos (Handbeispiel H3)."""
    assert C.PRESETS["GD an der Schwelle"]["e3_frac"] == 1.0
    assert C.PRESETS["cos (Banach)"]["e4_pos"] == 1.0
    for name, p in C.PRESETS.items():
        for key, v in p.items():
            if isinstance(v, str) or v == 0.0:
                continue
            m, _ = math.frexp(abs(float(v)))
            if m == 0.5:
                assert (name, key) in {("GD an der Schwelle", "e3_frac"), ("cos (Banach)", "e4_pos")}, (name, key, v)


def test_permalink_values_are_clamped_and_invalid_ones_ignored():
    spec = P.SETTING_SPECS["e2_a"]
    assert P._clean(spec, "5.0") == spec.hi and P._clean(spec, "-3") == spec.lo and P._clean(spec, "0.9") == 0.9
    assert P._clean(spec, "abc") is None and P._clean(spec, "nan") is None
    assert P._clean(P.SETTING_SPECS["e4_map"], "wurzel") == "wurzel" and P._clean(P.SETTING_SPECS["e4_map"], "tan") is None
    assert P._clean(P.SETTING_SPECS["k_steps"], "9999") == C.K_STEPS_MAX


def test_buttons_set_the_best_step_and_the_critical_h():
    assert P.critical_h_for_slider(0.81) == pytest.approx(0.01, abs=1e-15)
    for kappa in (2.0, 10.0, 100.0):
        assert C.E3_FRAC_MIN < kappa / (kappa + 1.0) < C.E3_FRAC_MAX
