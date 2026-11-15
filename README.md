# Iterieren und Fixpunkt – Eigenwerte sagen den Grenzwert, nicht den Weg – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-fixpunkt-iteration-demo.streamlit.app/)**

Stück 1 der Linie **Dynamische Systeme** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, und zwar die Wurzel: Es gibt keinen Vorgänger. Die Frage der Linie ist, was passiert, wenn man eine Regel immer wieder auf ihr eigenes Ergebnis anwendet. Dieses Stück bleibt bei **linearen** Iterationen x → A·x; im nächsten Stück ([logistische-abbildung-demo](https://github.com/sebastian-hanisch/logistische-abbildung-demo)) macht die Nichtlinearität aus dem Eigenwert −1 die Periodenverdopplung, und [schrittweite-dynamik-demo](https://github.com/sebastian-hanisch/schrittweite-dynamik-demo) führt die Schwelle 2/λmax des Gradientenabstiegs in die Dynamik.

Vehikel sind eine Zahl, eine 2×2-Matrix, der Gradientenabstieg auf einer Quadratik und der Gradientenabstieg mit Momentum (Heavy Ball). Die Rechnung ist durchweg Float64 mit numpy; Gegenproben sind geschlossene Formeln, die Matrixpotenz mit Singulärwertzerlegung, exakte Brüche und scipy.

## Warum dieses Problem

**Wiederholt man x → A·x, entscheidet der Spektralradius ρ(A), ob die Bahn für alle Startwerte gegen den Ruhepunkt geht (genau dann, wenn ρ < 1). Er sagt aber nichts über den Weg dorthin:** Bei einer nicht-normalen Matrix mit ρ = 0,9 wächst die Norm zuerst auf das 19-Fache, und ein Gradientenverfahren mit Momentum am kritisch gedämpften Punkt zeigt in der Norm des Zustands ebenfalls einen Buckel (hier das 7-Fache). Ohne Momentum ist die Iterationsmatrix des Gradientenabstiegs auf einer Quadratik symmetrisch und der Weg monoton: Dort reichen die Eigenwerte (Rate max|1 − ηλ|, bester Schritt 2/(λmin + λmax)).

Alle Aussagen gelten für lineare Iterationen und für die in der Demo gewählten Matrizen; sie sagen nichts über Netze oder nichtlineare Probleme.

## Modell

Fünf Experimente (Formeln im 📐-Expander der App; sie sind dort ein vollständiges Modell für genau diese fünf Fälle):

1. **Eine Zahl iterieren:** x → a·x + c mit Spinnweb und Abstand zum Ruhepunkt x* = c/(1 − a). Verdikt nach |a| (unter 1, genau 1, über 1).
2. **Eigenwert gegen Weg:** A = [[a, b], [0, a]], beide Eigenwerte a. A^k in geschlossener Form, ‖A^k‖₂ = (√(q² + 4p²) + |q|)/2 mit p = a^k, q = k·a^(k−1)·b, gegen ρ^k. Spitze, Lage der Spitze, erstes k darunter, Verhältnis bei k = 100.
3. **Gradientenabstieg auf der Quadratik** f = (x² + κy²)/2, Start (1, 1): Faktoren 1 − η und 1 − ηκ, Rate, Schritte bis ‖x‖ < 10⁻⁶, Rate gegen Schritt mit Minimum (κ − 1)/(κ + 1) bei 2/(λmin + λmax). Der Schritt ist als Anteil der Schwelle 2/λmax einstellbar (die Schwelle selbst ist der Anteil 1). Zur Schwelle ausführlich: [gradient-descent-demo](https://github.com/sebastian-hanisch/gradient-descent-demo).
4. **Banach-Schranke** für drei Abbildungen mit je einem Intervall, das sie in sich abbildet: cos x auf [cos 1, 1], 0,6·x + 1 auf [0, 5], √(x + 1,5) auf [0, 3]. Die Lipschitz-Konstante L = max |f′| auf dem Intervall wird an 20 001 Stützstellen bestimmt (die Ableitungen sind monoton, der Rand ist Stützstelle), der Fixpunkt durch Bisektion von f(x) − x (nicht durch Iterieren), die Schranke L^k/(1 − L)·|x₁ − x₀| gegen den Fehler. Garantierte Schritte bis 10⁻⁶: kleinstes k mit Schranke unter 10⁻⁶; tatsächliche: kleinstes k mit Fehler unter 10⁻⁶ im Float64-Lauf.
5. **Momentum (Heavy Ball)** auf einer Krümmung: Zustand (x_k, x_(k−1)), M = [[1 + β − h, −β], [1, 0]], h = η·λ. Spektralradius aus Spur und Determinante; ‖M^k‖₂ durch fortlaufendes Multiplizieren; kritisch gedämpft bei h = (1 − √β)². Dazu die Zählung über ein festes Raster stabiler Punkte (49 Werte β von 0,02 bis 0,98, je 80 Werte h im Innern von (0; 2(1 + β)), Spitze über k = 0 bis 2000).

Dazu eine **Kernfrage** (Spitze von ‖A^k‖ für a ∈ {0,6; 0,9; 0,99} und b ∈ {0, 1, 5, 20}) und der Abschnitt „Wo die Annahmen enden“.

## Methodik

- **Eingabe:** Die Regler der Experimente stehen im Hauptbereich bei ihrem Experiment, der Schnellstart (sechs Szenarien) und die allgemeine Einstellung „Schritte im Diagramm K“ in der Seitenleiste. Ein Szenario setzt alle Regler (die nicht genannten auf ihre Voreinstellung). Keine Voreinstellung ist eine Zweierpotenz; die zwei gewollten Ausnahmen sind der Schritt genau an der Schwelle (Anteil 1,0) und der Start 1 bei cos (Handbeispiel).
- **Spitze und „erstes k unter 1“** gelten immer für k = 0 bis 399, unabhängig vom Anzeigefenster K; k = 0 gibt die Norm 1. Ein „Buckel“ heißt hier: Die Norm steigt über 1. Im Raster des Experiments 5 zählt dagegen nur eine Spitze über 2 als Buckel, denn die Norm des Zustands steigt bei β > 0 in allen 3920 stabilen Rasterpunkten über 1 (schon bei k = 1).
- **Float64 überall.** Die Norm der 2×2-Matrix wird aus Frobeniusnorm und Determinante gerechnet (die Einträge werden vorher durch ihren größten Betrag geteilt, damit instabile Parameter nicht überlaufen); die Orakel-Tests vergleichen mit der Singulärwertzerlegung.
- **Banach-Schranke und Rundung:** Die Schranke fällt unter jede Maschinengenauigkeit, der gemessene Fehler nicht. „Die Schranke gilt in n von n Schritten“ zählt deshalb mit einem Rundungsrauschen von 4 Maschinengenauigkeiten (4·ε·max(|x*|, 1)) mit; ohne dieses Rauschen gälte sie bei √(x + 1,5) nur in 38 bis 41 von 400 Schritten (sechs Startlagen), bei 0,6·x + 1 ab der Startlage 0,5 in keinem Schritt und bei cos in allen (`tests/test_claims.py`, `tests/test_edge_cases.py`).
- **Rechenzeit:** Alle Rechnungen sind vektorisiert und laufen live; das Raster (3920 Punkte, 2000 Schritte) wird einmal gerechnet und zwischengespeichert (Messung auf dem Entwicklungsrechner: etwa 0,3 Sekunden, je nach Rechner anders; der Test `test_claim_compute_budget_of_the_live_calculation` begrenzt nur großzügig). Nichts ist vorgerechnet oder eingefroren.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen sind Messungen dieser README, nachgerechnet in `tests/test_claims.py` über dieselben Funktionen wie die App (Band oder exakt, wie angegeben). Voraussetzung jeweils: lineare Iteration, genau diese Matrizen und Parameter, Float64.

| Frage | Befund | Test (Art) |
|---|---|---|
| Spitze von ‖A^k‖ für A = [[0,9; 5], [0; 0,9]] (ρ = 0,9), k = 0 bis 399 | **19,379 bei k = 9** (Näherung −1/ln a = 9,49); die Norm liegt erst ab k = 55 wieder unter 1 | `test_claim_peak_19_379_at_k_9_and_first_below_one_55` (Band ±5·10⁻⁴, k exakt) |
| ‖A^100‖ gegen ρ^100 | 1,476·10⁻² gegen 2,656·10⁻⁵, das **556-Fache** | `test_claim_norm_at_k_100_is_556_times_the_eigenvalue_power` (Band) |
| Kernfrage: Spitze für a = 0,6 / 0,9 / 0,99 bei b = 0, 1, 5, 20 | b = 0: genau 1 in allen drei Zeilen; a = 0,6: 1,3 / 6,0 / 24,0; a = 0,9: 3,9 / 19,4 / 77,5; a = 0,99: 37,0 / 184,9 / 739,5; in allen drei Zeilen steigt die Spitze streng mit b | `test_claim_core_table_twelve_cells`, `test_claim_core_table_peak_is_exactly_one_without_coupling_and_rises_with_b_in_all_rows` (Band ±0,05 bzw. exakt) |
| Erstes k nach der Spitze mit Norm unter 1 | a = 0,6: 4 / 9 / 12; a = 0,9: 35 / 55 / 70 (b = 1, 5, 20); **a = 0,99: in keiner dieser Zellen bis k = 399** (Spitze bei k = 99; ‖A^399‖ für b = 1 noch über 1) | `test_claim_first_k_below_one_for_the_core_cells` (exakt) |
| Gradientenabstieg, κ = 10 und 100: bester Schritt und beste Rate | 0,18182 und 0,01980; Rate (κ − 1)/(κ + 1) = 0,81818 und 0,98020; Scan über 20 000 Schritte liefert 0,81819 (κ = 10) bzw. 0,98020 | `test_claim_best_step_and_best_rate_for_kappa_10_and_100` (Band ±6·10⁻⁶ bzw. Scan-Raster) |
| „Rate (κ − 1)/(κ + 1) ist das Minimum“ | Kein Scan-Wert unter dem Minimum bei 200 Zufalls-κ in [2, 100] (je 2000 Schritte) | `test_claim_no_step_beats_the_best_rate_over_200_random_condition_numbers` (Zählung) |
| Norm nach 2000 Schritten, Start (1, 1), bei 0,99 · 2/λmax / 2/λmax / 1,01 · 2/λmax | κ = 10: 2,8·10⁻¹⁸ / 1,000 / 1,6·10¹⁷; κ = 100: 5,1·10⁻¹⁸ / 1,000 / 1,6·10¹⁷ | `test_claim_threshold_norms_after_2000_steps` (Band ±3 %) |
| Schritte bis ‖x‖ < 10⁻⁶ für κ = 10, Start (1, 1) | 71 beim besten Schritt 2/11 (von Hand: k ≥ ln(√2·10⁶)/ln(11/9) = 70,57); 132 bei η = 1/λmax = 0,1 | `test_claim_steps_to_1e6_for_kappa_10` (exakt) |
| Banach, x → cos x, Start 1 auf [cos 1, 1] | L = sin 1 = 0,841471; Fixpunkt 0,7390851332151607; **garantiert 87 Schritte bis 10⁻⁶, tatsächlich 32**; die Schranke gilt in 120 von 120 Schritten; bei k = 10: Fehler 5,15·10⁻³, Schranke 0,516 | `test_claim_cos_constants_and_step_counts`, `test_claim_cos_at_ten_steps_error_and_bound` (exakt bzw. Band) |
| cos: Fehlerrate | Verhältnis der Fehler bei k = 30 gleich 0,67361 = \|sin x*\| auf fünf Stellen | `test_claim_cos_error_rate_equals_abs_sin_xstar_to_five_digits` (Band 2·10⁻⁵) |
| Banach, 0,6·x + 1 (Start 1) und √(x + 1,5) (Start 0,6) | 0,6·x + 1: garantiert 28, tatsächlich 28, Rate 0,6 gleich L, also **scharfe Schranke**; √(x + 1,5): garantiert 16, tatsächlich 11, Rate 0,2743 gegen L = 0,4082 | `test_claim_other_two_maps_step_counts` (exakt bzw. Band) |
| Schranke und Rundung | Mit 4 Maschinengenauigkeiten Rundungsrauschen gilt sie für alle drei Abbildungen und fünf Startlagen in 400 von 400 Schritten | `test_claim_bound_holds_in_every_step_for_all_three_maps_and_five_starts` (Zählung) |
| Momentum, β = 0,81, h = 0,01 (doppelter Eigenwert 0,9) | N = M − 0,9·I erfüllt N² = 0 (unter 10⁻¹⁵); Spitze von ‖M^k‖ **7,034 bei k = 9**; bei k = 100 das **201-Fache** von 0,9^100 | `test_claim_critical_damping_peak_7_034_at_k_9_and_factor_201` (Band) |
| Raster stabiler Punkte (β von 0,02 bis 0,98 in 49 Werten, je 80 Werte h) | **3920** stabile Punkte; **518 (13,2 %)** mit Spitze über 2 (Schwelle für „Buckel“ im Raster); **0** über 10; reelle Eigenwerte 46 von 496, komplexe 472 von 3424; größte Spitze 8,3 bei β = 0,98, h = 3,9111 | `test_claim_raster_counts` (exakt bzw. Band) |
| Raster: Stabilität der Zählung | Spitzensuche bis k = 300, 600 und 2000 gibt dieselben Zahlen | `test_claim_raster_count_does_not_depend_on_the_length_of_the_search` (exakt) |
| Funktionswert beim Momentum | Mit Start x₀ = x₋₁ = 1 und λ = 1 steigt f_k bei β = 0,81, h = 0,01 in 200 Schritten **kein einziges Mal**; der Buckel steckt im Zustand: ungünstigster Einheitsstart gibt max_k \|x_k\| = 4,986 bei k = 9 | `test_claim_function_value_is_monotone_and_the_worst_start_peak_is_4_986` (exakt bzw. Band) |

Die Raster-Zählung gilt für **dieses Raster**, nicht für „13 % aller Parameter“. Ohne Momentum (β = 0) fällt die Norm für jedes getestete h monoton, weil der Zustand dann ein Skalar ist (`test_claim_without_momentum_no_hump_for_every_h`).

## Befunde und Korrekturen gegenüber dem Plan

- **Verhältnis 556 statt 550:** Der Plan (und die Gliederung) nannten für ‖A^100‖/0,9^100 gerundet 550; gemessen sind 555,6.
- **Kernfrage mit a = 0,6 statt 0,5:** Der Plan nennt a ∈ {0,5; 0,9; 0,99}. Weil Parameter nach dem Auftrag keine Zweierpotenzen sein sollen (Dualbrüche sind in Gleitkomma exakt), steht 0,6 in der Tabelle. Neu gegenüber dem Plan: Bei a = 0,99 liegt die Norm für b = 1, 5, 20 bis k = 399 nirgends wieder unter 1.
- **Experiment 1 zeigt nur die affine Abbildung.** Der Plan nennt dort cos als zweite Abbildung; cos steht stattdessen in Experiment 4 (und im Schnellstart „cos (Banach)“). Grund: Ein Regler, der nur bei einer Abbildung wirkt, wäre ein toter Regler, und ausgeblendete Regler bringen den Zustand von Frontend und Backend auseinander (DEMO-PLAYBOOK §3).
- **Schnellstart „GD bester Schritt (κ 10)“ stellt 0,909 ein, nicht 10/11 genau.** Der Regler hat drei Stellen; die Rate ist dort 0,8182 gegen 0,81818 beim exakten Wert. Der Knopf „Bester Schritt“ setzt den Wert κ/(κ + 1) genau.
- **Neu: die Schranke ist bei der affinen Abbildung scharf** (garantiert gleich tatsächlich, Rate gleich L); bei cos und √(x + 1,5) liegt die wirkliche Rate unter L. Die Aussage „die Schranke ist pessimistisch“ gilt also nicht für jede Abbildung.
- **Neu: Rundungsrauschen der Schranke** (siehe Methodik). Der Plan sah es nicht vor; ohne den Zusatz zählt die App bei großem K „verletzt“, obwohl nur der Gleitkommafehler den Abstand hält.
- **Das Raster wird live gerechnet** (etwa 0,3 Sekunden auf dem Entwicklungsrechner), nicht vorgerechnet; die Zählung ist die der Vormessung (518 von 3920, nachgerechnet mit vektorisierter Rechnung und gegen eine Schleife mit Matrixpotenz an 152 Punkten, `tests/test_oracle_fixpunkt_iteration.py`).
- **Ohne Momentum (β = 0)** wird der Zustand als Skalar 1 − h behandelt (Plan: „Matrix ist ein Skalar“). Als 2×2-Matrix mit der Verzögerungskoordinate hätte die Norm schon bei k = 1 einen scheinbaren Buckel (√(1 + (1 − h)²) > 1), der nur Buchführung wäre.

## Ehrliche Grenzen

- **Nur lineare Iterationen.** Was der Buckel in einem **nichtlinearen** System bewirkt (etwa ein kleineres Einzugsgebiet), ist nicht gemessen und wird nicht behauptet.
- **Nur diese Matrizen.** Der Buckel ist an der 2×2-Matrix [[a, b], [0, a]] und der Heavy-Ball-Matrix gemessen. Jacobi- und Gauß-Seidel-Verfahren auf nichtsymmetrischen Systemen sind nicht gemessen; die Rundung der Norm bei sehr großem k ist nur bis k = 399 geprüft.
- **Momentum nur als Zustandsnorm.** Die Aussage „Momentum macht den Verlust nicht monoton“ ist **nicht** gemessen: Mit dem Standardstart ist der Funktionswert in 200 Schritten monoton (Tabelle oben). Gezeigt wird nur, dass die Norm des Zustands (x_k, x_(k−1)) steigt.
- **Die Raster-Zählung ist eine Zählung für dieses Raster** (β-Bereich, h-Bereich und Raster-Dichte wie angegeben), kein Anteil „aller“ Parameter; der Anteil hängt vom gewählten Bereich ab.
- **Eine Krümmung je Richtung, quadratische Zielfunktion.** Bei Netzen ist die Schwelle 2/η nur eine Näherung; dort setzt die Dynamik der Schrittweite an (schrittweite-dynamik-demo).
- **Quellen:** Banachscher Fixpunktsatz und Spektralradius sind Lehrbuchwissen und hier nicht anhand einer bestimmten Quelle geprüft; die Norm-Formel ist hergeleitet und gegen numpy geprüft. Es gibt keine Literaturwerte, deshalb kein `test_literatur_*`.
- **Die Messzahlen sind Float64 auf einer Plattform;** Bänder statt exakter Gleichheit stehen dort, wo BLAS oder Rundung die letzte Stelle verschieben können.

## Verifikation

Die Suite hat 158 Tests (`pytest --collect-only -q tests`).

- **Kern:** Handbeispiele H1 bis H3 des Plans von Hand nachgerechnet (`tests/test_algorithm.py`: A² = [[0,81; 9], [0; 0,81]], Normformel aus Singulärwerten, Faktoren ±9/11, Lipschitz-Konstante sin 1, Heavy-Ball-Matrix mit N² = 0).
- **Orakel (`tests/test_oracle_fixpunkt_iteration.py`):** ‖A^k‖ gegen `numpy.linalg.matrix_power` mit Singulärwertzerlegung an 7 × 5 Parameterpaaren, gegen die Singulärwertgleichung in exakten Brüchen (`fractions`), Spitze und „erstes k unter 1“ gegen einen Scan; 2×2-Norm gegen SVD an 200 Zufallsmatrizen; Heavy-Ball-Normen gegen Matrixpotenz an 200 Zufallspunkten; Spektralradius gegen `numpy.linalg.eigvals` an 200 Zufallspunkten; Raster-Spitzen gegen eine Schleife an 152 Punkten (jeder 26. Rasterpunkt und der Punkt der größten Spitze); Gradientenabstieg: bester Schritt und Rate gegen einen Scan über 20 000 Schritte und gegen 200 Zufallsspektren; Banach: Fixpunkte gegen `scipy.optimize.brentq`, Lipschitz-Konstanten gegen die Handherleitung.
- **Eigenschaften (`tests/test_eigenschaften_fixpunkt_iteration.py`):** ‖A^k‖ ≥ ρ^k, Skalierung A → cA, normale Matrix ohne Buckel, Spitze wächst mit b, Invarianz der 2×2-Norm unter Drehungen, Linearität im Startabstand, Symmetrie der Raten um den besten Schritt, Monotonie unter der Schwelle, Fehlerkontraktion um L, je 50 Zufallsinstanzen mit festem Zufalls-Seed (SplitMix64).
- **Randfälle (`tests/test_edge_cases.py`):** ρ = 1 mit und ohne Kopplung, a = 0 (nilpotent: ‖A^k‖ = 0 ab k = 2), Schritt genau an der Schwelle (Gleitkomma-Reste), κ = 1, Start im Fixpunkt, Reglerränder, Momentum an der Stabilitätsgrenze und ohne Momentum.
- **Claims (`tests/test_claims.py`):** jede Zahl der Tabelle oben; **Verdikte** (`tests/test_verdicts.py`) als reine Funktionen an handgesetzten Werten, jede Zahl im Text.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an Minimum und Maximum), Schnellstart in der Seitenleiste und aktives Szenario hervorgehoben, jede Regleränderung hebt die Markierung auf, die zwei Knöpfe „Bester Schritt“ und „Kritisch dämpfen“ (`tests/test_layout.py`), Footer, Laufzeit-Importe.
- **Zweierpotenz-Falle:** Parameter a, β, h, c, η sind nie Zweierpotenzen voreingestellt (`tests/test_presets.py`); `tests/test_claims.py` rechnet zusätzlich mit a = 0,9 und a + 10⁻⁹.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: fünf Experimente, Kernfrage, Annahmen, Formeln |
| `fpi_algorithm.py` | Kern: ‖A^k‖ in geschlossener Form, Gradientenabstieg, Banach-Schranke, Heavy Ball mit Raster |
| `fpi_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Schnellstart-Szenarien |
| `fpi_evaluation.py` | Kennzahlen, Verdikte (reine Funktionen), Kernfrage-Tabelle |
| `fpi_layout.py` | Fester Kopf (Stylesheet) |
| `fpi_presets.py` | Permalink (Begrenzen), Schnellstart (aktives Szenario hervorgehoben), die zwei Knöpfe der Experimente |
| `fpi_rng.py` | SplitMix64, nur für die Zufallsfälle der Tests (Kopie aus `saal-abstand-demo`) |
| `fpi_scenario.py` | Die drei Banach-Abbildungen mit Intervall und Ableitung |
| `fpi_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur für das Orakel) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Randfälle, Claims, Verdikte, Schnellstart, Layout, AppTest-Rauchtest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Nichtlineare Folgen des Buckels** (Einzugsgebiet): nicht gemessen; bei Bedarf als Anhang an das Stück zur Schrittweite.
- **Jacobi und Gauß-Seidel** auf nichtsymmetrischen Systemen.
- **Newton als Fixpunktiteration** und das Newton-Fraktal: stehen in `newton-verfahren-demo` bzw. im Stück zur Schrittweite.
- **Kontraktion in der Wertiteration:** [value-iteration-demo](https://github.com/sebastian-hanisch/value-iteration-demo) hat dazu eine Zeile in seiner Grenzen-Tabelle; hier nur die Banach-Schranke mit A-priori-Fehler.
- **Zufallsstarts:** Die App zieht nichts zufällig; deshalb gibt es keinen Zufalls-Seed-Regler und keinen PDF-Export.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
