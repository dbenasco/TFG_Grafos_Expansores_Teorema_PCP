# Grafos de expansión, códigos comprobables localmente y el Teorema PCP

Código fuente del Trabajo de Fin de Grado en Matemáticas, Grafos de expansión, códigos comprobables localmente y teorema PCP
## Estructura

```
Expanders/          — Grafos expansores y generadores de figuras (capítulo 2)
  SipserSpielmanExpander.py   — Construcción de códigos expansores de Sipser-Spielman
  expander_quality.py         — Métricas: hueco espectral, expansión de vértices, corrección de errores
  degree_study.py             — Estudio del impacto del grado en grafos expansores
  generate_expander_examples.py — Ejemplos: grafo de Petersen, grafo de Paley
  graph_products_demo.py      — Demostración del producto tensorial y producto zigzag
  spectral_gap_detailed.py    — Análisis detallado del hueco espectral

PCP/                — Pipeline de Dinur y generadores de figuras (capítulos 3 y 4)
  constraint_graph.py         — Estructura de datos central: grafo de restricciones (CSP)
  nice_transformation.py      — Reducción de aridad, reducción de grado, superposición de expansores
  amplifier.py                — Amplificación del hueco: potencia de grafo G^t
  alphabet_reduction.py       — Reducción de alfabeto: codificación Walsh-Hadamard + tests BLR
  pcpifier.py                 — Bucle principal de iteración (pcpify)
  divide_and_conquer.py       — Partición BFS y métricas de corte
  reference_instance.py       — Instancia de referencia concreta para experimentos
  random_local_instance.py    — Generador de fórmulas 3SAT aleatorias locales
  generate_*.py               — Scripts que producen las tablas y figuras de la memoria
  plot_*.py / verify_*.py     — Scripts de visualización y verificación

  tests/                      — Tests unitarios e integración
```

## Instalación

```bash
pip install -r requirements.txt
```

O con conda:

```bash
conda create -n TFGMates python=3.12
conda activate TFGMates
pip install -r requirements.txt
```

## Uso

### Ejecutar el pipeline completo

```bash
cd PCP
python3 pcpifier.py
```

### Ejecutar los tests

```bash
cd PCP
python3 -m unittest tests.test_pcpifier
python3 -m unittest tests.test_amplifier
python3 -m unittest tests.test_alphabet_reduction
```

### Generar las figuras de la memoria

```bash
# Capítulo 2 — Grafos expansores
cd Expanders
python3 generate_expander_examples.py   # paley_expander.png, petersen_expander.png
python3 graph_products_demo.py          # graph_products_demo.png
python3 degree_study.py                 # study_impact_density.png, study_impact_rate.png
python3 spectral_gap_detailed.py        # plot_spectral_gap.png
python3 expander_quality.py             # plot_vertex_expansion.png, plot_error_correction.png,
                                        # plot_probabilistic_verification.png, plot_probabilistic_independence.png

# Capítulos 3 y 4 — Teorema PCP
cd PCP
python3 plot_gap_evolution.py           # gap_evolution_dinur.png
python3 generate_dc_amplification.py   # dc_amplification.png
python3 verify_dc_soundness_random.py  # dc_soundness_random.png
python3 generate_execution_trace.py    # pcpify_trace_diagram.png
```

## Principales referencias

- Arora, S., y Barak, B. (2009). *Computational Complexity: A Modern Approach*. Cambridge University Press.
- Dinur, I. (2007). The PCP theorem by gap amplification. *Journal of the ACM*, 54(3).
- 
