"""
Genera el diagrama de 3.6.4: una ejecucion completa y anotada de una
iteracion del pipeline sobre la instancia de referencia compartida (ver
reference_instance.py), representada como un diagrama de flujo en vez
de una traza de texto/consola.

Reutiliza los mismos pasos que demo_single_iteration_blowup.py y
generate_pipeline_table.py, pero acumula las metricas de cada etapa
(n, m, W, unsat) para dibujarlas como cajas conectadas por flechas, con
una mini-curva de evolucion de unsat(phi) superpuesta debajo.
"""
import sys
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
random.seed(0)
np.random.seed(0)

from nice_transformation import degree_reduction, arity_reduction, _compute_degrees
from amplifier import power_graph
from alphabet_reduction import alphabet_reduction
from reference_instance import reference_clauses, reference_best_assignment, K_PAIRS, C_CLOUD
from tests.test_pcp_complete import (
    trace_arity_assignment, trace_degree_reduction, build_honest_proof,
    evaluate_acceptance_probability,
)

lines = []
stages = []  # list of dicts: name, n, m, w, unsat


def log(msg=""):
    print(msg)
    lines.append(msg)


def draw_diagram():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.95)
    ax.axis('off')

    n_stages = len(stages)
    box_w, box_h = 0.18, 0.30
    box_y = 0.72
    xs = [0.5 / n_stages + i / n_stages for i in range(n_stages)]

    colors = ['white', 'white', 'white', 'white']

    for i, (x, s) in enumerate(zip(xs, stages)):
        box = mpatches.FancyBboxPatch(
            (x - box_w / 2, box_y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.02", linewidth=1.5,
            edgecolor='#2c3e50', facecolor=colors[i % len(colors)]
        )
        ax.add_patch(box)
        ax.text(x, box_y + 0.10, s['name'], ha='center', va='center',
                 fontsize=14, fontweight='bold')
        stats_txt = f"$n$={s['n']:,}\n$m$={s['m']:,}\n$W$={s['w']}"
        ax.text(x, box_y - 0.04, stats_txt, ha='center', va='center', fontsize=12)

        if i < n_stages - 1:
            ax.annotate('', xy=(xs[i + 1] - box_w / 2 - 0.005, box_y),
                         xytext=(x + box_w / 2 + 0.005, box_y),
                         arrowprops=dict(arrowstyle='-|>', color='#2c3e50', lw=1.5))

    # Mini-curva de evolucion de unsat(phi) bajo el diagrama
    unsat_y0, unsat_y1 = 0.05, 0.32
    unsats = [s['unsat'] for s in stages]
    norm = [unsat_y0 + (u - min(unsats)) / (max(unsats) - min(unsats) + 1e-9) * (unsat_y1 - unsat_y0)
            for u in unsats]
    ax.plot(xs, norm, 'o-', color='#e74c3c', linewidth=2, markersize=7, clip_on=False)
    for x, y, u in zip(xs, norm, unsats):
        ax.text(x, y - 0.06, f"unsat={u:.3f}", ha='center', va='top', fontsize=12, color='#c0392b')
    ax.text(0.5, unsat_y1 + 0.08, r'Evolución de $\mathrm{unsat}(\varphi)$ a través del pipeline',
             ha='center', va='bottom', fontsize=13, style='italic', color='#c0392b')

    fig.suptitle(f'Traza de una iteración del pipeline ($K={K_PAIRS}$ pares contradictorios)', fontsize=17, y=0.99)

    save_path = "/home/damian/Documentos/TFG Mates/Memoria /Imagenes/Resultados/pcpify_trace_diagram.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nDiagrama guardado en: {save_path}")


def fmt_w(w):
    return f"{w:,}" if w < 10**6 else f"{w:.2e}"


def main():
    clauses = reference_clauses()
    log("=== Traza de ejecución: pipeline de Dinur sobre la instancia de referencia ===")
    log(f"Fórmula: {K_PAIRS} pares independientes (x_i) AND (NOT x_i), i=1..{K_PAIRS}")
    log("  Clausula 2i-2: (x_i v x_i v x_i)    -- equivale a 'x_i'")
    log("  Clausula 2i-1: (-x_i v -x_i v -x_i) -- equivale a 'NOT x_i'")
    log("  Insatisfactible: cada par fuerza a violar exactamente una de sus dos clausulas.")
    log("")

    log("--- Paso 1: arity_reduction (3SAT -> 2CSP) ---")
    G0 = arity_reduction(clauses)
    log(f"  Variables originales: x_1..x_{K_PAIRS} (dominio 2)")
    log(f"  Variables auxiliares: w_0..w_{2*K_PAIRS - 1} (dominio 7 cada una)")
    log(f"  |V| = {len(G0.variables)}, |E| = {G0.num_constraints}")

    best_init = reference_best_assignment()
    h0 = trace_arity_assignment(clauses, best_init)
    val0 = evaluate_acceptance_probability(G0, h0)
    log(f"  Mejor asignación posible (todo x_i=1): val(phi) = {val0:.4f}  =>  unsat(phi) = {1 - val0:.4f}")
    log("")
    stages.append({"name": "arity_reduction", "n": len(G0.variables), "m": G0.num_constraints,
                    "w": fmt_w(max(G0.variables.values())), "unsat": 1 - val0})

    log("--- Paso 2: degree_reduction (irregular -> d-regular por nube expansora) ---")
    G1 = degree_reduction(G0, use_expander_cloud=True, c_cloud=C_CLOUD)
    d1 = max(_compute_degrees(G1).values())
    h1 = trace_degree_reduction(G1, h0)
    val1 = evaluate_acceptance_probability(G1, h1)
    log(f"  Cada variable se reemplaza por tantas copias como su grado original,")
    log(f"  conectadas internamente por una nube expansora de Sipser-Spielman (c={C_CLOUD}).")
    log(f"  Grado regularizado: d = {d1}")
    log(f"  |V| = {len(G1.variables)}, |E| = {G1.num_constraints}")
    log(f"  unsat(phi) tras regularizar = {1 - val1:.4f}  (se diluye por las aristas de relleno/ciclo)")
    log("")
    stages.append({"name": "degree_reduction", "n": len(G1.variables), "m": G1.num_constraints,
                    "w": fmt_w(max(G1.variables.values())), "unsat": 1 - val1})

    log("--- Paso 3: power_graph (amplificación de brecha, t=1) ---")
    G2 = power_graph(G1, t=1)
    from amplifier import get_walks
    walks = get_walks(G1, 1)
    W_orig = max(G1.variables.values())
    h2 = {}
    for u in G1.variables:
        val_list = [h1[node] for p in walks[u] for node in p]
        idx = 0
        for i, val in enumerate(val_list):
            idx += val * (W_orig ** (len(val_list) - 1 - i))
        h2[u] = idx
    val2 = evaluate_acceptance_probability(G2, h2)
    log(f"  Cada nodo pasa a representar una 'nube de opiniones' sobre sus caminos de longitud t=1.")
    log(f"  |V| = {len(G2.variables):,} (se conserva), |E| = {G2.num_constraints:,} (crece a n*d^(2t))")
    log(f"  Alfabeto máximo: W = {max(G2.variables.values()):,} (crece a W_orig^(d^t * (t+1)))")
    log(f"  unsat(phi) amplificado = {1 - val2:.4f}  (la brecha crece, como predice el Lema de Amplificación)")
    log("")
    stages.append({"name": "power_graph(t=1)", "n": len(G2.variables), "m": G2.num_constraints,
                    "w": fmt_w(max(G2.variables.values())), "unsat": 1 - val2})

    log("--- Paso 4: alphabet_reduction (composición PCP, vuelta a alfabeto booleano) ---")
    G3 = alphabet_reduction(G2, num_blr_tests=1, num_edge_tests=2)
    k = math.ceil(math.log2(max(G2.variables.values())))
    h3 = build_honest_proof(G3, G2, h2, k)
    val3 = evaluate_acceptance_probability(G3, h3)
    log(f"  k = ceil(log2(W)) = {k} bits por variable de G^t")
    log(f"  |V| = {len(G3.variables):,} (variables WH/PI/W_BLR/W_EVAL instanciadas perezosamente)")
    log(f"  |E| = {G3.num_constraints:,}")
    log(f"  Alfabeto máximo residual: {max(G3.variables.values()):.3g} (variable W_EVAL, dominio W_u*W_v)")
    log(f"  unsat(phi) final tras una iteración = {1 - val3:.4f}")
    log("")
    stages.append({"name": "alphabet_reduction", "n": len(G3.variables), "m": G3.num_constraints,
                    "w": fmt_w(max(G3.variables.values())), "unsat": 1 - val3})

    log("=== Fin de la traza (una sola iteración) ===")
    log("Nota: el alfabeto residual de W_EVAL retroalimentaría una segunda")
    log("iteración como W_orig del siguiente power_graph, lo que dispara k a")
    log("centenas de bits y el número de variables muy por encima de lo que")
    log("soporta una máquina doméstica (ver 3.6.1 y demo_single_iteration_blowup.py).")

    save_path = "/home/damian/Documentos/TFG Mates/Memoria /Imagenes/Resultados/pcpify_trace.txt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nTraza de texto (referencia) guardada en: {save_path}")

    draw_diagram()


if __name__ == "__main__":
    main()
