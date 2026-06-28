"""
Ultimo experimento de 1.5 (Snark.tex): comprobar que el pipeline REAL de
Dinur (degree_reduction -> power_graph -> alphabet_reduction, ejecutado
paso a paso con sus funciones reales, no el estimador
calculate_acceptance_probability) tambien amplifica la brecha cuando se
aplica a un subgrafo de una particion, igual que sobre la formula
completa en el Capitulo 3 (generate_execution_trace.py).

Se reutiliza la misma instancia 3SAT aleatoria y particion de
generate_dc_gap_distribution.py, eligiendo el subgrafo con mayor
unsat(G_i) inicial. La regularizacion usa degree_reduction con
use_expander_cloud=True (la nube expansora de Sipser-Spielman integrada
en el propio degree_reduction), exactamente como en
generate_execution_trace.py, y NO la funcion add_expander_edges: como ya
se documento en la Seccion de escala, add_expander_edges infla el grado
a d=21 independientemente del tamano del subgrafo de entrada, lo que en
un subgrafo pequeno diluye la brecha mas de lo que un solo paso de
power_graph(t=1) puede compensar, y enmascararia la amplificacion real
que SI se observa con la misma regularizacion que usa el Capitulo 3.
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

from divide_and_conquer import bfs_partition, induced_subgraph
from nice_transformation import arity_reduction, degree_reduction, _compute_degrees
from amplifier import power_graph, get_walks
from alphabet_reduction import alphabet_reduction
from tests.test_pcp_complete import (
    trace_arity_assignment, trace_degree_reduction, build_honest_proof,
    evaluate_acceptance_probability,
)

N_VARS = 200
N_CLAUSES = 16
SEED = 6
S_MAX = 10
T_POWER = 1
C_CLOUD = 3
NUM_BLR_TESTS = 1


def random_3sat(n, m, seed):
    rng = random.Random(seed)
    clauses = []
    for _ in range(m):
        vars3 = rng.sample(range(1, n + 1), 3)
        signs = [rng.choice([1, -1]) for _ in range(3)]
        clauses.append([s * v for s, v in zip(signs, vars3)])
    return clauses


def trace_power_assignment(G_orig, h_orig, t):
    walks = get_walks(G_orig, t)
    h_pow = {}
    for u in G_orig.variables:
        val_list = [h_orig[node] for p in walks[u] for node in p]
        W = G_orig.variables[u]
        idx = 0
        for i, val in enumerate(val_list):
            idx += val * (W ** (len(val_list) - 1 - i))
        h_pow[u] = idx
    return h_pow


def main():
    clauses = random_3sat(N_VARS, N_CLAUSES, SEED)
    G0 = arity_reduction(clauses)
    h0 = trace_arity_assignment(clauses, {i: 1 for i in range(1, N_VARS + 1)})

    partition = bfs_partition(G0, S_MAX)
    best_i, best_unsat = None, -1
    for i, V_i in enumerate(partition):
        G_i = induced_subgraph(G0, V_i)
        h_i = {v: h0[v] for v in V_i}
        u = 1 - evaluate_acceptance_probability(G_i, h_i)
        if u > best_unsat:
            best_unsat, best_i = u, i
    V_i = partition[best_i]
    G_i = induced_subgraph(G0, V_i)
    h_i = {v: h0[v] for v in V_i}
    print(f"Subgrafo elegido: parte {best_i}, |V_i|={len(V_i)}, unsat(G_i) inicial = {best_unsat:.4f}")

    # --- Paso A: degree_reduction (nube expansora, igual que Capitulo 3) ---
    G1 = degree_reduction(G_i, use_expander_cloud=True, c_cloud=C_CLOUD)
    h1 = trace_degree_reduction(G1, h_i)
    d1 = max(_compute_degrees(G1).values())
    unsat1 = 1 - evaluate_acceptance_probability(G1, h1)
    print(f"Tras degree_reduction: n={len(G1.variables)}, d={d1}, unsat={unsat1:.4f}")

    # --- Paso B: power_graph (igual que pcpify, t_power dado) ---
    G2 = power_graph(G1, t=T_POWER)
    h2 = trace_power_assignment(G1, h1, T_POWER)
    unsat2 = 1 - evaluate_acceptance_probability(G2, h2)
    print(f"Tras power_graph (t={T_POWER}): n={len(G2.variables)}, "
          f"max_dom={max(G2.variables.values()):.3g}, unsat={unsat2:.4f}")

    # --- Paso C: alphabet_reduction (igual que pcpify, num_blr_tests dado) ---
    G3 = alphabet_reduction(G2, num_blr_tests=NUM_BLR_TESTS)
    k = math.ceil(math.log2(max(G2.variables.values())))
    h3 = build_honest_proof(G3, G2, h2, k)
    unsat3 = 1 - evaluate_acceptance_probability(G3, h3)
    print(f"Tras alphabet_reduction: n={len(G3.variables):,}, k={k} bits, unsat={unsat3:.4f}")

    print("\nResumen unsat(G_i) por etapa:")
    print(f"  inicial={best_unsat:.4f}  degree_reduction={unsat1:.4f}  "
          f"power_graph={unsat2:.4f}  alphabet_reduction={unsat3:.4f}")

    stages = [
        ("$G_i$\n(inicial)", best_unsat),
        ("degree_reduction", unsat1),
        ("power_graph", unsat2),
        ("alphabet_reduction", unsat3),
    ]
    draw_diagram(stages, len(V_i))


def draw_diagram(stages, n_initial):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.95)
    ax.axis('off')

    n_stages = len(stages)
    box_w, box_h = 0.16, 0.30
    box_y = 0.72
    xs = [0.5 / n_stages + i / n_stages for i in range(n_stages)]

    for i, (x, (name, unsat)) in enumerate(zip(xs, stages)):
        box = mpatches.FancyBboxPatch(
            (x - box_w / 2, box_y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.02", linewidth=1.5,
            edgecolor='#2c3e50', facecolor='white'
        )
        ax.add_patch(box)
        ax.text(x, box_y, name, ha='center', va='center', fontsize=10, fontweight='bold')
        if i < n_stages - 1:
            ax.annotate('', xy=(xs[i + 1] - box_w / 2 - 0.005, box_y),
                         xytext=(x + box_w / 2 + 0.005, box_y),
                         arrowprops=dict(arrowstyle='-|>', color='#2c3e50', lw=1.5))

    unsat_y0, unsat_y1 = 0.05, 0.32
    unsats = [u for _, u in stages]
    lo, hi = min(unsats + [0.5]), max(unsats + [0.5])
    norm = [unsat_y0 + (u - lo) / (hi - lo + 1e-9) * (unsat_y1 - unsat_y0) for u in unsats]
    half_y = unsat_y0 + (0.5 - lo) / (hi - lo + 1e-9) * (unsat_y1 - unsat_y0)

    ax.axhline(half_y, color='gray', linestyle=':', linewidth=1.5, xmin=0.03, xmax=0.97)
    ax.text(0.005, half_y, r'$\frac{1}{2}$', ha='right', va='center', fontsize=10, color='gray')

    ax.plot(xs, norm, '-', color='#7f8c8d', linewidth=2, zorder=1, clip_on=False)
    colors = ['#e74c3c' if u >= 0.5 else '#3498db' for u in unsats]
    ax.scatter(xs, norm, c=colors, s=70, zorder=2, edgecolor='#2c3e50', linewidth=1, clip_on=False)
    for x, y, u in zip(xs, norm, unsats):
        ax.text(x, y - 0.055, f"unsat={u:.3f}", ha='center', va='top', fontsize=9,
                 color='#c0392b' if u >= 0.5 else '#2c3e50')
    ax.text(0.5, unsat_y1 + 0.08,
             r'Evolución de $\mathrm{unsat}(G_i)$: rojo si $\geq\frac{1}{2}$, azul si $<\frac{1}{2}$',
             ha='center', va='bottom', fontsize=10, style='italic', color='#2c3e50')

    fig.suptitle(f'Traza del pipeline pcpify() real sobre un subgrafo de una partición ($|V_i|={n_initial}$)',
                 fontsize=13, y=0.99)

    save_path = "/home/damian/Documentos/TFG Mates/Memoria /Imagenes/Resultados/dc_amplification.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nGuardado en: {save_path}")


if __name__ == "__main__":
    main()
