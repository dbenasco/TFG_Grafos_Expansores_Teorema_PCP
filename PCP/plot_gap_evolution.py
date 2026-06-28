"""
Genera la Figura 3.6.2: evolucion de unsat(phi) a lo largo del
parametro de potenciacion t DENTRO DE UNA UNICA etapa de amplificacion,
sobre la instancia de referencia compartida (ver reference_instance.py).

POR QUE NO SE ITERA EL BUCLE COMPLETO DE DINUR AQUI:
demo_single_iteration_blowup.py demuestra que una sola pasada real por
degree_reduction -> power_graph(t=1) -> alphabet_reduction ya eleva el
alfabeto residual varios ordenes de magnitud. Una segunda pasada
realimentaria ese valor como W_orig del siguiente power_graph y no es
viable en una maquina domestica (ver 3.6.1). Por eso esta figura mide
el efecto de t en una SOLA etapa de amplificacion sin materializar G^t.

Para ello NO se llama a power_graph(G, t) (coste O(n * d^(2t)), explota
el alfabeto a W^(d^t)). En su lugar se usa
amplifier.calculate_acceptance_probability(G, t, assignment), que
calcula la probabilidad de aceptacion de UNA asignacion fija sobre G^t
mediante potenciacion de matrices (O(N^3 log t), N = nodos de G, no de
G^t). Esto permite recorrer t hasta varios cientos sin coste exponencial,
a cambio de no obtener G^t como objeto (no se puede encadenar una
reduccion de alfabeto real sobre el resultado).
"""
import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
random.seed(0)
np.random.seed(0)

from nice_transformation import degree_reduction, arity_reduction
from amplifier import calculate_acceptance_probability
from reference_instance import reference_clauses, reference_best_assignment, K_PAIRS, C_CLOUD

from tests.test_pcp_complete import trace_arity_assignment, trace_degree_reduction, evaluate_acceptance_probability

T_VALUES = list(range(0, 151, 3))


def run_experiment():
    clauses = reference_clauses()
    best_init = reference_best_assignment()

    G0 = arity_reduction(clauses)
    h0 = trace_arity_assignment(clauses, best_init)

    G1 = degree_reduction(G0, use_expander_cloud=True, c_cloud=C_CLOUD)
    h1 = trace_degree_reduction(G1, h0)

    gap0 = 1.0 - evaluate_acceptance_probability(G1, h1)
    print(f"unsat(phi) antes de amplificar (t=0): {gap0:.4f}")

    gaps = [gap0]
    for t in T_VALUES[1:]:
        acc = calculate_acceptance_probability(G1, t, h1)
        gap = 1.0 - acc
        gaps.append(gap)
        print(f"  t={t:3d} | unsat(phi^t) = {gap:.6f}")

    return T_VALUES, gaps


def plot_results(t_vals, gaps):
    plt.figure(figsize=(10, 6))

    plt.plot(t_vals, gaps, 'o-', color='#e74c3c', linewidth=2, markersize=4,
              label=r'$\mathrm{unsat}(\varphi^t)$ (mejor asignación posible)')
    plt.axhline(1.0, color='black', linestyle='--', linewidth=2, alpha=0.7,
                label='Cota superior ($\\mathrm{unsat}=1$)')

    plt.xlabel(r'Parámetro de potenciación $t$', fontsize=14)
    plt.ylabel(r'$\mathrm{unsat}(\varphi^t)$', fontsize=14)
    plt.title(f'Amplificación de la Brecha de Insatisfacibilidad ($n={K_PAIRS}$)\n'
              r'$\mathrm{unsat}(\varphi^t)$ bajo $G^t$ sin materializar $G^t$ explícitamente', fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_path = "/home/damian/Documentos/TFG Mates/Memoria /Imagenes/Resultados/gap_evolution_dinur.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\nGuardado en: {save_path}")


if __name__ == "__main__":
    t_vals, gaps = run_experiment()
    plot_results(t_vals, gaps)
