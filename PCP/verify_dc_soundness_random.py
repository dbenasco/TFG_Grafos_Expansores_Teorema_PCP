"""
Verificacion empirica de la Proposicion prop:dc_solidez (Snark.tex)
sobre formulas 3SAT GENUINAMENTE ALEATORIAS (literales y variables
elegidos al azar sobre todo 1..n, sin ninguna estructura fabricada a
mano tipo "K pares" ni ventana local). Se generan muchas instancias, se
particionan con varios s_max, y se conservan SOLO las combinaciones
(instancia, s_max) cuya particion satisface la hipotesis de la
proposicion (unsat(phi)>=eps y delta<=eps/2). Sobre esas, se comprueba
que la conclusion (algun unsat(G_i)>=eps/2) se cumple siempre.

Para que existan instancias con corte pequeno hace falta densidad de
clausulas baja (m << n): el grafo de restricciones de un 3SAT aleatorio
se fragmenta entonces en muchas componentes conexas pequenas en vez de
una sola componente gigante, y bfs_partition (Algoritmo 1, Snark.tex)
las absorbe enteras siembrando un nuevo arbol BFS por cada componente
que cabe en la parte actual, logrando delta pequeno (o cero) sin
ninguna construccion deliberada.
"""
import sys
import os
import random
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nice_transformation import arity_reduction
from divide_and_conquer import bfs_partition, induced_subgraph, cut_ratio
from tests.test_pcp_complete import trace_arity_assignment, evaluate_acceptance_probability

N_INSTANCES = 300


def random_3sat(n, m, seed):
    rng = random.Random(seed)
    clauses = []
    for _ in range(m):
        vars3 = rng.sample(range(1, n + 1), 3)
        signs = [rng.choice([1, -1]) for _ in range(3)]
        clauses.append([s * v for s, v in zip(signs, vars3)])
    return clauses


def main():
    rng = random.Random(0)
    n_pairs_checked = 0   # (instancia, s_max) con particion no trivial
    n_hyp_true = 0
    n_concl_true_given_hyp = 0
    margins = []
    counterexamples = []
    all_points = []  # (delta, eps, hyp_true) para TODAS las combinaciones, no solo las que satisfacen la hipotesis

    for instance_idx in range(N_INSTANCES):
        n = rng.randint(50, 400)
        m = rng.randint(max(5, n // 30), max(6, n // 8))  # densidad baja, varias componentes
        seed = instance_idx

        clauses = random_3sat(n, m, seed)
        G0 = arity_reduction(clauses)
        truth = {i: 1 for i in range(1, n + 1)}
        h0 = trace_arity_assignment(clauses, truth)
        eps = 1 - evaluate_acceptance_probability(G0, h0)
        if eps == 0:
            continue

        for s_max in [5, 10, 20, 40]:
            part = bfs_partition(G0, s_max)
            if len(part) < 2:
                continue
            n_pairs_checked += 1
            delta = cut_ratio(G0, part)
            hyp = delta <= eps / 2
            all_points.append((delta, eps, hyp))
            if not hyp:
                continue
            n_hyp_true += 1

            max_unsat_gi = 0.0
            for V_i in part:
                G_i = induced_subgraph(G0, V_i)
                h_i = {v: h0[v] for v in V_i}
                u_i = 1 - evaluate_acceptance_probability(G_i, h_i)
                max_unsat_gi = max(max_unsat_gi, u_i)

            margin = max_unsat_gi - eps / 2
            margins.append(margin)
            if max_unsat_gi >= eps / 2:
                n_concl_true_given_hyp += 1
            else:
                counterexamples.append((n, m, seed, s_max, eps, delta, max_unsat_gi))

    print(f"Instancias generadas: {N_INSTANCES}")
    print(f"Pares (instancia, s_max) con partición no trivial y eps>0: {n_pairs_checked}")
    print(f"De ellos, hipótesis (delta<=eps/2) satisfecha en: {n_hyp_true}")
    print(f"De esos, conclusión verificada en: {n_concl_true_given_hyp}/{n_hyp_true}")
    if margins:
        print(f"Margen medio: {sum(margins)/len(margins):.4f}  mínimo: {min(margins):.4f}")
    if counterexamples:
        print("CONTRAEJEMPLOS:")
        for c in counterexamples:
            print("  ", c)

    # --- Figura 1: delta vs eps, con la frontera de la hipotesis delta=eps/2 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    xs_yes = [eps for d, eps, hyp in all_points if hyp]
    ys_yes = [d for d, eps, hyp in all_points if hyp]
    xs_no = [eps for d, eps, hyp in all_points if not hyp]
    ys_no = [d for d, eps, hyp in all_points if not hyp]
    ax.scatter(xs_no, ys_no, color='#bdc3c7', s=18, label=f'hipótesis no satisfecha ($n={len(xs_no)}$)', alpha=0.7)
    ax.scatter(xs_yes, ys_yes, color='#27ae60', s=22, label=f'hipótesis satisfecha ($n={len(xs_yes)}$)', alpha=0.85)
    eps_range = [0, max(eps for _, eps, _ in all_points) * 1.05]
    ax.plot(eps_range, [e / 2 for e in eps_range], 'k--', linewidth=1.5, label=r'$\delta=\varepsilon/2$')
    ax.set_xlabel(r'$\varepsilon=\mathrm{unsat}(\varphi)$', fontsize=16)
    ax.set_ylabel(r'$\delta$ (ratio de corte)', fontsize=16)
    ax.set_title('Cada punto es una combinación (instancia, $s_{\\max}$)\nDebajo de la línea: hipótesis satisfecha', fontsize=16)
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.text(-0.08, 1.08, 'A', transform=ax.transAxes, fontsize=18, fontweight='bold')

    # --- Figura 2: densidad (KDE) del margen max unsat(G_i) - eps/2 cuando se satisface ---
    ax2 = axes[1]
    import numpy as np
    from scipy.stats import gaussian_kde
    margins_arr = np.array(margins)
    kde = gaussian_kde(margins_arr)
    xs = np.linspace(margins_arr.min(), margins_arr.max(), 300)
    ax2.fill_between(xs, kde(xs), color='#3498db', alpha=0.4)
    ax2.plot(xs, kde(xs), color='#2c3e50', linewidth=1.8)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='diferencia$=0$')
    ax2.set_xlabel(r'$\max_i\,\mathrm{unsat}(G_i) - \varepsilon/2$', fontsize=16)
    ax2.set_ylabel('Densidad', fontsize=16)
    ax2.set_title(f'Diferencia $\\max_i\\mathrm{{unsat}}(G_i)-\\varepsilon/2$ cuando $\\delta \\le \\varepsilon/2$\n({len(margins)} casos, 0 contraejemplos)', fontsize=16)
    ax2.tick_params(axis='both', labelsize=13)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.text(-0.08, 1.08, 'B', transform=ax2.transAxes, fontsize=18, fontweight='bold')

    fig.tight_layout()
    save_path = "/home/damian/Documentos/TFG Mates/Memoria /Imagenes/Resultados/dc_soundness_random.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"\nGuardado en: {save_path}")


if __name__ == "__main__":
    main()
