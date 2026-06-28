"""
Validacion empirica con significancia estadistica de la Proposicion
prop:cut_bound (Snark.tex, sec:analisis_corte): delta decrece con
s_max para formulas con localidad real. En vez de una unica formula
fabricada a mano, se generan N_SEEDS formulas 3SAT aleatorias con
anchura de banda acotada (random_local_instance.py) y se reporta
media +/- desviacion tipica de delta para cada s_max.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from random_local_instance import random_local_formula
from nice_transformation import arity_reduction
from divide_and_conquer import bfs_partition, cut_ratio

N_VARS = 60
WINDOW = 6
N_CLAUSES = 180
N_SEEDS = 100
S_MAX_VALUES = [10, 15, 20, 25, 30, 40, 50]


def main():
    means, stds = [], []
    for s_max in S_MAX_VALUES:
        deltas = []
        for seed in range(N_SEEDS):
            clauses = random_local_formula(N_VARS, WINDOW, N_CLAUSES, seed=seed)
            G0 = arity_reduction(clauses)
            part = bfs_partition(G0, s_max)
            if len(part) < 2:
                continue
            deltas.append(cut_ratio(G0, part))
        deltas = np.array(deltas)
        means.append(deltas.mean())
        stds.append(deltas.std())
        print(f"s_max={s_max:3d}  delta_mean={deltas.mean():.4f}  delta_std={deltas.std():.4f}  n_seeds={len(deltas)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(S_MAX_VALUES, means, yerr=stds, fmt='o-', color='#3498db',
                ecolor='#2c3e50', capsize=4, linewidth=2, markersize=7,
                label=fr'$\delta$ media $\pm$ desv. típica ({N_SEEDS} semillas)')
    ax.set_xlabel(r'$s_{\max}$', fontsize=12)
    ax.set_ylabel(r'Ratio de corte $\delta$', fontsize=12)
    ax.set_title(f'Ratio de corte vs. tamaño de partición\n'
                 f'Fórmulas 3SAT aleatorias con anchura de banda $w={WINDOW}$ ($n={N_VARS}$, {N_SEEDS} semillas por punto)',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    fig.tight_layout()

    save_path = "/home/damian/Documentos/TFG Mates/Memoria /Imagenes/Resultados/cut_ratio_stats.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"\nGuardado en: {save_path}")


if __name__ == "__main__":
    main()
