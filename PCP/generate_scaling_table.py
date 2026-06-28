"""
Genera la Tabla de escalado de 3.6.1: para varios tamanos K de la
instancia de referencia (ver reference_instance.py), ejecuta UNA SOLA
pasada del pipeline y reporta n tras cada etapa, confirmando
empiricamente que escalar K (y por tanto n) es lineal y barato,
mientras que el grado regularizado d se mantiene constante.

Esto respalda directamente la afirmacion de 3.6.1 de que la explosion
de recursos depende de los parametros internos de una etapa (t, d, k),
no del tamano K de la formula. Sin esta tabla, esa afirmacion no tenia
ninguna evidencia citable en el documento.
"""
import sys
import os
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nice_transformation import degree_reduction, arity_reduction, _compute_degrees
from amplifier import power_graph
from alphabet_reduction import alphabet_reduction
from reference_instance import C_CLOUD

K_VALUES = [10, 30, 100, 300]


def reference_clauses(k):
    return [[i, i, i] for i in range(1, k + 1)] + [[-i, -i, -i] for i in range(1, k + 1)]


def fmt(x):
    return f"{x:,}".replace(",", "\\,")


def run_once(k, seed):
    random.seed(seed)
    np.random.seed(seed)
    clauses = reference_clauses(k)
    G0 = arity_reduction(clauses)
    G1 = degree_reduction(G0, use_expander_cloud=True, c_cloud=C_CLOUD)
    d1 = max(_compute_degrees(G1).values())
    G2 = power_graph(G1, t=1)
    G3 = alphabet_reduction(G2, num_blr_tests=1, num_edge_tests=2)
    return len(G0.variables), d1, len(G3.variables)


def main():
    rows = []
    for k in K_VALUES:
        n0, d1, nf = run_once(k, seed=0)
        rows.append((k, n0, d1, nf))
        print(f"K={k:4d}  n0={n0:5d}  d={d1}  n_final={nf:,}  n_final/K={nf/k:.1f}")

    # Control de ruido: con K fijo (=100), solo variando la semilla aleatoria,
    # cuanto varia n_final por azar. Esto separa "ruido de muestreo" de un
    # posible efecto super-lineal real en K.
    print("\nControl de ruido (K=100 fijo, variando solo la semilla):")
    noise_vals = []
    for seed in [1, 2, 3, 4]:
        _, _, nf_seed = run_once(100, seed=seed)
        noise_vals.append(nf_seed)
        print(f"  seed={seed}  n_final={nf_seed:,}")
    spread = (max(noise_vals) - min(noise_vals)) / min(noise_vals) * 100
    print(f"  Dispersión relativa por ruido de muestreo a K fijo: {spread:.1f}%")

    lines = [
        r"\begin{tabular}{rrrrr}",
        r"\hline",
        r"\textbf{$K$ (pares)} & \textbf{$n_0$ (tras arity\_reduction)} & \textbf{$d$ regularizado} & \textbf{$n$ final (tras una iteración)} & \textbf{$n_{\mathrm{final}}/K$} \\",
        r"\hline",
    ]
    for k, n0, d, nf in rows:
        lines.append(f"{k} & {fmt(n0)} & {d} & {fmt(nf)} & {nf/k:.1f} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    out = "\n".join(lines)
    print("\n" + out)

    save_path = "/home/damian/Documentos/TFG Mates/Memoria /Imagenes/Resultados/scaling_table.tex"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(out + "\n")
    print(f"\nGuardado en: {save_path}")


if __name__ == "__main__":
    main()
