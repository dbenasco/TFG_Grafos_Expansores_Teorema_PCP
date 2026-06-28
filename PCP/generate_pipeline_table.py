"""
Genera la Tabla 3.6.3: numero de variables, aristas y tamano del
alfabeto tras cada paso del pipeline, para la instancia de referencia
compartida (ver reference_instance.py) y UNA SOLA iteracion del bucle
de Dinur (t=1). Ver demo_single_iteration_blowup.py para la
justificacion de por que no se itera mas de una vez.

Escribe el cuerpo de una tabla LaTeX (solo el entorno tabular) en
Imagenes/Resultados/pipeline_scale_table.tex, pensada para incluirse
con \\input{} dentro de un entorno table/center en el capitulo.
"""
import sys
import os
import random
import numpy as np
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
random.seed(0)  # reproducibilidad: alphabet_reduction muestrea bits al azar
np.random.seed(0)  # reproducibilidad: la nube expansora usa np.random

from nice_transformation import degree_reduction, arity_reduction
from amplifier import power_graph
from alphabet_reduction import alphabet_reduction
from nice_transformation import _compute_degrees
from reference_instance import reference_clauses, K_PAIRS, C_CLOUD


def fmt(x):
    if x >= 10**6:
        return f"${x:.2e}$".replace("e+0", "e").replace("e+", "e")
    return f"{x:,}".replace(",", "\\,")


def mode_domain(G):
    """Dominio mas frecuente entre las variables de G (no el maximo)."""
    if not G.variables:
        return 0
    return Counter(G.variables.values()).most_common(1)[0][0]


def main():
    clauses = reference_clauses()
    rows = []

    G0 = arity_reduction(clauses)
    rows.append(("3SAT original (\\code{arity\\_reduction})", G0))

    G1 = degree_reduction(G0, use_expander_cloud=True, c_cloud=C_CLOUD)
    rows.append(("\\code{degree\\_reduction} ($d=%d$)" % max(_compute_degrees(G1).values()), G1))

    G2 = power_graph(G1, t=1)
    rows.append(("\\code{power\\_graph} ($t=1$)", G2))

    G3 = alphabet_reduction(G2, num_blr_tests=1, num_edge_tests=2)
    rows.append(("\\code{alphabet\\_reduction}", G3))

    lines = []
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Etapa} & \textbf{Variables ($n$)} & \textbf{Aristas ($m$)} & \textbf{Alf. máx. ($W$)} & \textbf{Alf. típico (moda)} \\")
    lines.append(r"\hline")
    for label, G in rows:
        n = len(G.variables)
        m = G.num_constraints
        w = max(G.variables.values()) if G.variables else 0
        w_mode = mode_domain(G)
        lines.append(f"{label} & {fmt(n)} & {fmt(m)} & {fmt(w)} & {fmt(w_mode)} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    out = "\n".join(lines)
    print(out)

    save_path = "/home/damian/Documentos/TFG Mates/Memoria /Imagenes/Resultados/pipeline_scale_table.tex"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(out + "\n")
    print(f"\nGuardado en: {save_path}")


if __name__ == "__main__":
    main()
