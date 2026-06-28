"""
Genera la Tabla de 1.5.2 (Snark.tex, sec:dc_escala): compara el pico de
memoria de pcpify() directo sobre la formula completa frente al pico de
memoria de divide_and_conquer_pcpify con s_max fijo, en funcion del
tamano total n de la formula.

Honestidad del resultado (ver discusion en el chat antes de escribir
esto): el particionado NO evita la explosion dirigida por t o por
repetir el bucle exterior -- esas dependen de constantes (d, t, k) que
son las mismas en un trozo pequeno que en el grafo completo, exactamente
la misma leccion del Capitulo 3 ("constantes, no tamano de instancia").
Lo que el particionado sí ofrece es acotar el PICO de memoria de cada
llamada a O(s_max) en vez de O(n): el coste total summed across
particiones sigue siendo lineal en n (igual que pcpify directo), pero
cada llamada individual usa memoria acotada, lo que permite procesar
formulas demasiado grandes para que pcpify() directo quepa en RAM de una
sola vez.

Cada llamada a pcpify() se ejecuta en un subproceso aislado (via
/usr/bin/time -v) para medir su pico de RSS real, en vez de depender de
resource.getrusage en el mismo proceso (que es un maximo histo'rico que
no baja aunque Python libere memoria entre llamadas).

Para significancia estadistica se repite con N_SEEDS formulas 3SAT
aleatorias distintas por cada n (random_local_instance.py, no la cadena
de implicaciones fabricada a mano), reportando media y desviacion
tipica del pico de memoria.
"""
import sys
import os
import subprocess
import re
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from divide_and_conquer import bfs_partition
from nice_transformation import arity_reduction
from random_local_instance import random_local_formula

S_MAX = 10
N_VALUES = [10, 20, 30]
N_SEEDS = 3
WINDOW = 6
C_EXP = 1
T_POWER = 1
NUM_BLR_TESTS = 1

# ConstraintGraph stores closures (check_fn) that cannot be pickled across
# a process boundary. Instead of serializing the graph object itself, the
# subprocess rebuilds it from the original clauses (always picklable) and,
# for D&C calls, restricts it to a given set of variable indices.
_WORKER_TEMPLATE = """
import sys, pickle, base64
sys.path.append({path!r})
from nice_transformation import arity_reduction
from pcpifier import pcpify

clauses = pickle.loads(base64.b64decode({clauses_b64!r}))
G0 = arity_reduction(clauses)

restrict_b64 = {restrict_b64!r}
if restrict_b64 is not None:
    V_i = pickle.loads(base64.b64decode(restrict_b64))
    from constraint_graph import ConstraintGraph
    H = ConstraintGraph()
    for v in V_i:
        H.add_variable(v, G0.variables[v])
    for u, v, check_fn in G0.constraints:
        if u in V_i and v in V_i:
            H.add_constraint(u, v, check_fn)
    G0 = H

G_final = pcpify(G0, iterations=1, t_power={t_power}, c_exp={c_exp}, num_blr_tests={num_blr_tests})
print("N_FINAL", len(G_final.variables))
"""


def _run_pcpify_subprocess(clauses, restrict_to=None, t_power=T_POWER, c_exp=C_EXP,
                            num_blr_tests=NUM_BLR_TESTS, timeout=120):
    import pickle, base64
    here = os.path.dirname(os.path.abspath(__file__))
    clauses_b64 = base64.b64encode(pickle.dumps(clauses)).decode()
    restrict_b64 = base64.b64encode(pickle.dumps(restrict_to)).decode() if restrict_to is not None else None
    script = _WORKER_TEMPLATE.format(
        path=here, clauses_b64=clauses_b64, restrict_b64=restrict_b64,
        t_power=t_power, c_exp=c_exp, num_blr_tests=num_blr_tests,
    )
    proc = subprocess.run(
        ["/usr/bin/time", "-v", "conda", "run", "-n", "TFGMates", "python3", "-c", script],
        capture_output=True, text=True, timeout=timeout,
    )
    stderr = proc.stderr
    m_mem = re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr)
    m_n = re.search(r"N_FINAL (\d+)", proc.stdout)
    peak_kb = int(m_mem.group(1)) if m_mem else None
    n_final = int(m_n.group(1)) if m_n else None
    if peak_kb is None or n_final is None:
        print("WORKER FAILED, stderr tail:\\n", stderr[-2000:])
    return peak_kb, n_final


def main():
    rows = []
    for n in N_VALUES:
        direct_peaks, dc_peaks, ks = [], [], []
        print(f"--- n={n} ---")
        for seed in range(N_SEEDS):
            clauses = random_local_formula(n, WINDOW, n, seed=seed)
            G0 = arity_reduction(clauses)

            peak_direct, nf_direct = _run_pcpify_subprocess(clauses)

            partition = bfs_partition(G0, S_MAX)
            peaks_i = []
            for V_i in partition:
                peak_i, nf_i = _run_pcpify_subprocess(clauses, restrict_to=V_i)
                peaks_i.append(peak_i)
            peak_dc = max(peaks_i)

            direct_peaks.append(peak_direct)
            dc_peaks.append(peak_dc)
            ks.append(len(partition))
            print(f"  seed={seed}: directo={peak_direct/1024:.0f} MB  "
                  f"D&C(k={len(partition)})={peak_dc/1024:.0f} MB")

        direct_peaks = np.array(direct_peaks) / 1024.0
        dc_peaks = np.array(dc_peaks) / 1024.0
        print(f"  => directo: {direct_peaks.mean():.0f} +/- {direct_peaks.std():.0f} MB | "
              f"D&C: {dc_peaks.mean():.0f} +/- {dc_peaks.std():.0f} MB")
        rows.append((n, np.mean(ks), direct_peaks.mean(), direct_peaks.std(), dc_peaks.mean(), dc_peaks.std()))

    lines = [
        r"\begin{tabular}{rrrr}",
        r"\hline",
        r"\textbf{$n$} & \textbf{$k$ medio} & \textbf{Pico directo (MB)} & \textbf{Pico D\&C (MB)} \\",
        r"\hline",
    ]
    for n, k_mean, d_mean, d_std, dc_mean, dc_std in rows:
        lines.append(f"{n} & {k_mean:.1f} & ${d_mean:.0f} \\pm {d_std:.0f}$ & ${dc_mean:.0f} \\pm {dc_std:.0f}$ \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    out = "\n".join(lines)
    print("\n" + out)

    save_path = "/home/damian/Documentos/TFG Mates/Memoria /Imagenes/Resultados/dc_scale_table.tex"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(out + "\n")
    print(f"\nGuardado en: {save_path}")


if __name__ == "__main__":
    main()
