"""
Implementacion de 1.3-1.4: particionado del grafo de restricciones (BFS) y
aplicacion autonoma del pipeline original (pcpify) sobre cada subgrafo.

Ver Capitulos/Snark.tex, secciones "Particionado del grafo de
restricciones" y "Aplicacion del pipeline original a subproblemas" para
las definiciones formales (k-particion, aristas de corte, anchura de
banda) y las demostraciones de completitud/solidez que este modulo
implementa.
"""
import sys
import os
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from constraint_graph import ConstraintGraph
from nice_transformation import arity_reduction, _compute_degrees
from pcpifier import pcpify


def _build_adjacency(G):
    adj = {v: [] for v in G.variables}
    for u, v, _ in G.constraints:
        if u == v:
            continue
        adj[u].append(v)
        adj[v].append(u)
    return adj


def bfs_partition(G, s_max):
    """
    Algoritmo 1 (Snark.tex, sec:particion_bfs): particion heuristica por
    localidad. Semilla = nodo de grado maximo entre los no asignados;
    crece por BFS hasta s_max nodos; repite hasta agotar V.

    Parameters
    ----------
    G : ConstraintGraph
    s_max : int
        Tamano maximo de cada parte.

    Returns
    -------
    list of set
        Particion {V_1, ..., V_k} de G.variables, |V_i| <= s_max.
    """
    if s_max < 1:
        raise ValueError("s_max debe ser >= 1.")

    adj = _build_adjacency(G)
    degree = _compute_degrees(G)
    # Orden estable (insercion en G.variables) para desempatar grados
    # iguales de forma consistente, en vez de el orden arbitrario de un
    # set de Python. Para formulas con localidad esto aproxima mejor los
    # bloques contiguos de la Proposicion prop:cut_bound.
    order = {v: i for i, v in enumerate(G.variables)}
    unassigned = set(G.variables)
    parts = []

    while unassigned:
        part = set()
        # Si la componente conexa de una semilla se agota antes de llegar
        # a s_max (p.ej. porque el grafo esta formado por muchas
        # componentes pequenas e inconexas, como K pares contradictorios
        # independientes), en vez de rellenar con nodos sueltos de otra
        # componente cualquiera -- lo que cortaria esa otra componente
        # por la mitad sin necesidad -- se siembra un NUEVO arbol BFS
        # desde la siguiente semilla de mayor grado, absorbiendo otra
        # componente entera (o tanto como quepa) en la misma parte. Esto
        # nunca corta una componente que cabe entera en una parte.
        while len(part) < s_max and unassigned:
            seed = max(unassigned, key=lambda v: (degree.get(v, 0), -order[v]))
            queue = deque([seed])
            visited = {seed}
            while queue and len(part) < s_max:
                node = queue.popleft()
                if node not in unassigned:
                    continue
                part.add(node)
                unassigned.discard(node)
                for nbr in adj[node]:
                    if nbr in unassigned and nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)

        parts.append(part)

    return parts


def induced_subgraph(G, V_i):
    """
    Subgrafo inducido G[V_i] (Definicion def:kparticion de Snark.tex):
    conserva las variables de V_i y solo las aristas con ambos extremos
    en V_i. Las aristas de corte se descartan silenciosamente.
    """
    H = ConstraintGraph()
    for v in V_i:
        H.add_variable(v, G.variables[v])
    for u, v, check_fn in G.constraints:
        if u in V_i and v in V_i:
            H.add_constraint(u, v, check_fn)
    return H


def cut_ratio(G, partition):
    """delta = |E_cut| / |E| para una particion dada."""
    part_of = {}
    for i, V_i in enumerate(partition):
        for v in V_i:
            part_of[v] = i
    total = len(G.constraints)
    if total == 0:
        return 0.0
    cut = sum(1 for u, v, _ in G.constraints if part_of[u] != part_of[v])
    return cut / total


def divide_and_conquer_pcpify(clauses, s_max, iterations=1, t_power=1, **pcpify_kwargs):
    """
    Algoritmo 2 (Snark.tex, sec:dc_implementacion): aplica arity_reduction
    una vez, particiona por localidad, y ejecuta pcpify() de forma
    autonoma sobre cada subgrafo inducido (sin compartir ningun estado
    entre ellos: cada G_i pasa por su propia degree_reduction,
    add_expander_edges y bucle de amplificacion).

    Returns
    -------
    list of ConstraintGraph
        Una instancia booleana G_i' por cada parte de la particion.
    list of set
        La particion usada (para poder reportar |V_i|, delta, etc.)
    """
    G0 = arity_reduction(clauses)
    partition = bfs_partition(G0, s_max)

    results = []
    for V_i in partition:
        G_i = induced_subgraph(G0, V_i)
        G_i_prime = pcpify(G_i, iterations=iterations, t_power=t_power, **pcpify_kwargs)
        results.append(G_i_prime)

    return results, partition
