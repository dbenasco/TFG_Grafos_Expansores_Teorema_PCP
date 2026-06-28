"""
Generador de formulas 3SAT aleatorias con anchura de banda acotada
(ver Definicion def:bandwidth, Snark.tex sec:analisis_corte), para
validar con significancia estadistica (muchas semillas) las
afirmaciones empiricas de las secciones 1.3 y 1.5: que el particionado
por localidad solo funciona cuando la formula tiene localidad real, y
que el pico de memoria de D&C se mantiene acotado independientemente de
n.

Cada clausula elige sus 3 literales dentro de una ventana aleatoria de
anchura `window` variables consecutivas, con signos aleatorios. Esto
genera una formula con anchura de banda <= window respecto al orden
natural 1..n, pero por lo demas aleatoria (posicion de la ventana,
signos, variables exactas dentro de ella).
"""
import random


def random_local_formula(n, window, m, seed=None):
    """
    Parameters
    ----------
    n : int
        Numero de variables (1..n).
    window : int
        Anchura de la ventana de la que se muestrean los literales de
        cada clausula (anchura de banda resultante <= window).
    m : int
        Numero de clausulas.
    seed : int or None

    Returns
    -------
    list of list of int
        Clausulas en formato DIMACS.
    """
    rng = random.Random(seed)
    clauses = []
    for _ in range(m):
        center = rng.randint(1, n)
        lo = max(1, center - window // 2)
        hi = min(n, center + window // 2)
        if hi - lo < 2:
            lo, hi = max(1, n - window), n
        lits = []
        for _ in range(3):
            var = rng.randint(lo, hi)
            sign = rng.choice([1, -1])
            lits.append(sign * var)
        clauses.append(lits)
    return clauses


def best_assignment_unsat(G, assignment_candidates):
    """unsat(phi) bajo la mejor de varias asignaciones candidatas."""
    from tests.test_pcp_complete import evaluate_acceptance_probability
    best = 0.0
    for assign in assignment_candidates:
        unsat = 1 - evaluate_acceptance_probability(G, assign)
        best = min(best, unsat) if best else unsat
        best = min(best, unsat)
    return best
