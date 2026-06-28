"""
Fixed Expander Graph Examples
==============================
Draws two well-known *fixed* (non-random) explicit expander graphs, to
illustrate the informal definition of expander given in the introduction
of the Expander Graphs chapter. Both titles report lambda(G), the
algebraic expansion parameter (max |eigenvalue| of the adjacency matrix
excluding the top one), computed directly from the graph's spectrum:

  - Petersen graph: 10 vertices, 3-regular. Eigenvalues {3, 1^5, -2^4},
    so lambda(G) = 2, well below d = 3.
  - Paley graph of order 13: 13 vertices, 6-regular, built from the
    quadratic residues mod 13. Non-bipartite explicit construction with
    eigenvalues close to the Ramanujan bound 2*sqrt(d-1).

(A hypercube Q_k was considered but rejected: it is bipartite, which
forces an eigenvalue of exactly -d, so lambda(G) = d and it fails the
algebraic expander definition for any epsilon > 0.)

Usage:
    conda activate TFGMates
    python generate_expander_examples.py
"""

import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def lambda_of(G):
    """max |eigenvalue| of the adjacency matrix, excluding the largest (=d)."""
    eigs = np.linalg.eigvalsh(nx.to_numpy_array(G))
    eigs_sorted = sorted(eigs, key=lambda x: -x)
    return max(abs(e) for e in eigs_sorted[1:])


def paley_graph(q):
    """Paley graph of order q (q prime, q = 1 mod 4): vertices Z_q, edges
    between x, y whose difference is a nonzero quadratic residue mod q."""
    residues = {(x * x) % q for x in range(1, q)}
    G = nx.Graph()
    G.add_nodes_from(range(q))
    for x in range(q):
        for y in range(x + 1, q):
            if (y - x) % q in residues:
                G.add_edge(x, y)
    return G


def draw_expander(G, pos, title, out_path, node_color="#4C72B0"):
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color, node_size=450)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_color="white")
    ax.set_title(title, fontsize=13, wrap=True)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
    os.makedirs(out_dir, exist_ok=True)

    petersen = nx.petersen_graph()
    pos_petersen = nx.shell_layout(petersen, nlist=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    lam_petersen = lambda_of(petersen)
    print(f"Petersen: d=3, lambda(G)={lam_petersen}")
    draw_expander(
        petersen, pos_petersen,
        rf"Grafo de Petersen ($3$-regular, $n=10$, $\lambda(G)={lam_petersen:.0f}$)",
        os.path.join(out_dir, "petersen_expander.png"),
    )

    paley = paley_graph(13)
    pos_paley = nx.circular_layout(paley)
    lam_paley = lambda_of(paley)
    print(f"Paley(13): d=6, lambda(G)={lam_paley}")
    draw_expander(
        paley, pos_paley,
        rf"Grafo de Paley de orden $13$ ($6$-regular, $\lambda(G)\approx{lam_paley:.2f}$)",
        os.path.join(out_dir, "paley_expander.png"),
        node_color="#DD8452",
    )
