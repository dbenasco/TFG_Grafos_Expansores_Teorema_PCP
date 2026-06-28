"""
Graph Products Demo
===================
Builds two tiny example graphs G (big graph) and H (small graph) and
draws G, H, their replacement product G {r} H and their zig-zag
product G {z} H side by side, to illustrate the "cloud" construction
used in the degree-reduction step of Dinur's PCP proof.

G = K4 (4 vertices, 3-regular)   -> the "big" graph whose size we keep
H = C3 (3 vertices, 2-regular)   -> the "small" graph that fixes the degree

Each vertex of G is replaced by a cloud of |V(H)|=3 nodes; nodes in the
same cloud are drawn close together and share a colour.

For visual clarity the replacement product is drawn with a single
external edge per port (degree d+1) instead of the two duplicated
parallel edges of the formal (2d-regular) definition in the thesis;
duplicating a parallel edge does not change what is visible in a
simple line drawing.

Usage:
    conda activate TFGMates
    python graph_products_demo.py
"""

import os
import networkx as nx
import matplotlib.pyplot as plt


def build_port_labeling(G):
    """Assigns to every vertex u a bijection {0,...,D-1} -> neighbours of u.
    Returns port[u][i] = v and port_index[u][v] = i."""
    port = {u: {} for u in G.nodes()}
    port_index = {u: {} for u in G.nodes()}
    for u in G.nodes():
        for i, v in enumerate(sorted(G.neighbors(u))):
            port[u][i] = v
            port_index[u][v] = i
    return port, port_index


def replacement_product(G, H, port, port_index):
    """G (r) H, single external edge per port (degree d+1)."""
    R = nx.Graph()
    for u in G.nodes():
        for i in H.nodes():
            R.add_node((u, i))
    # internal edges (copies of H inside each cloud)
    for u in G.nodes():
        for i, j in H.edges():
            R.add_edge((u, i), (u, j))
    # external edges (one per G-edge, connecting matching ports)
    for u, v in G.edges():
        i = port_index[u][v]
        j = port_index[v][u]
        R.add_edge((u, i), (v, j))
    return R


def zigzag_product(G, H, port, port_index):
    """G (z) H: edge (u,i)-(v,j) iff exists k with i~k in H, v=port[u][k],
    j0 = port_index[v][u], and j0~j in H."""
    Z = nx.Graph()
    for u in G.nodes():
        for i in H.nodes():
            Z.add_node((u, i))
    for u in G.nodes():
        for i in H.nodes():
            for k in H.neighbors(i):
                v = port[u][k]
                j0 = port_index[v][u]
                for j in H.neighbors(j0):
                    Z.add_edge((u, i), (v, j))
    return Z


def cloud_layout(G, H, pos_G, spread=0.35):
    """Places each cloud (u, *) around pos_G[u]."""
    pos_H = nx.circular_layout(H, scale=spread)
    pos = {}
    for u in G.nodes():
        for i in H.nodes():
            ox, oy = pos_H[i]
            ux, uy = pos_G[u]
            pos[(u, i)] = (ux + ox, uy + oy)
    return pos


def draw_graph(ax, G, pos, title, node_color="#4C72B0"):
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color, node_size=350)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=11, font_color="white")
    ax.set_title(title, fontsize=16)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_product(ax, P, G, H, pos_G, title):
    pos = cloud_layout(G, H, pos_G)
    cmap = plt.get_cmap("tab10")
    colors = {u: cmap(k % 10) for k, u in enumerate(G.nodes())}
    node_colors = [colors[node[0]] for node in P.nodes()]
    nx.draw_networkx_edges(P, pos, ax=ax, alpha=0.5, width=1.2)
    nx.draw_networkx_nodes(P, pos, ax=ax, node_color=node_colors, node_size=180)
    ax.set_title(title, fontsize=16)
    ax.set_aspect("equal")
    ax.axis("off")


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
    os.makedirs(out_dir, exist_ok=True)

    G = nx.complete_graph(4)          # D = 3-regular, n = 4 vertices ("big" graph)
    H = nx.cycle_graph(3)              # d = 2-regular, D = 3 vertices ("small" graph)

    port, port_index = build_port_labeling(G)
    R = replacement_product(G, H, port, port_index)
    Z = zigzag_product(G, H, port, port_index)

    pos_G = nx.circular_layout(G, scale=1.0)
    pos_H = nx.circular_layout(H, scale=1.0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    draw_graph(axes[0, 0], G, pos_G, r"$G$ = $K_4$ ($D=3$, $n=4$)")
    draw_graph(axes[0, 1], H, pos_H, r"$H$ = $C_3$ ($d=2$, $D=3$)",
               node_color="#DD8452")
    draw_product(axes[1, 0], R, G, H, pos_G,
                 r"Producto de reemplazo $G \circ_r H$")
    draw_product(axes[1, 1], Z, G, H, pos_G,
                 r"Producto Zig-Zag $G \circ_z H$")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "graph_products_demo.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
