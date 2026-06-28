"""
Detailed Spectral Gap Analysis
==============================
Generates a high-resolution version of the plot_spectral_gap.png 
to better visualize the convergence towards the Ramanujan bound.
"""

import numpy as np
import math
import sys
import os
import io
import contextlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── import the expander code ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from SipserSpielmanExpander import InnerCode, SSExpanderCode
from expander_quality import spectral_gap

def run_detailed_spectral():
    c, d = 6, 6
    n_min, n_max = 10, 100000
    # 60 points for a smooth curve
    n_values = np.unique(np.logspace(math.log10(n_min), math.log10(n_max), num=60, dtype=int))
    
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
    os.makedirs(out_dir, exist_ok=True)

    Hs = np.ones((1, d), dtype=np.uint8)
    inner = InnerCode(d, Hs)

    lambda_G_list = []
    ram_bound = (math.sqrt(c - 1) + math.sqrt(d - 1)) / math.sqrt(c * d)

    print(f"Generando gráfica detallada de λ(G) con {len(n_values)} puntos...")

    for n in n_values:
        # Adjustment for d-regularity: (c*n) % d == 0
        n_adj = n
        while (c * n_adj) % d != 0:
            n_adj += 1
            
        print(f"  Procesando n = {n_adj}...", end="\r")
        
        # Build graph with random seed
        random_seed = int(np.random.randint(0, 2**31))
        with contextlib.redirect_stdout(io.StringIO()):
            code = SSExpanderCode(n_adj, c, d, inner, seed=random_seed, allow_multi_edges=True)
            
        sg = spectral_gap(code)
        lambda_G_list.append(sg["lambda_G"])

    print("\nGraficando...")
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, lambda_G_list, 'o-', markersize=4, linewidth=1.5, label=r"$\lambda(G)$ medido")
    plt.axhline(ram_bound, color='red', linestyle='--', linewidth=2, label=f"Cota de Ramanujan ({ram_bound:.4f})")
    
    plt.xscale('log')
    plt.xlabel(r"Tamaño del Grafo $n$ (Escala Log)", fontsize=12)
    plt.ylabel(r"Segundo Valor Singular $\lambda(G)$", fontsize=12)
    plt.title(f"Convergencia de la Expansión Espectral $(c,d)=({c},{d})$\nEstabilidad para grafos de gran escala", fontsize=14)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.savefig(os.path.join(out_dir, "plot_spectral_gap.png"), dpi=200, bbox_inches='tight')
    print(f"Gráfica guardada en {out_dir}/plot_spectral_gap.png")

if __name__ == "__main__":
    run_detailed_spectral()
