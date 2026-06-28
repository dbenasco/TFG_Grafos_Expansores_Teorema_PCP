"""
Degree Study: Impact of c and d on Expander Properties
======================================================
This script analyzes how variable degree (c) and check degree (d) 
affect the spectral gap, vertex expansion, and error correction 
of Sipser-Spielman expander codes.

1. Impact of Density: Vary c while keeping R = 1 - c/d = 0.5 (d = 2c).
2. Impact of Rate: Vary d while keeping c = 6 fixed.

Usage:
    conda activate TFGMates
    python degree_study.py
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
from expander_quality import spectral_gap, vertex_expansion, error_correction_test_fast

def run_degree_study():
    n_fixed = 2000
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs_degree")
    os.makedirs(out_dir, exist_ok=True)

    # ═══════════════════════════════════════════════════════
    # PART 1: Varying density (c, d=2c) - Constant Rate R=0.5
    # ═══════════════════════════════════════════════════════
    c_values_p1 = [3, 4, 5, 6, 8, 10, 12]
    
    res_spectral_p1 = []
    res_expansion_p1 = []
    res_decoding_p1 = []
    
    print(f"Part 1: Varying c (d=2c, Rate=0.5, n={n_fixed})")
    for c in c_values_p1:
        d = 2 * c
        print(f"  Testing (c,d) = ({c},{d})...")
        # Dummy inner code (single parity check)
        Hs = np.ones((1, d), dtype=np.uint8)
        inner = InnerCode(d, Hs)
        
        with contextlib.redirect_stdout(io.StringIO()):
            code = SSExpanderCode(n_fixed, c, d, inner, seed=42, allow_multi_edges=True)
        
        # Spectral
        sg = spectral_gap(code)
        res_spectral_p1.append(sg["lambda_G"])
        
        # Vertex Expansion (fixed |S|=5%n)
        ve = vertex_expansion(code, subset_sizes=[int(0.05 * n_fixed)], num_samples=50)
        res_expansion_p1.append(ve[int(0.05 * n_fixed)])
        
        # Error correction (fixed weight w=2%n)
        ec = error_correction_test_fast(code, error_weights=[int(0.02 * n_fixed)], num_trials=30)
        res_decoding_p1.append(ec[int(0.02 * n_fixed)])

    # ═══════════════════════════════════════════════════════
    # PART 2: Varying rate (c=6, d varies) - d increases, Rate increases
    # ═══════════════════════════════════════════════════════
    d_values_p2 = [8, 10, 12, 18, 24, 30, 48]
    c_fixed = 6

    res_spectral_p2 = []
    res_expansion_p2 = []
    res_decoding_p2 = []
    rates_p2 = []

    print(f"\nPart 2: Varying d (c={c_fixed}, n={n_fixed})")
    for d in d_values_p2:
        # Ensure n is divisible by d/gcd(c,d) - for 2000, we might need to adjust n slightly
        # m = c*n/d. For c=6, n=2000: m = 12000/d.
        # If d doesn't divide 12000 exactly, we adjust n.
        n_adj = n_fixed
        while (c_fixed * n_adj) % d != 0:
            n_adj += 1
            
        rate = 1.0 - (c_fixed / d)
        rates_p2.append(rate)
        print(f"  Testing (c,d) = ({c_fixed},{d}) [Rate={rate:.2f}, adj_n={n_adj}]...")
        
        Hs = np.ones((1, d), dtype=np.uint8)
        inner = InnerCode(d, Hs)
        
        with contextlib.redirect_stdout(io.StringIO()):
            code = SSExpanderCode(n_adj, c_fixed, d, inner, seed=42, allow_multi_edges=True)
            
        res_spectral_p2.append(spectral_gap(code)["lambda_G"])
        ve = vertex_expansion(code, subset_sizes=[int(0.05 * n_adj)], num_samples=50)
        res_expansion_p2.append(ve[int(0.05 * n_adj)])
        ec = error_correction_test_fast(code, error_weights=[int(0.02 * n_adj)], num_trials=30)
        res_decoding_p2.append(ec[int(0.02 * n_adj)])

    # ────── PLOTTING PART 1 (DENSITY) ──────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1.1 Spectral
    axes[0].plot(c_values_p1, res_spectral_p1, 'o-', color='#1e40af')
    axes[0].set_title(r"Expansión Algebraica ($\lambda(G)$)" + "\nVariando $c$ (Tasa=0.5)", fontsize=17)
    axes[0].set_xlabel("Grado $c$", fontsize=15)
    axes[0].set_ylabel(r"$\lambda(G)$ (menor es mejor)", fontsize=15)
    axes[0].tick_params(axis='both', labelsize=12)

    # 1.2 Vertex
    axes[1].plot(c_values_p1, res_expansion_p1, 's-', color='#16a34a', label="Ratio Absoluto")
    axes[1].set_title(r"Expansión de Vértices $|\Gamma(S)|/|S|$" + "\n|S|=5%n fijo, Tasa=0.5", fontsize=17)
    axes[1].set_xlabel("Grado $c$", fontsize=15)
    axes[1].set_ylabel("Ratio mín. encontrado", fontsize=15)
    axes[1].tick_params(axis='both', labelsize=12)

    # 1.3 Error Correction
    axes[2].plot(c_values_p1, [v*100 for v in res_decoding_p1], '^-', color='#dc2626')
    axes[2].set_title("Éxito de Decodificación (%)\nErrores=2%n fijos, Tasa=0.5", fontsize=17)
    axes[2].set_xlabel("Grado $c$", fontsize=15)
    axes[2].set_ylabel("Decodificación correcta (%)", fontsize=15)
    axes[2].set_ylim(-5, 105)
    axes[2].tick_params(axis='both', labelsize=12)

    plt.suptitle("Impacto de la Densidad del Grafo (Aumentando $c$ con Tasa Fija $R=0.5$)", fontsize=19, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "study_impact_density.png"), dpi=150, bbox_inches='tight')

    # ────── PLOTTING PART 2 (RATE) ──────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 2.1 Spectral vs Rate
    axes[0].plot(rates_p2, res_spectral_p2, 'o-', color='#1e40af')
    axes[0].set_title(r"Brecha Algebraica $\lambda(G)$" + "\nvs Tasa de Código ($c=6$ fijo)", fontsize=17)
    axes[0].set_xlabel("Tasa de Código $R$", fontsize=15)
    axes[0].set_ylabel(r"$\lambda(G)$", fontsize=15)
    axes[0].tick_params(axis='both', labelsize=12)

    # 2.2 Vertex Expansion vs Rate
    axes[1].plot(rates_p2, res_expansion_p2, 's-', color='#16a34a')
    axes[1].set_title("Expansión de Vértices vs Tasa\n|S|=5%n fijo, $c=6$ fijo", fontsize=17)
    axes[1].set_xlabel("Tasa de Código $R$", fontsize=15)
    axes[1].set_ylabel("Ratio de Expansión", fontsize=15)
    axes[1].tick_params(axis='both', labelsize=12)

    # 2.3 Success vs Rate
    axes[2].plot(rates_p2, [v*100 for v in res_decoding_p2], '^-', color='#dc2626')
    axes[2].set_title("Éxito de Decodificación (%) vs Tasa\nErrores=2%n fijos, $c=6$ fijo", fontsize=17)
    axes[2].set_xlabel("Tasa de Código $R$", fontsize=15)
    axes[2].set_ylabel("Decodificación correcta (%)", fontsize=15)
    axes[2].set_ylim(-5, 105)
    axes[2].tick_params(axis='both', labelsize=12)

    plt.suptitle("Impacto de la Tasa de Código (Aumentando $d$ con $c=6$ fijo)", fontsize=19, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "study_impact_rate.png"), dpi=150, bbox_inches='tight')

    print(f"\nDone. Degree study graphs saved to {out_dir}/")

if __name__ == "__main__":
    run_degree_study()
