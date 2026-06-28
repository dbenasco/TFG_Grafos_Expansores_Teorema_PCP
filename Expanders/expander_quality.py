"""
Expander Quality Analysis
=========================
Measures how large (c,d)-regular random bipartite graphs need to be
to exhibit good spectral expansion.

Key metrics:
  1. Spectral gap:  λ₂  (second largest singular value of the
     normalised biadjacency matrix).  A good expander has λ₂ small.
     The Ramanujan bound (best possible for random regular graphs)
     is  λ₂ ≤ (√(c-1) + √(d-1)) / √(c·d).

  2. Vertex expansion ratio:  min_{|S|≤αn} |Γ(S)| / |S|
     estimated by random sampling of small subsets.

  3. Error-correction test:  fraction of random error patterns that
     the bit-flip decoder can correct, for increasing error weight.

Usage:
    conda activate TFGMates
    python expander_quality.py
"""

import numpy as np
import itertools
import math
import sys
import os
import io
import contextlib

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── import the expander code from the same directory ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from SipserSpielmanExpander import InnerCode, SSExpanderCode


# ═══════════════════════════════════════════════════════
# 1.  Spectral Gap  (λ(G) = second largest singular value)
# ═══════════════════════════════════════════════════════
def spectral_gap(code: SSExpanderCode) -> dict:
    """
    Compute the singular values of the normalized biadjacency matrix:
       M = (1/√(cd)) · A
    where A is the {0,1} biadjacency matrix (m × n).

    Returns a dict with:
      sigma_1  : largest singular value (should be ≈ 1.0)
      lambda_G : second largest singular value (the smaller, the better)
      ramanujan_bound : theoretical optimum (√(c-1)+√(d-1))/√(cd)
      gap      : 1.0 - lambda_G (spectral gap)
    """
    m, n = code.m, code.n
    normalization = 1.0 / math.sqrt(code.c * code.d)

    # For small n, use dense SVD; for large n, use iterative sparse SVD if possible.
    if HAS_SCIPY and n > 500:
        # Construct sparse matrix
        rows, cols, data = [], [], []
        for u in range(m):
            for v in code.chk_nbh[u]:
                rows.append(u)
                cols.append(v)
                data.append(normalization)
        M_sparse = csr_matrix((data, (rows, cols)), shape=(m, n))
        # Find just the top 2 singular values
        # svds finds k largest by default
        sigmas = svds(M_sparse, k=2, which='LM', return_singular_vectors=False)
        sigmas = np.sort(sigmas)[::-1]
    else:
        # Build dense biadjacency matrix A
        A = np.zeros((m, n), dtype=np.float64)
        for u in range(m):
            for v in code.chk_nbh[u]:
                A[u, v] += 1.0
        M = A * normalization
        sigmas = np.linalg.svd(M, compute_uv=False)

    sigma_1 = sigmas[0]
    lambda_G = sigmas[1] if len(sigmas) > 1 else 0.0
    ram = (math.sqrt(code.c - 1) + math.sqrt(code.d - 1)) / math.sqrt(code.c * code.d)

    return {
        "sigma_1": sigma_1,
        "lambda_G": lambda_G,
        "ramanujan_bound": ram,
        "gap": 1.0 - lambda_G,
    }


# ═══════════════════════════════════════════════════════
# 2.  Vertex Expansion  (Sampling the ratio |Γ(S)|/|S|)
# ═══════════════════════════════════════════════════════
def vertex_expansion(code: SSExpanderCode,
                     subset_sizes=None,
                     num_samples: int = 100,
                     seed: int = 42) -> dict:
    """
    Estimates the expansion ratio min_{|S|=s} |Γ(S)|/|S|.

    Interpreting the Graph:
    - Max possible ratio is 'c' (every edge from S leads to a unique neighbor).
    - Higher ratio means better connectivity and fewer 'collisions' between nodes.
    - Good expanders maintain a high ratio even as subset size |S| increases.
    - If the ratio is > c/2, it guarantees bit-flip decoding works for errors in S.
    """
    rng = np.random.default_rng(seed)
    if subset_sizes is None:
        subset_sizes = [max(1, int(f * code.n)) for f in [0.01, 0.05, 0.1]]

    results = {}
    for s in subset_sizes:
        if s > code.n or s <= 0: continue
        min_ratio = float("inf")
        # Sampling many small subsets to find the bottleneck
        for _ in range(num_samples):
            S = rng.choice(code.n, size=s, replace=False)
            neighbours = set()
            for v in S:
                neighbours.update(code.var_nbh[v])
            ratio = len(neighbours) / s
            if ratio < min_ratio:
                min_ratio = ratio
        results[s] = min_ratio

    return results


# ═══════════════════════════════════════════════════════
# 3.  Decoding Performance
# ═══════════════════════════════════════════════════════
def error_correction_test_fast(code: SSExpanderCode,
                               error_weights: list[int],
                               num_trials: int = 20,
                               max_rounds: int = 100,
                               seed: int = 42) -> dict:
    """
    Faster version of error correction test that uses the all-zeros
    codeword (always valid for linear codes) to avoid computing the
    generator matrix G (O(n^3)).
    """
    rng = np.random.default_rng(seed)
    results = {}

    for w in error_weights:
        if w > code.n or w <= 0: continue
        successes = 0
        for _ in range(num_trials):
            # All-zeros is always a valid codeword
            noisy = np.zeros(code.n, dtype=np.uint8)
            positions = rng.choice(code.n, size=w, replace=False)
            noisy[positions] = 1 # add errors

            decoded, rounds = code.decode_parallel(
                noisy, max_rounds=max_rounds, threshold_majority=code.c // 2
            )
            if rounds >= 0 and np.all(decoded == 0):
                successes += 1
        results[w] = successes / num_trials

    return results


# ═══════════════════════════════════════════════════════
# 4. O(1) Error Location (Local Decodability)
# ═══════════════════════════════════════════════════════
def probabilistic_verification_test(code: SSExpanderCode,
                                  error_rates: list[float],
                                  k_samples_list: list[int],
                                  num_trials: int = 200,
                                  seed: int = 42) -> dict:
    """
    Tests the Completeness and Soundness of the local probabilistic verifier.
    
    Returns a dictionary mapping error rates (0 for Completeness, >0 for Soundness)
    to a list of acceptance probabilities corresponding to each k in k_samples_list.
    """
    rng = np.random.default_rng(seed)
    results = {}
    
    def get_random_valid_word(code: SSExpanderCode, rng_val):
        # To avoid the O(n^3) cost of creating the Generator matrix for n > 5000,
        # we generate a random valid codeword by starting with a random noise vector
        # and running the parallel bit-flip decoder. If the noise is within the 
        # correcting radius, it converges to a valid codeword.
        # We try a few times until we hit a valid one (syndrome == 0).
        max_attempts = 10
        for _ in range(max_attempts):
            # Inject a small amount of random noise (e.g., 2% of n)
            num_errors = max(1, int(0.02 * code.n))
            word = np.zeros(code.n, dtype=np.uint8)
            positions = rng_val.choice(code.n, size=num_errors, replace=False)
            word[positions] = 1
            
            # Decode it using parallel bit-flip
            decoded, success = code.decode_parallel(word)
            if success:
                return decoded
        
        # Fallback to all-zeros if it constantly fails to converge 
        # (unlikely for good expanders with small initial noise)
        return np.zeros(code.n, dtype=np.uint8)

    print(f"      Generating random valid words (fast iterative method)...")
    
    # 1. Completeness Test (0 errors)
    completeness_probs = []
    for k in k_samples_list:
        accepted = 0
        for _ in range(num_trials):
            valid_word = get_random_valid_word(code, rng)
            
            with contextlib.redirect_stdout(io.StringIO()):
                if code.local_verify(valid_word, num_samples=k, seed=int(rng.integers(0, 2**31))):
                    accepted += 1
        completeness_probs.append(accepted / num_trials)
        
    results[0.0] = completeness_probs
    
    # 2. Soundness Test (fixed error rates)
    for err_rate in error_rates:
        soundness_probs = []
        num_errors = max(1, int(err_rate * code.n))
        
        for k in k_samples_list:
            accepted = 0
            for _ in range(num_trials):
                noisy_word = get_random_valid_word(code, rng).copy()
                
                positions = rng.choice(code.n, size=num_errors, replace=False)
                noisy_word[positions] ^= 1 
                
                with contextlib.redirect_stdout(io.StringIO()):
                    if code.local_verify(noisy_word, num_samples=k, seed=int(rng.integers(0, 2**31))):
                        accepted += 1
            soundness_probs.append(accepted / num_trials)
        
        results[err_rate] = soundness_probs

    return results


# ═══════════════════════════════════════════════════════
# Main Experiment Loop
# ═══════════════════════════════════════════════════════
def run_experiments():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c, d = 6, 6
    Hs = np.ones((1, d), dtype=np.uint8)
    inner = InnerCode(d, Hs)

    # Expanded range to 10k
    n_values = [10, 50, 100, 200, 300, 500, 1000, 3000, 7500, 10000]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Running experiments up to n=10000 (Scipy Sparse={HAS_SCIPY})")

    lambda_G_list = []
    ram_bound = None
    codes = {}
    ve_data = {frac: [] for frac in [0.01, 0.02, 0.05, 0.10]}
    ec_data = {frac: [] for frac in [0.01, 0.02, 0.03, 0.05]}
    
    # PCP test params (we will plot this for a specific large n, e.g., tracking the largest)
    pcp_k_samples = list(range(1, 41, 4))
    pcp_error_rates = [0.01, 0.05, 0.10]
    pcp_final_results = None
    
    # Independence of n test params
    pcp_fixed_k = 15
    pcp_fixed_err_list = [0.01, 0.05, 0.10, 0.20, 0.30]
    pcp_indep_comp = []
    pcp_indep_sound = {err: [] for err in pcp_fixed_err_list}

    for n in n_values:
        print(f"  Testing n = {n}...")
        # Build graph with random seed
        random_seed = int(np.random.randint(0, 2**31))
        with contextlib.redirect_stdout(io.StringIO()):
            code = SSExpanderCode(n, c, d, inner, seed=random_seed, allow_multi_edges=True)
        codes[n] = code

        # 1. Spectral
        sg = spectral_gap(code)
        lambda_G_list.append(sg["lambda_G"])
        ram_bound = sg["ramanujan_bound"]

        # 2. Vertex Expansion
        subset_fracs = [0.01, 0.02, 0.05, 0.10]
        sizes = [max(1, int(f * n)) for f in subset_fracs]
        # Reducing samples for large n to keep it fast
        ve = vertex_expansion(code, subset_sizes=sizes, num_samples=50 if n < 1000 else 20)
        for f, s in zip(subset_fracs, sizes):
            ve_data[f].append(ve.get(s, 0))

        # 3. Error correction
        print(f"    Error trials...")
        err_fracs = [0.01, 0.02, 0.03, 0.05]
        weights = [max(1, int(f * n)) for f in err_fracs]
        
        ec = {}
        rng_ec = np.random.default_rng(random_seed)
        for w in weights:
            success_count = 0
            for _ in range(30 if n < 1000 else 10):
                # Optimization: We use all-zeros codeword for large n as it's always valid
                word = np.zeros(n, dtype=np.uint8)
                positions = rng_ec.choice(n, size=w, replace=False)
                word[positions] = 1
                
                with contextlib.redirect_stdout(io.StringIO()):
                    _, success = code.decode_parallel(word)
                if success:
                    success_count += 1
            ec[w] = success_count / (30 if n < 1000 else 10)

        for f, w in zip(err_fracs, weights):
            ec_data[f].append(ec.get(w, 0))
        # 4. Probabilistic Verification Test 
        # Computing the generator G is O(n^3) and takes too long / too much RAM for n=10000.
        # We will run this exhaustive test on n=500 instead, which is large enough
        # to show the probabilistic properties clearly.
        if n == 500:
            print(f"    Running exhaustive Probabilistic Verifier Test on random valid words...")
            pcp_final_results = probabilistic_verification_test(code, pcp_error_rates, pcp_k_samples, num_trials=500, seed=random_seed)

        # 5. Independence of n Test
        # Test acceptance prob for fixed k=15, across several fixed error rates, for all n
        print(f"    Running Independence Test (fixed k=15, error rates {pcp_fixed_err_list})...")
        indep_results = probabilistic_verification_test(code, pcp_fixed_err_list, [pcp_fixed_k], num_trials=200, seed=random_seed)
        pcp_indep_comp.append(indep_results[0.0][0])
        for err in pcp_fixed_err_list:
            pcp_indep_sound[err].append(indep_results[err][0])

    # --- Plot 1: λ(G) ---
    plt.figure(figsize=(9, 6))
    plt.plot(n_values, lambda_G_list, 'o-', linewidth=2, label=r"$\lambda(G)$ medido")
    plt.axhline(ram_bound, color='red', linestyle='--', label=f"Cota de Ramanujan ({ram_bound:.4f})")
    plt.xscale('log')
    plt.xlabel(r"Tamaño del Grafo $n$ (Escala Log)", fontsize=12)
    plt.ylabel(r"Segundo Valor Singular $\lambda(G)$", fontsize=12)
    plt.title(f"Expansión Espectral $(c,d)=({c},{d})$\nUn $\lambda(G)$ menor significa mejor expansión", fontsize=14)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(out_dir, "plot_spectral_gap.png"), dpi=150)

    # --- Plot 2: Vertex Expansion ---
    plt.figure(figsize=(9, 6))
    for f in subset_fracs:
        plt.plot(n_values, ve_data[f], 's-', label=f"Subconjuntos de tamaño {f:.0%}·n")
    plt.axhline(c, color='black', alpha=0.3, label=f"Máximo Teórico (grado c={c})")
    plt.axhline(c/2, color='orange', linestyle='--', label=f"Umbral de Decodificación (c/2={c/2})")
    plt.xscale('log')
    plt.ylim(0, c + 0.5)
    plt.xlabel(r"Tamaño del Grafo $n$", fontsize=14)
    plt.ylabel(r"Ratio de Expansión Mínimo $|\Gamma(S)|/|S|$", fontsize=14)
    plt.title("Expansión de Vértices: Fuerza de Conectividad\nCuántos vecinos distintos tiene un subconjunto", fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(out_dir, "plot_vertex_expansion.png"), dpi=150)

    # --- Plot 3: Error Correction ---
    plt.figure(figsize=(9, 6))
    for f in err_fracs:
        plt.plot(n_values, [v*100 for v in ec_data[f]], '^-', label=f"{f:.0%} errores")
    plt.xscale('log')
    plt.ylim(-5, 105)
    plt.xlabel(r"Tamaño del Grafo $n$", fontsize=12)
    plt.ylabel("Tasa de éxito de decodificación (%)", fontsize=12)
    plt.title("Rendimiento de la Corrección de Errores\n¿Converge el bit-flip a la palabra de código correcta?", fontsize=14)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(out_dir, "plot_error_correction.png"), dpi=150)

    # --- Plot 4: Probabilistic Verification (Completeness and Soundness) ---
    if pcp_final_results is not None:
        plt.figure(figsize=(9, 6))
        
        # Plot Completeness (should be a flat line at 1.0)
        plt.plot(pcp_k_samples, pcp_final_results[0.0], 'g^-', linewidth=2, label="Completitud (Acepta palabra correcta con Prob = 1)")
        
        # Plot Soundness for each error rate
        colors = ['orange', 'red', 'darkred']
        for i, err_rate in enumerate(pcp_error_rates):
            plt.plot(pcp_k_samples, pcp_final_results[err_rate], 'o-', color=colors[i % len(colors)],
                     label=f"Solidez (Acepta palabra con {err_rate:.0%} de errores)")
            
        plt.axhline(0.5, color='black', linestyle='--', linewidth=2, label=r"Umbral de Solidez (Prob $\leq 0.5$)")
        
        plt.ylim(-0.05, 1.05)
        plt.xlabel(r"Número de bloques de restricciones consultados ($k$)", fontsize=12)
        plt.ylabel("Probabilidad de Aceptación", fontsize=12)
        plt.title(f"Verificación Probabilística del Código Expansor ($n=500$)\nCompletitud $= 1$, Solidez $\leq 1/2$", fontsize=14)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(out_dir, "plot_probabilistic_verification.png"), dpi=150)
        
    # --- Plot 5: Independence of n ---
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, pcp_indep_comp, 'g^-', linewidth=2, label="Completitud (0% errores)")
    indep_colors = ['gold', 'orange', 'red', 'darkred', 'maroon']
    for i, err in enumerate(pcp_fixed_err_list):
        plt.plot(n_values, pcp_indep_sound[err], 'o-', color=indep_colors[i % len(indep_colors)],
                 label=f"Solidez ({err:.0%} errores)")
    plt.axhline(0.5, color='black', linestyle='--', linewidth=2, label=r"Umbral de Solidez (Prob $\leq 0.5$)")

    plt.xscale('log')
    plt.ylim(-0.05, 1.05)
    plt.xlabel(r"Tamaño del Grafo $n$ (Escala Logarítmica)", fontsize=12)
    plt.ylabel("Probabilidad de Aceptación", fontsize=12)
    plt.title(f"Independencia de la Longitud $n$\n(Verificando $k={pcp_fixed_k}$ bloques aleatorios)", fontsize=14)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_probabilistic_independence.png"), dpi=150)

    print(f"\nSaved all graphs to {out_dir}/")

if __name__ == "__main__":
    run_experiments()
