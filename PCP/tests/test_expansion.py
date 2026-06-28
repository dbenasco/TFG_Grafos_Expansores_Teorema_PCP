"""
Experiment: Verify expansion properties of the 'Nice' constraint graph.

Builds a somewhat large 3SAT formula, runs it through the full PCP
Preprocessing phase:
  1. Arity Reduction (3SAT -> 2CSP)
  2. Degree Reduction (Cycle of Copies, 3-regular)
  3. Self-Loop Doubling (Claim 22.38, 6-regular)
  4. Expansion Overlay (Claim 22.39, +3 edges = 9-regular)

Then extracts the Adjacency Matrix and computes the eigenvalues
to check the expansion parameter lambda <= 0.9.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.linalg as la
from nice_transformation import (
    arity_reduction, 
    degree_reduction, 
    add_expander_edges
)

def build_large_3sat(num_vars=20, num_clauses=50):
    """Generates a random 3SAT formula."""
    clauses = []
    for _ in range(num_clauses):
        # Pick 3 distinct variables 1..num_vars
        vars_chosen = np.random.choice(np.arange(1, num_vars+1), 3, replace=False)
        # Random negations
        signs = np.random.choice([-1, 1], 3)
        clause = (vars_chosen * signs).tolist()
        clauses.append(clause)
    return clauses

def get_adjacency_matrix(G):
    """Constructs the dense adjacency matrix for the ConstraintGraph."""
    n = len(G.variables)
    var_list = list(G.variables.keys())
    var_idx = {v: i for i, v in enumerate(var_list)}
    
    A = np.zeros((n, n))
    for u, v, _ in G.constraints:
        i, j = var_idx[u], var_idx[v]
        A[i, j] += 1
        if i != j:
            A[j, i] += 1
    return A

def measure_expansion(A, d):
    """Computes eigenvalues of A and returns the expansion parameter."""
    eigenvalues = la.eigvalsh(A)
    # Sort by absolute value in descending order
    sorted_evs = sorted(eigenvalues, key=abs, reverse=True)
    
    print(f"Top 5 eigenvalues: {[round(e, 4) for e in sorted_evs[:5]]}")
    
    # The largest eigenvalue should be exactly d (for a d-regular graph)
    lam_1 = max(sorted_evs)
    assert abs(lam_1 - d) < 1e-5, f"Graph is not {d}-regular! Max eig={lam_1}"
    
    # The expansion parameter is the second largest eigenvalue (in absolute value)
    lam_2 = abs(sorted_evs[1])
    normalized_lambda = lam_2 / d
    return normalized_lambda

def run_experiment():
    print("Generating random 3SAT formula (20 vars, 50 clauses)...")
    clauses = build_large_3sat(20, 50)
    
    print("\nRunning PCP Nice Transformation Pipeline...")
    G1 = arity_reduction(clauses)
    print(f"  Step 1.2 (Arity Reduction): {G1}")
    
    G2 = degree_reduction(G1)
    print(f"  Step 1.3 (Degree Reduction): {G2}")
    
    # Arora & Barak Sec 22.2.3 handles both self-loops and expander edges in one step
    G4 = add_expander_edges(G2, c_exp=3)
    print(f"  Sec 22.2.3 (Expander Overlay & Self Loops): {G4}")
    
    # Check degree: d=6. Original padded to 6 + Expander (6) + Self loops (12) = 24-regular.
    d_expected = 24
    print(f"\nExpected degree: {d_expected}")
    
    A = get_adjacency_matrix(G4)
    lambda_param = measure_expansion(A, d_expected)
    
    print(f"\nRESULTS:")
    print(f"  Graph Size: {len(G4.variables)} vertices")
    print(f"  Normalized λ_2 (Expansion Parameter): {lambda_param:.4f}")
    
    if lambda_param <= 0.9:
        print("  ✓ SUCCESS: Graph is an excellent expander (λ <= 0.9)!")
    else:
        print("  ✗ FAILURE: Graph expansion parameter is too high.")

if __name__ == "__main__":
    run_experiment()
