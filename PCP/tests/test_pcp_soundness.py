import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import random
from collections import defaultdict

from constraint_graph import ConstraintGraph
from nice_transformation import arity_reduction, degree_reduction, _satisfying_assignments, _compute_degrees
from alphabet_reduction import alphabet_reduction, walsh_hadamard_encode
from amplifier import power_graph

# Import functions from test_pcp_complete (or redefine if needed)
from tests.test_pcp_complete import evaluate_acceptance_probability, trace_arity_assignment, trace_degree_reduction, build_honest_proof

def run_pipeline_with_t(t_val, clauses, truth_assignment):
    # 1. Arity
    G1 = arity_reduction(clauses)
    h1 = trace_arity_assignment(clauses, truth_assignment)
    
    # 2. Degree
    G2 = degree_reduction(G1, use_expander_cloud=False)
    h2 = trace_degree_reduction(G2, h1)
    
    # 3. Powering
    G_pow = power_graph(G2, t=t_val)
    from amplifier import get_walks
    walks = get_walks(G2, t_val)
    W_orig = 7
    h_pow = {}
    for u in G2.variables:
        paths = walks[u]
        val_list = []
        for p in paths:
            for node in p: val_list.append(h2[node])
        idx = 0
        for i, val in enumerate(val_list):
            idx += val * (W_orig ** (len(val_list)-1-i))
        h_pow[u] = idx
        
    # 4. Alphabet Reduction
    G_bool = alphabet_reduction(G_pow, num_blr_tests=10)
    k = math.ceil(math.log2(max(G_pow.variables.values())))
    proof = build_honest_proof(G_bool, G_pow, h_pow, k)
    
    acc = evaluate_acceptance_probability(G_bool, proof)
    return 1.0 - acc

def main():
    print("=== PCP SOUNDNESS AMPLIFICATION TEST (UNSAT CASE) ===")
    
    # TRULY UNSATISFIABLE 3SAT: (x1 and not x1)
    clauses = [
        [1, 1, 1],
        [-1, -1, -1]
    ]
    # No matter what we pick for x1, one clause fails.
    best_assignment = {1: 1} # Satisfies clause 0, fails clause 1.
    
    print("\nRunning T=0 Soundness...")
    gap0 = run_pipeline_with_t(0, clauses, best_assignment)
    print(f"Unsat Gap (T=0): {gap0:.4f}")
    
    print("\nRunning T=1 Soundness (Amplified)...")
    gap1 = run_pipeline_with_t(1, clauses, best_assignment)
    print(f"Unsat Gap (T=1): {gap1:.4f}")
    
    print("\n=== Result Summary ===")
    if gap1 > gap0:
        factor = gap1 / gap0 if gap0 > 0 else float('inf')
        print(f"SUCCESS: Gap amplified from {gap0:.4f} to {gap1:.4f} (Factor: {factor:.2f}x)")
    elif gap1 == gap0:
        print(f"INFO: Gap stayed the same at {gap1:.4f}. (T=1 over a small graph might not always boost the RANDOMIZED gap depending on test counts)")
    else:
        print(f"FAILURE: Gap decreased from {gap0:.4f} to {gap1:.4f}. Check implementation.")

if __name__ == "__main__":
    main()
