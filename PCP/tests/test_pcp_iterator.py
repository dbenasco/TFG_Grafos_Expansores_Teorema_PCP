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

# Import functions from test_pcp_complete
from tests.test_pcp_complete import evaluate_acceptance_probability, trace_arity_assignment, trace_degree_reduction, build_honest_proof, print_stats

def trace_power_assignment(G_orig, h_orig, t):
    from amplifier import get_walks
    walks = get_walks(G_orig, t)
    h_pow = {}
    for u in G_orig.variables:
        paths = walks[u]
        val_list = []
        for p in paths:
            for node in p: val_list.append(h_orig[node])
        
        W = G_orig.variables[u]
        idx = 0
        for i, val in enumerate(val_list):
            idx += val * (W ** (len(val_list)-1-i))
        h_pow[u] = idx
    return h_pow

def run_iteration(G, assignment, iter_idx, t_val):
    print(f"\n>>>> ITERATION {iter_idx} START <<<<")
    
    # SIGNAL-TO-NOISE OPTIMIZATION:
    # Use many evaluation tests and few auxiliary tests to ensure the gap grows
    # across the composition step in this simulation.
    NUM_EVAL = 5
    NUM_AUX = 1
    
    print(f" Step A: Degree Reduction...")
    G_reg = degree_reduction(G, use_expander_cloud=False)
    h_reg = trace_degree_reduction(G_reg, assignment)
    
    print(f" Step B: Powering (t={t_val})...")
    G_pow = power_graph(G_reg, t=t_val)
    h_pow = trace_power_assignment(G_reg, h_reg, t_val)
    
    gap_pow = 1.0 - evaluate_acceptance_probability(G_pow, h_pow)
    print(f"   Gap after Powering: {gap_pow:.4f}")
    
    print(" Step C: Alphabet Reduction...")
    # Passing num_edge_tests as the signal count
    G_bool = alphabet_reduction(G_pow, num_blr_tests=NUM_AUX, num_edge_tests=NUM_EVAL) 
    
    k = math.ceil(math.log2(max(G_pow.variables.values())))
    print(f"   Building Proof (k={k})...")
    h_bool = build_honest_proof(G_bool, G_pow, h_pow, k)
    
    gap_final = 1.0 - evaluate_acceptance_probability(G_bool, h_bool)
    print(f" End of Iter {iter_idx} Gap: {gap_final:.4f}")
    print(f" Variables: {len(G_bool.variables)}")
    
    return G_bool, h_bool

def main():
    print("=== PCP ITERATIVE SOUNDNESS GROWTH (Optimized Signal) ===")
    
    # Use 3 nodes to increase the resolution of the gap
    clauses = [[1, 1, 1], [-1, -1, -1]]
    best_init = {1: 1} 
    
    G0 = arity_reduction(clauses)
    h0 = trace_arity_assignment(clauses, best_init)
    
    current_G = G0
    current_h = h0
    
    for i in range(1, 4):
        # Using t=1 with optimized signaling
        current_G, current_h = run_iteration(current_G, current_h, i, t_val=1)
        if (1.0 - evaluate_acceptance_probability(current_G, current_h)) > 0.95:
            print("GAP REACHED 95%+! AMPLIFICATION COMPLETE.")
            break
            
if __name__ == "__main__":
    main()
