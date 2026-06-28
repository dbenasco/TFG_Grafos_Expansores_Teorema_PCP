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

def evaluate_acceptance_probability(G, assignment, debug=False):
    if not G.constraints: return 1.0
    satisfied = 0
    total = len(G.constraints)
    for u, v, check_fn in G.constraints:
        val_u = assignment.get(u)
        val_v = assignment.get(v)
        if val_u is None or val_v is None:
            if debug: print(f"[DEBUG] MISSING VAR: u={u} ({val_u}), v={v} ({val_v})")
            continue
            
        if check_fn(val_u, val_v):
            satisfied += 1
        elif debug:
            print(f"[DEBUG] FAILED Constraint: u={u} (val={val_u}), v={v} (val={val_v})")
    return satisfied / total

def trace_arity_assignment(clauses, truth_assignment):
    assign = {}
    for i, clause in enumerate(clauses):
        var_vals = tuple(truth_assignment.get(abs(lit), 0) for lit in clause)
        sat_list = _satisfying_assignments(clause)
        match_tuple = tuple(truth_assignment.get(abs(lit), 0) for lit in clause)
        
        try:
            w_idx = sat_list.index(match_tuple)
        except ValueError:
            print(f"Warning: Clause {i} {clause} not satisfied by {match_tuple}!")
            w_idx = 0
            
        assign[("w", i)] = w_idx
        for lit in clause:
            assign[("x", abs(lit))] = truth_assignment.get(abs(lit), 0)
    return assign

def trace_degree_reduction(G_nice, base_assign):
    assign = {}
    for v in G_nice.variables:
        if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], int):
            origin = v[0]
            assign[v] = base_assign.get(origin, 0)
        else:
            assign[v] = base_assign.get(v, 0)
    return assign

def build_honest_proof(G_bool, G_pow, honest_pow, k):
    proof = {}
    
    def get_wh_bit(val, bit_len, query):
        ans = 0
        for i in range(bit_len):
            if ((val >> i) & 1) and ((query >> i) & 1):
                ans ^= 1
        return ans

    # 1. Base Variables (WH and PI)
    for var in G_bool.variables:
        if not isinstance(var, tuple): continue
        if var[0] == "WH":
            _, u, x = var
            proof[var] = get_wh_bit(honest_pow[u], k, x)
        elif var[0] == "PI":
            _, e_id_str, z = var
            idx = int(e_id_str.split("_")[1])
            u, v, check_fn = G_pow.constraints[idx]
            w_val = (honest_pow[u] << k) | honest_pow[v]
            proof[var] = get_wh_bit(w_val, 2*k, z)
        elif var[0] == "W_EVAL":
            _, e_id_str = var
            idx = int(e_id_str.split("_")[1])
            u, v, check_fn = G_pow.constraints[idx]
            a_val, b_val = honest_pow[u], honest_pow[v]
            
            # THE NEW O(1) COMMITMENT: combined_idx = a * W_v + b
            W_v = G_pow.variables[v]
            proof[var] = a_val * W_v + b_val

    # 2. Auxiliary Check Variables
    edges_by_var = defaultdict(list)
    for u, v, check_fn in G_bool.constraints:
        edges_by_var[u].append((v, check_fn, True))
        edges_by_var[v].append((u, check_fn, False))

    for var in G_bool.variables:
        if isinstance(var, tuple) and var[0].startswith("W_") and var[0] != "W_EVAL":
            found = False
            for val in range(4):
                all_good = True
                for neighbor, check_fn, is_u in edges_by_var[var]:
                    if neighbor in proof:
                        n_val = proof[neighbor]
                        if is_u:
                            if not check_fn(val, n_val): all_good = False; break
                        else:
                            if not check_fn(n_val, val): all_good = False; break
                if all_good:
                    proof[var] = val
                    found = True
                    break
            if not found:
                print(f"Warning: Could not satisfy auxiliary variable {var}!")

    return proof

def print_stats(label, G, assignment=None):
    n = len(G.variables)
    degrees = _compute_degrees(G).values()
    d = max(degrees) if degrees else 0
    alphabet = max(G.variables.values()) if G.variables else 0
    
    gap = "N/A"
    if assignment:
        acc = evaluate_acceptance_probability(G, assignment)
        gap = f"{1.0 - acc:.4f}"
        
    print(f"\n[STATS] {label}")
    print(f"   Variables (n): {n}")
    print(f"   Max Degree (d): {d}")
    print(f"   Alphabet (W): {alphabet}")
    print(f"   Unsat Gap: {gap}")
    print("-" * 30)

def main():
    print("=== STARTING FULL PCP PIPELINE TEST (T=1) ===")
    
    clauses = [
        [1, 2, -3],
        [-1, 2, 4],
        [2, -3, -4],
        [3, 5, -6],
        [-2, 4, 6]
    ]
    truth = {1: 1, 2: 1, 3: 0, 4: 0, 5: 1, 6: 1} # Perfectly Satisfying 
    
    print("\n--- Step 1: Arity Reduction ---")
    G1 = arity_reduction(clauses)
    h1 = trace_arity_assignment(clauses, truth)
    print_stats("After Arity Reduction", G1, h1)
    
    print("\n--- Step 2: Degree Reduction ---")
    print_stats("Before Degree Reduction", G1, h1)
    G2 = degree_reduction(G1, use_expander_cloud=False)
    h2 = trace_degree_reduction(G2, h1)
    print_stats("After Degree Reduction", G2, h2)
    
    print("\n--- Step 3: Gap Amplification (t=1) ---")
    print_stats("Before Gap Amplification", G2, h2)
    t = 1
    G_pow = power_graph(G2, t=t)
    
    # Trace for powering t=1
    from amplifier import get_walks
    walks = get_walks(G2, t)
    W_orig = 7 # max(G2.variables.values())
    h_pow = {}
    for u in G2.variables:
        paths = walks[u]
        val_list = []
        for p in paths:
            # Each p is a tuple of t+1 nodes
            for node in p: val_list.append(h2[node])
        
        idx = 0
        for i, val in enumerate(val_list):
            idx += val * (W_orig ** (len(val_list)-1-i))
        h_pow[u] = idx
        
    print_stats("After Gap Amplification (t=1)", G_pow, h_pow)

    print("\n--- Step 4: Alphabet Reduction (Lazy) ---")
    print_stats("Before Alphabet Reduction", G_pow, h_pow)
    G_bool = alphabet_reduction(G_pow, num_blr_tests=10)
    
    k = math.ceil(math.log2(max(G_pow.variables.values())))
    print(f"Building Honest Proof (Lazy Prover, k={k})...")
    proof = build_honest_proof(G_bool, G_pow, h_pow, k)
    print_stats("After Alphabet Reduction", G_bool, proof)
    
    if evaluate_acceptance_probability(G_bool, proof) == 1.0:
        print("\nSUCCESS: Pipeline preserved completeness for t=1!")
    else:
        print("\nFAILURE: Completeness check failed for t=1.")

if __name__ == "__main__":
    main()
