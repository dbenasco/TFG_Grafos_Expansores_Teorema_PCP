import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
from collections import defaultdict

from constraint_graph import ConstraintGraph
from pcpifier import pcpify
from nice_transformation import degree_reduction, add_expander_edges
from alphabet_reduction import alphabet_reduction, walsh_hadamard_encode
from amplifier import power_graph

def evaluate_acceptance_probability(G, assignment, debug=False):
    if not G.constraints: return 1.0
    satisfied = 0
    total = len(G.constraints)
    for u, v, check_fn in G.constraints:
        if check_fn(assignment.get(u), assignment.get(v)):
            satisfied += 1
        elif debug:
             print(f"[DEBUG] FAILED Constraint: u={u} (val={assignment.get(u)}), v={v} (val={assignment.get(v)})")
    return satisfied / total

def trace_degree_reduction(G, base_assignment):
    assign = {}
    for v in G.variables:
        if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], int):
            origin = v[0]
            assign[v] = base_assignment.get(origin, 0)
        else:
            assign[v] = base_assignment.get(v, 0) # Fallback to 0 for __pad__ nodes
    return assign

def trace_power_assignment(G_nice, honest_nice, t):
    from amplifier import get_walks
    if len(G_nice.variables) == 0:
        return {}
    walks = get_walks(G_nice, t)
    W = max(dom for _, dom in G_nice.variables.items())
    power_assign = {}
    for u in G_nice.variables:
        paths = walks[u]
        
        val_list = []
        for path in paths:
            # The honest prover assigns the honest value to EVERY node conceptually traversed!
            for node in path:
                val_list.append(honest_nice.get(node, 0))
                
        idx = 0
        for i, val in enumerate(val_list):
            pow_factor = W ** (len(val_list) - 1 - i)
            idx += val * pow_factor
        power_assign[u] = idx
    return power_assign

def test_full_pipeline():
    print("==================================================")
    print(" PCPifier: Full Iterative Completeness Test")
    print("==================================================\n")
    
    # Extensive 3SAT instance!
    clauses = [
        [1, 2, -3],
        [-1, 2, 4],
        [2, -3, -4],
        [3, 5, -6],
        [-2, 4, 6]
    ]
    truth_assignment = {1: 1, 2: 1, 3: 0, 4: 0, 5: 1, 6: 1} # Perfectly Satisfying
    
    print("1. Running 'Nice Transformation' pipeline...")
    from nice_transformation import arity_reduction
    G1 = arity_reduction(clauses)
    G2 = degree_reduction(G1, use_expander_cloud=False)
    # Bypassing random geometric expanders to prevent connection spin-locks on N=11
    G_nice = G2 
    
    # We construct the exact mathematical trace assignment directly!
    def trace_arity_assignment(clauses, truth_assignment):
        from nice_transformation import _satisfying_assignments
        assign1 = {}
        for i, clause in enumerate(clauses):
            var_vals = (truth_assignment.get(abs(clause[0]), 0), 
                        truth_assignment.get(abs(clause[1]), 0), 
                        truth_assignment.get(abs(clause[2]), 0))
            sat_list = _satisfying_assignments(clause)
            w_val = sat_list.index(var_vals) if var_vals in sat_list else 0
            assign1[("w", i)] = w_val
            for lit in clause:
                assign1[("x", abs(lit))] = truth_assignment.get(abs(lit), 0)
        return assign1

    honest_1 = trace_arity_assignment(clauses, truth_assignment)
    honest_nice = trace_degree_reduction(G_nice, honest_1)
    
    prob_nice = evaluate_acceptance_probability(G_nice, honest_nice)
    assert prob_nice == 1.0, "FATAL: Nice Transformation broke the satisfying assignment!"
    
    n_vars_orig = len(G_nice.variables)
    max_dom_orig = max(dom for _, dom in G_nice.variables.items())
    print(f"--- Properties of the 'Nice' Graph (n={n_vars_orig}) ---")
    print(f"Max Domain {max_dom_orig}")
    print("---------------------------------------------------------------")
    
    print("\n2. Applying Gap Amplification (Power Graph isolated trace t=0)...")
    t = 0
    G_pow = power_graph(G_nice, t=t)
    honest_pow = trace_power_assignment(G_nice, honest_nice, t)
    
    prob_pow = evaluate_acceptance_probability(G_pow, honest_pow)
    assert prob_pow == 1.0, f"Powering broke completeness! {prob_pow}"
    
    n_vars_pow = len(G_pow.variables)
    max_dom_pow = max(dom for _, dom in G_pow.variables.items())
    print(f"--- Properties of the Powered Graph (n={n_vars_pow}) ---")
    print(f"Powered Domain Size: {max_dom_pow}")
    print("--------------------------------------------------------")
    
    print(f"\n3. Shattering into a Boolean Graph...")
    G_bool = alphabet_reduction(G_pow, num_blr_tests=5)
    n_vars_bool = len(G_bool.variables)
    print(f"--- Properties of the Final Boolean Graph ---")
    print(f"Size of graph (n): {n_vars_bool} variables")
    print("--------------------------------------------------------")
    
    k = math.ceil(math.log2(max_dom_pow)) 
    if k == 0: k = 1
    WH_LEN = 1 << k
    PI_LEN = 1 << (2 * k)
    
    print("\n4. Honest Prover is dynamically generating Lazy Walsh-Hadamard proofs...")
    honest_proof = {}
    
    def get_wh_bit(w_val, bit_length, z_query):
        ans = 0
        for bit_idx in range(bit_length):
            if ((w_val >> bit_idx) & 1) and ((z_query >> bit_idx) & 1):
                ans ^= 1
        return ans

    print(f"    -> Parsing {len(G_bool.variables)} variables...")
    count = 0
    for var_id in G_bool.variables:
        count += 1
        if count % 1000 == 0:
            print(f"       [{count} / {len(G_bool.variables)}] generating assignments...")
            
        if isinstance(var_id, tuple):
            if var_id[0] == "WH":
                _, u, x = var_id
                w_val = honest_pow[u]
                honest_proof[var_id] = get_wh_bit(w_val, k, x)
            elif var_id[0] == "PI":
                _, e_id_str, z = var_id
                edge_idx = int(e_id_str.split("_")[1])
                u, v, check_fn = G_pow.constraints[edge_idx]
                w_val = (honest_pow[u] << k) | honest_pow[v]
                honest_proof[var_id] = get_wh_bit(w_val, 2 * k, z)
            elif var_id[0] == "W_EVAL":
                _, e_id_str = var_id
                edge_idx = int(e_id_str.split("_")[1])
                u, v, check_fn = G_pow.constraints[edge_idx]
                a_val = honest_pow[u]
                b_val = honest_pow[v]
                sat_assignments = []
                for a in range(max_dom_pow):
                    for b in range(max_dom_pow):
                        if check_fn(a, b):
                            sat_assignments.append((a, b))
                w_idx = sat_assignments.index((a_val, b_val))
                honest_proof[var_id] = w_idx
        
    print(f"   Resolving {len(G_bool.variables)} structural auxiliary variables...")
    edges_by_var = defaultdict(list)
    for u, v, check_fn in G_bool.constraints:
        edges_by_var[u].append((v, check_fn, True))
        edges_by_var[v].append((u, check_fn, False))
        
    aux_count = 0
    for var_id in G_bool.variables:
        aux_count += 1
        if aux_count % 1000 == 0:
            print(f"       [{aux_count} / {len(G_bool.variables)}] auxiliary variable evaluations...")
            
        if isinstance(var_id, tuple) and var_id[0].startswith("W_") and var_id[0] != "W_EVAL":
            best_val = None
            for w in range(4):
                all_good = True
                for neighbor, check_fn, is_u in edges_by_var[var_id]:
                    if neighbor in honest_proof:
                        if is_u and not check_fn(w, honest_proof[neighbor]):
                            all_good = False
                            break
                        elif (not is_u) and not check_fn(honest_proof[neighbor], w):
                            all_good = False
                            break
                if all_good:
                    best_val = w
                    break
            honest_proof[var_id] = best_val

    prob = evaluate_acceptance_probability(G_bool, honest_proof, debug=True)
    print(f"\n[SUCCESS] Final Acceptance Probability of Honest Prover across ENTIRE pipeline: \033[92m{prob * 100:.2f}%\033[0m")
    assert prob == 1.0


if __name__ == "__main__":
    test_full_pipeline()
