import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraint_graph import ConstraintGraph
from alphabet_reduction import alphabet_reduction, walsh_hadamard_encode
from nice_transformation import arity_reduction, degree_reduction, add_expander_edges, _satisfying_assignments
from plot_degree_reduction_gap import get_second_eigenvalue
from collections import defaultdict

def evaluate_acceptance_probability(G, assignment, debug=False):
    """
    Evaluates what percentage of the constraints in G are satisfied
    under the given assignment. (Matches amplifier.py calculation)
    """
    if not G.constraints:
        return 1.0
        
    satisfied = 0
    total = len(G.constraints)
    
    for u, v, check_fn in G.constraints:
        if check_fn(assignment.get(u), assignment.get(v)):
            satisfied += 1
        elif debug:
            print(f"[DEBUG] FAILED Constraint: u={u} (val={assignment.get(u)}), v={v} (val={assignment.get(v)})")
            
    return satisfied / total

def trace_prover_assignment(clauses, truth_assignment):
    """
    Automagically translates a base 3SAT truth assignment (whether 
    Honest or Fraudulent) completely through the 3 layers of the 
    'Nice Transformation' pipeline to produce the prover's macro graph assignment.
    """
    # 1. Arity Reduction Layer
    assign1 = {}
    for i, clause in enumerate(clauses):
        var_vals = (truth_assignment.get(abs(clause[0]), 0), 
                    truth_assignment.get(abs(clause[1]), 0), 
                    truth_assignment.get(abs(clause[2]), 0))
        sat_list = _satisfying_assignments(clause)
        if var_vals in sat_list:
            w_val = sat_list.index(var_vals)
        else:
            # CHEATING PROVER: Pick the closest valid assignment to trick the verifier
            def distance(s_val):
                return sum(1 for a, b in zip(s_val, var_vals) if a != b)
            w_val = min(range(len(sat_list)), key=lambda idx: distance(sat_list[idx]))
        
        assign1[("w", i)] = w_val
        for lit in clause:
            assign1[("x", abs(lit))] = truth_assignment.get(abs(lit), 0)
            
    # 2. Degree Reduction Layer (Copies variables into clouds)
    def get_assign_from_graph(G, base_assignment):
        assign = {}
        for v in G.variables:
            if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], int) and isinstance(v[0], tuple) and v[0][0] in ["x", "w"]:
                # Variable was degree-reduced. Its origin is v[0].
                origin = v[0]
                assign[v] = base_assignment.get(origin, 0)
            else:
                assign[v] = base_assignment.get(v, 0) # Fallback to 0 for __pad__ nodes
        return assign

    return assign1, get_assign_from_graph

def test_completeness():
    print("==================================================")
    print(" Alphabet Reduction: Completeness Proof (HONEST)")
    print("==================================================\n")
    
    # 1. Define a realistic 3SAT formula and its known honest satisfying assignment
    clauses = [
        [1, 2, -3],
        [-1, 2, 4],
        [2, -3, -4]
    ]
    truth_assignment = {1: 1, 2: 1, 3: 0, 4: 0} # Perfectly Satisfying
    
    print("1. Running 'Nice Transformation' pipeline on valid 3SAT formula...")
    G1 = arity_reduction(clauses)
    G2 = degree_reduction(G1, use_expander_cloud=False)
    G_nice = add_expander_edges(G2, c_exp=3)
    
    # Trace the assignment through the graph layers
    assign1, mapper = trace_prover_assignment(clauses, truth_assignment)
    honest_g_nice = mapper(G_nice, assign1)
    
    # Mathematical sanity check!
    prob_nice = evaluate_acceptance_probability(G_nice, honest_g_nice)
    assert prob_nice == 1.0, "FATAL: Nice Transformation broke the satisfying assignment!"
    
    n_vars_orig = len(G_nice.variables)
    max_dom_orig = max(dom for _, dom in G_nice.variables.items())
    l2_orig = get_second_eigenvalue(G_nice)
    print(f"--- Properties of the 'Nice' Graph (n={n_vars_orig}) ---")
    print(f"Spectral Gap (1 - λ2): {1.0 - l2_orig:.4f}  (λ2 = {l2_orig:.4f})")
    print("---------------------------------------------------------------")
    
    # 2. Run Alphabet Reduction to generate the Boolean graph
    num_tests = 5
    print(f"\n2. Shattering into a Boolean Graph (with coin-flip limits {num_tests})...")
    G_bool = alphabet_reduction(G_nice, num_blr_tests=num_tests)
    n_vars = len(G_bool.variables)
    print(f"--- Properties of the Alphabet-Reduced Boolean Graph ---")
    print(f"Size of graph (n): {n_vars} variables")
    print("--------------------------------------------------------")
    
    k = math.ceil(math.log2(max_dom_orig)) 
    if k == 0: k = 1
    WH_LEN = 1 << k
    PI_LEN = 1 << (2 * k)
    
    # 3. Construct the Honest Prover Proof dynamically for ALL variables!
    print("\n3. Honest Prover is dynamically generating Walsh-Hadamard proofs...")
    honest_proof = {}
    
    for u in G_nice.variables:
        wh_enc = walsh_hadamard_encode(honest_g_nice[u], k)
        for x in range(WH_LEN):
            honest_proof[("WH", u, x)] = wh_enc[x]
            
    for edge_idx, (u, v, check_fn) in enumerate(G_nice.constraints):
        e_id = f"e_{edge_idx}"
        a_val = honest_g_nice[u]
        b_val = honest_g_nice[v]
        
        # Construct PI encodings
        w_val = (a_val << k) | b_val
        pi_enc = walsh_hadamard_encode(w_val, 2 * k)
        for z in range(PI_LEN):
            honest_proof[("PI", e_id, z)] = pi_enc[z]
            
        # Construct the W_EVAL exact commitment
        sat_assignments = []
        for a in range(max_dom_orig):
            for b in range(max_dom_orig):
                if check_fn(a, b):
                    sat_assignments.append((a, b))
        
        w_idx = sat_assignments.index((a_val, b_val))
        honest_proof[("W_EVAL", e_id)] = w_idx
        
    print("   Resolving structural auxiliary variables...")
    edges_by_var = defaultdict(list)
    for u, v, check_fn in G_bool.constraints:
        edges_by_var[u].append((v, check_fn, True))
        edges_by_var[v].append((u, check_fn, False))
        
    for var_id in G_bool.variables:
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

    # 4. Verify Acceptance Probability
    prob = evaluate_acceptance_probability(G_bool, honest_proof, debug=True)
    print(f"Final Acceptance Probability of Honest Prover: \033[92m{prob * 100:.2f}%\033[0m")
    assert prob == 1.0


def test_soundness():
    print("\n\n==================================================")
    print(" Alphabet Reduction: Soundness Proof (CHEATING)")
    print("==================================================\n")
    
    clauses = [
        [1, 2, -3],
        [-1, 2, 4],
        [2, -3, -4]
    ]
    
    # PROVER SUBMITS AN INVALID ASSIGNMENT!
    fraud_assignment = {1: 0, 2: 0, 3: 1, 4: 1} # Completely wrong assignment
    
    print("1. Cheating Prover forces an INVALID assignment into the Nice transformation...")
    G1 = arity_reduction(clauses)
    G2 = degree_reduction(G1, use_expander_cloud=False)
    G_nice = add_expander_edges(G2, c_exp=3)
    
    assign1, mapper = trace_prover_assignment(clauses, fraud_assignment)
    fraud_g_nice = mapper(G_nice, assign1)
    
    prob_nice = evaluate_acceptance_probability(G_nice, fraud_g_nice)
    print(f"   Cheating Prover passes {prob_nice * 100:.2f}% of the G_nice constraints.")
    
    max_dom_orig = max(dom for _, dom in G_nice.variables.items())
    
    num_tests = 5
    print(f"\n2. Shattering to Boolean Graph (coin-flip limits {num_tests})...")
    G_bool = alphabet_reduction(G_nice, num_blr_tests=num_tests)
    n_vars = len(G_bool.variables)
    
    k = math.ceil(math.log2(max_dom_orig)) 
    if k == 0: k = 1
    WH_LEN = 1 << k
    PI_LEN = 1 << (2 * k)
    
    print("\n3. Cheating Prover attempts to mathematically fake the Walsh-Hadamard proofs...")
    fraud_proof = {}
    
    for u in G_nice.variables:
        wh_enc = walsh_hadamard_encode(fraud_g_nice[u], k)
        for x in range(WH_LEN):
            fraud_proof[("WH", u, x)] = wh_enc[x]
            
    for edge_idx, (u, v, check_fn) in enumerate(G_nice.constraints):
        e_id = f"e_{edge_idx}"
        a_val = fraud_g_nice[u]
        b_val = fraud_g_nice[v]
        
        # Even if the edge fails the check recursively, the prover blindly encodes it
        # into the PI cheatsheet hoping to pass the Linearity and Consistency tests!
        w_val = (a_val << k) | b_val
        pi_enc = walsh_hadamard_encode(w_val, 2 * k)
        for z in range(PI_LEN):
            fraud_proof[("PI", e_id, z)] = pi_enc[z]
            
        # Try to explicitly COMMIT to a valid assignment to pass W_EVAL
        sat_assignments = []
        for a in range(max_dom_orig):
            for b in range(max_dom_orig):
                if check_fn(a, b):
                    sat_assignments.append((a, b))
                    
        if (a_val, b_val) in sat_assignments:
            w_idx = sat_assignments.index((a_val, b_val))
        else:
            w_idx = 0 # The prover's true assignment fundamentally DOES NOT EXIST in the commitment list. They fake it.
            
        fraud_proof[("W_EVAL", e_id)] = w_idx
            
    print("   Cheating Prover mathematically resolving Auxiliary XOR variables best-fit...")
    edges_by_var = defaultdict(list)
    for u, v, check_fn in G_bool.constraints:
        edges_by_var[u].append((v, check_fn, True))
        edges_by_var[v].append((u, check_fn, False))
        
    for var_id in G_bool.variables:
        if isinstance(var_id, tuple) and var_id[0].startswith("W_") and var_id[0] != "W_EVAL":
            best_val = 0
            best_satisfied = -1
            
            for w in range(4):
                sat_count = 0
                for neighbor, check_fn, is_u in edges_by_var[var_id]:
                    if neighbor in fraud_proof:
                        if is_u and check_fn(w, fraud_proof[neighbor]):
                            sat_count += 1
                        elif (not is_u) and check_fn(fraud_proof[neighbor], w):
                            sat_count += 1
                            
                if sat_count > best_satisfied:
                    best_satisfied = sat_count
                    best_val = w
                    
            fraud_proof[var_id] = best_val

    print("\n4. Simulating the Verifier reading the Fraudulent Proof...")
    prob = evaluate_acceptance_probability(G_bool, fraud_proof)
    print(f"Final Acceptance Probability of Cheating Prover: \033[91m{prob * 100:.2f}%\033[0m")
    
    if prob < 1.0:
        print("\n[SUCCESS] The Alphabet Reduction successfully CAUGHT the Cheating Prover!")
        print("          The commitment constraints explicitly forced the invalid logic")
        print("          to cascade and uniquely trigger the boolean boundary verification checks!")
    else:
        print("\n[FAILURE] Soundness is broken! The Cheating Prover achieved 100% acceptance.")

if __name__ == "__main__":
    test_completeness()
    test_soundness()
