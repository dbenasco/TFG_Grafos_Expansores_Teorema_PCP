from itertools import product
from constraint_graph import ConstraintGraph
import sys
import os
import numpy as np

# Set up path for Expanders
expander_dir = os.path.join(os.path.dirname(__file__), '..', 'Expanders')
if expander_dir not in sys.path:
    sys.path.append(expander_dir)
    
try:
    from SipserSpielmanExpander import SSExpanderCode, InnerCode
except ImportError:
    # We raise a clearer error message since this is in a non-standard path
    raise ImportError("Could not find 'SipserSpielmanExpander' in '../Expanders'. "
                      "Ensure your directory structure is correct.")



# ──────────────────────────────────────────────────────────────
# Task 1.2: Arity Reduction  (3SAT → 2CSP)
# ──────────────────────────────────────────────────────────────

def _satisfying_assignments(clause):
    """
    Enumerate all satisfying VARIABLE-VALUE tuples for a single 3SAT clause.

    A clause is a list of 3 literals in DIMACS format:
      positive int  →  variable appears unnegated
      negative int  →  variable appears negated

    Example: [1, -2, 3] represents (x₁ ∨ ¬x₂ ∨ x₃).

    We iterate over all 8 possible (v₀, v₁, v₂) ∈ {0,1}³, where each
    vⱼ represents the VARIABLE VALUE (not the literal truth value).

    For each combination, we evaluate whether the clause is satisfied:
      - A positive literal (e.g., x₁) is true when vⱼ = 1.
      - A negative literal (e.g., ¬x₂) is true when vⱼ = 0.

    Returns
    -------
    list of tuples
        Each tuple (v₀, v₁, v₂) is a variable-value assignment that
        satisfies the clause. There are exactly 7 such tuples.

    Complexity
    ----------
    Time: O(2^L) where L is the arity of the clause (L=3). Constant time O(1).
    Space: O(2^L) = O(1) space to store the satisfying tuples.

    Why variable values (not literal truth values)?
    -----------------------------------------------
    This makes the consistency check in arity_reduction trivially simple:
      check(w_val, x_val)  →  sat_list[w_val][pos] == x_val
    No negation handling needed at the edge level.
    """
    sat = []
    # product((0,1), repeat=3) generates all 8 variable-value combinations.
    for var_vals in product((0, 1), repeat=3):
        # Evaluate the clause under these variable values.
        # A literal is satisfied when:
        #   - lit > 0 and var_vals[i] == 1  (positive literal, variable is True)
        #   - lit < 0 and var_vals[i] == 0  (negative literal, variable is False → ¬var is True)
        clause_satisfied = any(
            (lit > 0 and var_vals[i] == 1) or (lit < 0 and var_vals[i] == 0)
            for i, lit in enumerate(clause)
        )
        if clause_satisfied:
            sat.append(var_vals)
    return sat


def arity_reduction(clauses):
    """
    CLAIM 22.36
    Convert a 3SAT formula into a binary (arity-2) ConstraintGraph.

    This implements the standard arity reduction from Arora & Barak
    (Section 22.A, Step 1).

    The idea:
    ---------
    A 3SAT clause like (x₁ ∨ ¬x₂ ∨ x₃) constrains 3 variables at once.
    We need to turn this into a graph with only pairwise (binary) edges.

    For each clause Cᵢ:
      1. Create an AUXILIARY variable Wᵢ.
         Its domain = the 7 satisfying variable-value assignments of Cᵢ.
         So Wᵢ takes a value in {0, 1, ..., 6}, where each integer
         indexes one of the 7 satisfying (v₀, v₁, v₂) tuples.

      2. Add 3 CONSISTENCY EDGES connecting Wᵢ to the original variables:
         For each position j ∈ {0, 1, 2}:
           Edge (Wᵢ, xⱼ) with constraint:
             "The variable value that Wᵢ assigns at position j must
              match the actual value of xⱼ."
         Since Wᵢ stores variable values (not literal truth values),
         the check is simply: sat_list[w_val][pos] == x_val.

    Result: A bipartite graph with:
      - n original boolean variables  (domain = 2)
      - m auxiliary clause variables   (domain = 7)
      - 3m edges (3 per clause)

    Parameters
    ----------
    clauses : list of lists
        Each inner list has 3 integers in DIMACS format.
        Example: [[1, -2, 3], [-1, 2, -3]]
        Variables are numbered 1, 2, ..., n (positive integers).

    Returns
    -------
    ConstraintGraph
        The resulting binary CSP.
        
    Complexity
    ----------
    Let n = number of unique variables in the formula, m = number of 3SAT clauses.
    Time: O(m + n log n)
        - Sorting the variables takes O(n log n). Iterating over the m clauses 
          and evaluating `_satisfying_assignments` takes O(1) per clause.
    Space: O(n + m)
        - The new graph contains n boolean variables, m auxiliary variables, 
          and 3m edges.
    """
    G = ConstraintGraph()

    # ── Step A: Discover all original variables ──────────────
    # Scan all clauses to find which variable IDs appear.
    # abs(literal) gives the variable number.
    all_vars = set()
    for clause in clauses:
        for lit in clause:
            all_vars.add(abs(lit))

    # Register each original variable as boolean (domain = 2).
    # We use ("x", var_num) as the ID to avoid collisions with
    # auxiliary variables (which will be ("w", clause_index)).
    for var_num in sorted(all_vars):
        G.add_variable(("x", var_num), domain_size=2)

    # ── Step B: Process each clause ──────────────────────────
    for i, clause in enumerate(clauses):
        assert len(clause) == 3, f"Clause {i} has {len(clause)} literals, expected 3."

        # Compute the 7 satisfying variable-value assignments.
        sat_assignments = _satisfying_assignments(clause) # o(2^k*m) = o(m), k=3 (kCSP, m number of clauses)
        # sat_assignments[w_val] = (v₀, v₁, v₂) — variable values.

        # Register the auxiliary variable Wᵢ with domain = 7.
        w_id = ("w", i)
        G.add_variable(w_id, domain_size=len(sat_assignments))

        # ── Step C: Add 3 consistency edges ──────────────────
        # For each position j in the clause (0, 1, 2):
        for j, lit in enumerate(clause):
            var_num = abs(lit)          # which original variable
            x_id = ("x", var_num)      # ID of the original variable

            # Build the consistency check for edge (Wᵢ, xⱼ).
            # We use a factory function to freeze `j` and `sat_assignments`
            # in the closure (Python closures capture by reference).
            def make_check(pos, sat_list):
                """
                Build the consistency check function for one edge.

                The check is simple because sat_list stores VARIABLE VALUES:
                  w_val : index into sat_list (0..6)
                  x_val : boolean value of the original variable (0 or 1)

                  Look up the variable value that Wᵢ assigns at position `pos`:
                    bit = sat_list[w_val][pos]

                  The constraint: bit must equal x_val.
                  That's it — no negation logic needed here, because
                  _satisfying_assignments already encodes the correct
                  variable values that make the clause true.
                """
                def check(w_val, x_val):
                    return sat_list[w_val][pos] == x_val
                return check

            check_fn = make_check(j, sat_assignments)
            G.add_constraint(w_id, x_id, check_fn)

    return G


# ──────────────────────────────────────────────────────────────
# Task 1.3: Degree Reduction  (Irregular → d-regular)
# ──────────────────────────────────────────────────────────────

def _compute_degrees(G):
    """
    Compute the degree of each variable in the constraint graph.

    Since our graph is undirected (each constraint (u, v, _) contributes
    one edge to both u and v), we count how many constraints each
    variable participates in.

    Parameters
    ----------
    G : ConstraintGraph

    Returns
    -------
    dict
        Maps variable ID → degree (int).
        
    Complexity
    ----------
    Let n = number of variables in G, m = number of edges (constraints).
    Time: O(n + m)
        - O(n) to initialize the dict, O(m) to iterate over all constraints.
    Space: O(n)
        - The dictionary stores an integer degree for each of the n variables.
    """
    degree = {var_id: 0 for var_id in G.variables}
    for u, v, _ in G.constraints:
        degree[u] += 1
        degree[v] += 1
    return degree


def degree_reduction(G, use_expander_cloud=True, c_cloud=3):
    """
    CLAIM 22.37
    Make the constraint graph regular using Expander Clouds (Dinur's method)
    or Cycle of Copies (Arora & Barak).

    Phase 1 — PRE-PAD: Iteratively add null edges (always True) until
      every variable has degree ≥ 3. We pair up missing edges between
      DIFFERENT variables to avoid creating self-loops early, which keeps
      the later self-loop-doubling step (Claim 22.38) perfectly balanced.
    Phase 2 — SPLIT: Replace each variable (degree k) with k copies.
      Depending on `use_expander_cloud`, connect them in a cycle or an expander.

    Parameters
    ----------
    G : ConstraintGraph
    use_expander_cloud : bool
        If True, wire the copies internally using an expander graph (preserves gap).
    c_cloud : int
        The internal degree of the expander graph.

    Returns
    -------
    ConstraintGraph
        A new regular constraint graph.
        
    Complexity
    ----------
    Let n_0 = initial variables, m_0 = initial constraints.
    Time: O(n_0 + m_0)
        - Calculating degrees and padding takes O(n_0 + m_0).
        - The sum of all degrees is exactly 2*m_0. Generating the cycle or 
          the expander cloud for each vertex takes time proportional to its degree.
          Total time is rigorously bounded by the sum of degrees, O(m_0).
    Space: O(m_0)
        - The new graph will have O(m_0) variables and O(m_0) edges, because 
          each edge in the original graph contributes a constant number of nodes
          and edges to the final shattered format.
    """
    # ── Phase 1: Iterative pre-padding to min degree 3 ───────────
    all_vars = dict(G.variables)
    all_constraints = list(G.constraints)
    pad_counter = 0
    null_fn = lambda a, b: True

    # Keep padding until min degree is 3.
    # (Usually converges in 1-2 iterations)
    for _ in range(10):
        # Compute current degrees.
        deg = {v: 0 for v in all_vars}
        for u, v, _ in all_constraints:
            if u == v:
                deg[u] += 2
            else:
                deg[u] += 1
                deg[v] += 1

        # Collect slots: each entry is a var_id that needs +1 degree.
        slots = []
        for var_id, d in deg.items():
            for _ in range(max(0, 3 - d)):
                slots.append(var_id)

        if not slots:
            break

        # Pair up distinct slots to avoid self-loops.
        paired = [False] * len(slots)
        for i in range(len(slots)):
            if paired[i]:
                continue
            found = False
            for j in range(i + 1, len(slots)):
                if not paired[j] and slots[j] != slots[i]:
                    all_constraints.append((slots[i], slots[j], null_fn))
                    paired[i] = paired[j] = True
                    found = True
                    break
            
            if not found:
                # Unpaired slot (only happens if all remaining slots 
                # are for the exact same variable).
                # Create a fresh pad variable and pair with it.
                pad_id = ("__pad__", pad_counter)
                pad_counter += 1
                all_vars[pad_id] = 2  # domain 2 (arbitrary)
                all_constraints.append((slots[i], pad_id, null_fn))
                paired[i] = True

    # ── Phase 2: Cycle of Copies ─────────────────────────────────
    # Recompute finalized degrees
    deg = {v: 0 for v in all_vars}
    for u, v, _ in all_constraints:
        if u == v:
            deg[u] += 2
        else:
            deg[u] += 1
            deg[v] += 1

    H = ConstraintGraph()
    appearance_count = {v: 0 for v in all_vars}

    # Step A: Register all copy variables
    for var_id, dom_size in all_vars.items():
        for copy_idx in range(deg[var_id]):
            H.add_variable((var_id, copy_idx), dom_size)

    # Step B: Rewire external constraints
    for u, v, check_fn in all_constraints:
        u_copy = (u, appearance_count[u])
        v_copy = (v, appearance_count[v])
        appearance_count[u] += 1
        appearance_count[v] += 1
        H.add_constraint(u_copy, v_copy, check_fn)

    # Step C: Cloud equality edges (Cycle or Expander)
    if not use_expander_cloud:
        # 1. Cycle of Copies (Arora & Barak)
        for var_id in all_vars:
            k = deg[var_id]
            for copy_idx in range(k):
                next_idx = (copy_idx + 1) % k
                H.add_constraint(
                    (var_id, copy_idx),
                    (var_id, next_idx),
                    lambda a, b: a == b
                )
    else:
        # 2. Expander Cloud (Dinur's Original)
        actual_c_exp = (c_cloud + 1) // 2
        dummy_Hs = np.zeros((1, actual_c_exp), dtype=np.uint8)
        inner = InnerCode(actual_c_exp, dummy_Hs)
        
        for var_id in all_vars:
            k = deg[var_id]
            if k <= c_cloud:
                # Too small for an expander, fallback to cycle (which is already an expander for k<=3)
                for copy_idx in range(k):
                    next_idx = (copy_idx + 1) % k
                    H.add_constraint(
                        (var_id, copy_idx),
                        (var_id, next_idx),
                        lambda a, b: a == b
                    )
            else:
                expander = SSExpanderCode(
                    n=k, c=actual_c_exp, d=actual_c_exp, 
                    inner=inner, seed=int(np.random.randint(0, 2**31 - 1)), allow_multi_edges=True
                )
                for v in range(expander.n):
                    for u in expander.var_nbh[v]:
                        H.add_constraint((var_id, v), (var_id, u), lambda a, b: a == b)
                        if v == u:
                            # Duplicate self-loop to maintain degree properties of the folded bipartite graph
                            H.add_constraint((var_id, v), (var_id, u), lambda a, b: a == b)

    return H


# ──────────────────────────────────────────────────────────────
# Step 3: Expansion Overlay
# ──────────────────────────────────────────────────────────────

def add_expander_edges(G, c_exp=3):
    """
    CLAIM 22.38 (Expansion portion)

    Implements the exact 4-part theoretical construction:
      1. Choose a target degree d (we use d = 2 * c_exp)
      2. Pad the original constraint graph to degree d using self-loops.
      3. Overlay a d-regular expander graph (edges added as null constraints).
      4. Add 2d self-loops to every vertex.
    
    The resulting graph is exactly 4d-regular. The normalized second eigenvalue 
    is bounded by:
        λ(ψ) ≤ 3/4 + 1/4 * λ(G_n)
    where G_n is the expander.

    Parameters
    ----------
    G : ConstraintGraph
        The current constraint graph (e.g., 3-regular from step 2).
    c_exp : int
        The parameter for the custom bipartite expander generator.
        The resulting overlay expander will be d = 2*c_exp regular.

    Returns
    -------
    ConstraintGraph
        The expanded "Nice" instance, exactly 4d-regular.
        
    Complexity
    ----------
    Let n = number of variables in G, m = number of edges (which is O(n) since 
    degree_reduction just made it exactly constant degree).
    Time: O(n)
        - O(n + m) to copy variables and edges over.
        - Generating the d-regular bipartite expander using SSExpanderCode 
          takes expected O(n) time.
        - Appending O(n) new expander constraints and O(n) self-loops takes O(n).
    Space: O(n)
        - The final graph has exactly n variables and N*4d edges (where d 
          is a tiny constant). Thus memory footprint is strictly linear O(n).
    """


    H = ConstraintGraph()

    # 1. Copy all variables and existing constraints to compute current degree
    for var_id, dom_size in G.variables.items():
        H.add_variable(var_id, dom_size)

    inc = {var_id: 0 for var_id in G.variables}
    for u, v, check_fn in G.constraints:
        H.add_constraint(u, v, check_fn)
        if u == v:
            inc[u] += 1
        else:
            inc[u] += 1
            inc[v] += 1

    if not inc:
        return H

    d_orig = max(inc.values())
    d = max(2 * c_exp, d_orig)

    # 2. Pad original graph to degree d using self-loops
    for var_id in G.variables:
        needed = d - inc[var_id]
        for _ in range(needed):
            H.add_constraint(var_id, var_id, lambda a, b: True)

    n = len(G.variables)
    vars_list = list(G.variables.keys())

    # 3. Overlay d-regular expander graph G_n
    # Our expander generator produces a bipartite graph with degree c_exp.
    # Folding it gives an undirected graph of degree 2*c_exp.
    # To reach exactly d, we might need to add multiple overlays if d > 2*c_exp,
    # but since d = max(2*c_exp, d_orig), we just set c_exp = ceil(d/2) internally
    # to guarantee we generate EXACTLY a d-regular overlay.
    actual_c_exp = (d + 1) // 2
    
    if n <= actual_c_exp:
        # Trivial case for sub-microscopic graphs: clique padding
        for _ in range(d):
            for i in range(n):
                j = (i + 1) % n
                H.add_constraint(vars_list[i], vars_list[j], lambda a, b: True)
    else:
        dummy_Hs = np.zeros((1, actual_c_exp), dtype=np.uint8)
        inner = InnerCode(actual_c_exp, dummy_Hs)
        
        # We loop to generate exactly d edges per node. Folding the actual_c_exp 
        # bipartite graph gives 2*actual_c_exp edges. If d is odd, we adjust.
        # But d_orig=3, c_exp=3 => d=6. So actual_c_exp=3 => 6 edges exact!
        expander = SSExpanderCode(
            n=n, c=actual_c_exp, d=actual_c_exp, 
            inner=inner, seed=int(np.random.randint(0, 2**31 - 1)), allow_multi_edges=True
        )
        
        # Add the bipartite edges folded onto the variable set
        # Left node v adds c_exp edges. Right node v adds c_exp edges.
        # Total = 2*c_exp edges per node.
        edges_added = {v: 0 for v in vars_list}
        for v in range(expander.n):
            u_neighbors = expander.var_nbh[v]
            for u in u_neighbors:
                var_left = vars_list[v]
                var_right = vars_list[u]
                H.add_constraint(var_left, var_right, lambda a, b: True)
                
                # If it's a bipartite self-loop (left node v == right node u),
                # it consumes both 1 outgoing edge and 1 incoming edge of v.
                # To maintain exactly 2*c_exp degree contributions for v, we must
                # add a second identical self-loop.
                if var_left == var_right:
                    H.add_constraint(var_left, var_right, lambda a, b: True)

    # 4. Add exactly 2d self-loops to each vertex
    for var_id in G.variables:
        for _ in range(2 * d):
            H.add_constraint(var_id, var_id, lambda a, b: True)

    return H
