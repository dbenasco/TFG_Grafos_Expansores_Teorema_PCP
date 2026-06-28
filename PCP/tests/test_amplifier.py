"""
Test for Phase 2: Gap Amplification (The Power Graph).

Verifies that:
1. A satisfying assignment to G can be lifted to a satisfying assignment for G^t.
2. The amplified graph G^t preserves val = 1.0 for satisfiable instances.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraint_graph import ConstraintGraph
from nice_transformation import arity_reduction, degree_reduction, add_expander_edges
from amplifier import power_graph, OpinionCloud, get_walks


def test_amplifier_preserves_satisfiability():
    """Satisfiable 3SAT -> Nice Instance G -> Power Graph G^t -> val = 1.0"""
    print("=" * 60)
    print("Test 1: Amplifier preserves satisfiability (val = 1.0)")
    print("=" * 60)
    
    # 1. Create a simple satisfiable formula
    # (x1 or x2 or x3)
    clauses = [[1, 2, 3]]
    
    # Satisfying assignment for original variables
    # We will just pick x1=1, x2=1, x3=1
    true_assignment = {
        ("x", 1): 1,
        ("x", 2): 1,
        ("x", 3): 1
    }
    
    # 2. Arity Reduction
    G1 = arity_reduction(clauses)
    
    # To lift the assignment to G1, we need to assign the auxiliary variable W_0
    from nice_transformation import _satisfying_assignments
    sat_assignments = _satisfying_assignments(clauses[0])
    # The variable values corresponding to our true_assignment are (1, 1, 1)
    target_vals = (1, 1, 1)
    w_index = sat_assignments.index(target_vals)
    
    assign_G1 = dict(true_assignment)
    assign_G1[("w", 0)] = w_index
    assert G1.verify_assignment(assign_G1) == 1.0, "G1 assignment failed!"
    print("  ✓ G1 (Arity Reduction) is Satisfied")
    
    # 3. Degree Reduction
    G2 = degree_reduction(G1)
    
    # Lift assignment to G2 (Cycle of Copies)
    assign_G2 = {}
    for var_id in G2.variables:
        orig_id = var_id[0] # var_id is e.g. (("x", 1), copy_idx) or ("__pad__", idx)
        if orig_id in assign_G1:
            assign_G2[var_id] = assign_G1[orig_id]
        else:
            # Padding variable, any assignment is fine since its constraints are Null
            assign_G2[var_id] = 0
            
    assert G2.verify_assignment(assign_G2) == 1.0, "G2 assignment failed!"
    print("  ✓ G2 (Degree Reduction) is Satisfied")
    
    # 4. Expansion
    G_nice = add_expander_edges(G2, c_exp=3)
    
    # The variables are exactly the same, and the new edges are Null constraints.
    # So the same assignment works.
    assign_nice = dict(assign_G2)
    assert G_nice.verify_assignment(assign_nice) == 1.0, "G_nice assignment failed!"
    print(f"  ✓ G_nice (Expansion) is Satisfied. Graph has {G_nice.num_variables} nodes and {G_nice.num_constraints} edges.")

    # 5. Amplification (Power Graph)
    # We use t=1 so the test runs quickly.
    t = 1
    print(f"\n  Running power_graph with t={t}...")
    G_pow = power_graph(G_nice, t=t)
    print(f"  G_pow created: {G_pow.num_variables} nodes, {G_pow.num_constraints} edges.")
    
    # 6. Lift the assignment to G_pow
    # For every node u in G_pow, its assignment is an OpinionCloud.
    # The OpinionCloud must report the TRUE underlying values for every walk
    # extending from u.
    walks = get_walks(G_nice, t=t)
    
    assign_pow = {}
    for u in G_pow.variables:
        u_walks = walks[u]
        walk_values = {}
        for w in u_walks:
            # Extract the true value for every node in this walk
            vals = tuple(assign_nice[node] for node in w)
            walk_values[w] = vals
            
        assign_pow[u] = OpinionCloud(node_id=u, walk_values=walk_values)
        
    # 7. Verify the Power Graph!
    val_pow = G_pow.verify_assignment(assign_pow)
    print(f"\n  G_pow.verify_assignment() returned: {val_pow}")
    
    assert val_pow == 1.0, f"Expected 1.0, got {val_pow}"
    print("  ✓ SUCCESS: Satisfiability is perfectly preserved in the Powered Graph!")


if __name__ == "__main__":
    test_amplifier_preserves_satisfiability()
