from nice_transformation import arity_reduction, degree_reduction, add_expander_edges
from amplifier import power_graph
from alphabet_reduction import alphabet_reduction
from constraint_graph import ConstraintGraph

def pcpify(G, iterations=1, t_power=1, c_exp=3, num_blr_tests=5):
    """
    Executes the full Dinur PCP transformation pipeline iteratively.
    Assuming G is already a formulated 2CSP (e.g. from arity_reduction).
    """
    print(f"--- Starting PCPfication Pipeline ({iterations} iterations) ---")
    
    for i in range(iterations):
        print(f"\n--- Iteration {i+1}/{iterations} ---")
        
        print(" Step A: Degree Reduction")
        G = degree_reduction(G, use_expander_cloud=False)
        
        print(f" Step B: Expander Overlay (c_exp={c_exp})")
        G = add_expander_edges(G, c_exp=c_exp)
        max_dom = max(dom for _, dom in G.variables.items())
        print(f"  -> 'Nice' Graph has {len(G.variables)} variables, max domain {max_dom}")
        
        if t_power > 0:
            print(f" Step C: Gap Amplification (Powering t={t_power})")
            G = power_graph(G, t=t_power)
            if len(G.variables) > 0:
                max_dom = max(dom for _, dom in G.variables.items())
            else:
                max_dom = 0
            print(f"  -> Amplified Graph has {len(G.variables)} variables, max domain {max_dom}")
        
        print(" Step D: Alphabet Reduction (Composition)")
        G = alphabet_reduction(G, num_blr_tests=num_blr_tests)
        max_dom = max(dom for _, dom in G.variables.items())
        print(f"  -> Final Boolean Graph has {len(G.variables)} variables, max domain {max_dom}")
        
    print("\n--- PCPfication Complete! ---")
    return G
