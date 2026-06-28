import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nice_transformation import arity_reduction, degree_reduction, add_expander_edges
from test_expansion import build_large_3sat

def test_self_loops():
    print("Generating random 3SAT formula (20 vars, 50 clauses)...")
    clauses = build_large_3sat(20, 50)
    
    print("\nRunning PCP Nice Transformation Pipeline...")
    G1 = arity_reduction(clauses)
    G2 = degree_reduction(G1)
    G3 = add_expander_edges(G2, c_exp=3)
    
    print(f"Final Graph constructed: {G3}")
    
    # Check degree and self-loops for each node
    n = len(G3.variables)
    
    # Count degree and self-loops
    degrees = {v: 0 for v in G3.variables}
    self_loops = {v: 0 for v in G3.variables}
    
    for u, v, _ in G3.constraints:
        if u == v:
            degrees[u] += 1
            self_loops[u] += 1
        else:
            degrees[u] += 1
            degrees[v] += 1
            
    # Verify the property
    all_valid = True
    min_fraction = 1.0
    
    for v in G3.variables:
        d = degrees[v]
        sl = self_loops[v]
        fraction = sl / d if d > 0 else 0
        min_fraction = min(min_fraction, fraction)
        
        if fraction < 0.5:
            print(f"FAILED on node {v}: degree={d}, self-loops={sl} (fraction: {fraction:.2f})")
            all_valid = False
            break
            
    if all_valid:
        print(f"\nSUCCESS! All {n} vertices have >= 50% self loops.")
        print(f"The minimum self-loop fraction across any vertex is: {min_fraction:.4f}")
        
if __name__ == "__main__":
    test_self_loops()
