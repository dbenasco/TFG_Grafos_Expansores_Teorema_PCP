from constraint_graph import ConstraintGraph
import itertools

def get_walks(G, t):
    """
    Generate all walks of length t starting from each node in the constraint graph.
    
    Parameters
    ----------
    G : ConstraintGraph
        The regular constraint graph.
    t : int
        The length of the walks (number of edges).
        
    Returns
    -------
    dict
        Maps var_id -> list of paths.
        Each path is a tuple of var_ids (length t+1).
        
    Complexity
    ----------
    Time: O(n * d^t)
        - where n is the number of nodes (variables)
        - d is the degree of the graph (assumed constant, e.g. 9)
        - The BFS generates exactly d^t walks for each of the n nodes.
    Space: O(n * d^t * t)
        - We store n * d^t total walks.
        - Each walk is a tuple of length t+1.
    """
    adj = {v: [] for v in G.variables}
    
    # True and False indicate if the edge is reversed or not
    for u, v, check_fn in G.constraints:
        adj[u].append((v, check_fn, False))
        adj[v].append((u, check_fn, True))
    
    walks = {}
    for node in G.variables:
        paths = [[node]]
        for _ in range(t):
            new_paths = []
            for path in paths:
                last_node = path[-1]
                for nbr_info in adj[last_node]:
                    neighbor = nbr_info[0]
                    new_paths.append(path + [neighbor])
            paths = new_paths
        walks[node] = [tuple(p) for p in paths]
        
    return walks


class OpinionCloud:
    """
    Represents the assigned value for a variable in the Power Graph G^t.
    In the power graph, a node's "value" is an opinion about all walks of length t 
    originating from it.
    """
    def __init__(self, node_id, walk_values):
        """
        Parameters
        ----------
        node_id : hashable
            The ID of the central node u in the original graph.
        walk_values : dict
            Maps a walk (tuple of nodes) -> tuple of values assigned to those nodes.
            Example: if walk is (u, x, y), value might be (1, 0, 1).
        """
        self.node_id = node_id
        self.walk_values = walk_values
        
    def get_value(self, walk, node_index):
        """Get the value assigned to the node at node_index in the given walk."""
        return self.walk_values[walk][node_index]
        
    @classmethod
    def int_to_opinions(cls, val_int, paths, W):
        """
        Translates a flat scalar integer (val_int) back into an OpinionCloud dictionary.
        The integer maps exactly to the lexicographical base-W sequence of the combined
        paths' evaluations, explicitly tying Alphabet-W reductions to cloud assignments.
        """
        opinions = {}
        remaining = val_int
        total_digits = sum(len(p) for p in paths)
        val_list = []
        for _ in range(total_digits):
            val_list.append(remaining % W)
            remaining //= W
        val_list.reverse()
        
        idx = 0
        for path in paths:
            opinions[path] = tuple(val_list[idx : idx + len(path)])
            idx += len(path)
            
        return opinions


def build_edge_lookup(G):
    """
    Build a lookup for the constraints to easily check internal validity.
    Returns a dict mapping (u, v) -> list of check_fn.
    Since there can be multi-edges, it returns a list.
    
    Complexity
    ----------
    Time: O(E) = O(n * d)
        - We iterate exactly once over all E constraints in the graph.
        - For a d-regular graph with n nodes, E = (n * d) / 2.
    Space: O(E) = O(n * d)
        - The dictionary stores 2 entries (forward and backward) for each of the E constraints.
    """
    lookup = {}
    for u, v, check_fn in G.constraints:
        if (u, v) not in lookup:
            lookup[(u, v)] = []
        lookup[(u, v)].append((check_fn, False))
        
        if (v, u) not in lookup:
            lookup[(v, u)] = []
        lookup[(v, u)].append((check_fn, True))
    return lookup


def check_power_constraint(walk_u, walk_v, edge_lookup, paths_u, paths_v, W):
    """
    Builds the consistency constraint function for an edge in G^t.
    
    Parameters
    ----------
    walk_u : tuple
        Walk of length t from u to z.
    walk_v : tuple
        Walk of length t from v to z.
    edge_lookup : dict
        Lookup table for original graph constraints.
        
    Returns
    -------
    callable(cloud_u, cloud_v) -> bool
    
    Complexity
    ----------
    Time (Builder): O(1) time to return the closure.
    Space (Builder): O(1) space to return the closure.
    
    Time (When check_fn is evaluated): O(t)
        - Checks internal validity for walk_u (len t) and walk_v (len t), 
          which takes O(t) time assuming O(1) constraint edge lookups.
    Space (When check_fn is evaluated): O(1)
        - No extra memory allocation besides local variables during execution.
    """
    # z is the intersection node (end of both walks)
    z_u_idx = len(walk_u) - 1
    z_v_idx = len(walk_v) - 1
    
    def internal_validity(walk_values, walk):
        """Check if the cloud's assignment for the walk satisfies all original edges traversed."""
        vals = walk_values.get(walk)
        if vals is None:
            return False
            
        for i in range(len(walk) - 1):
            a, b = walk[i], walk[i+1]
            val_a, val_b = vals[i], vals[i+1]
            constraints = edge_lookup.get((a, b), [])
            for check_fn, is_reversed in constraints:
                if is_reversed:
                    if not check_fn(val_b, val_a): return False
                else:
                    if not check_fn(val_a, val_b): return False
        return True

    def check_fn(val_u, val_v):
        ops_u = OpinionCloud.int_to_opinions(val_u, paths_u, W)
        ops_v = OpinionCloud.int_to_opinions(val_v, paths_v, W)
        
        # 1. Internal Validity
        if not internal_validity(ops_u, walk_u):
            return False
        if not internal_validity(ops_v, walk_v):
            return False
            
        # 2. Consistency: agree on the value of z
        val_z_u = ops_u[walk_u][z_u_idx]
        val_z_v = ops_v[walk_v][z_v_idx]
        
        return val_z_u == val_z_v

    return check_fn


def power_graph(G, t):
    """
    Create the powered constraint graph G^t.
    
    WARNING: This function is O(n * d^(2t)). It will likely crash for t > 1 
    on high-degree graphs (like G_nice). It is kept here for educational 
    purposes to show the formal construction of the graph.
    
    For verifying the value (acceptance probability) efficiently, 
    use `calculate_acceptance_probability(G, t, assignment)` instead.
    
    Parameters
    ----------
    G : ConstraintGraph
        The original (Nice) constraint graph.
    t : int
        The powering parameter.
        
    Returns
    -------
    ConstraintGraph
        The powered graph G^t.
        
    Complexity
    ----------
    Time: O(n * d^(2t))
        - get_walks takes O(n * d^t)
        - We group n * d^t walks by their endpoint.
        - For each endpoint (n possible endpoints for a d-regular expander),
          there are roughly d^t walks arriving at it.
        - We output the Cartesian product of arriving walks for each endpoint:
          n * (d^t)^2 = n * d^(2t) constraints generated.
    Space: O(n * d^(2t))
        - get_walks stores O(n * d^t * t) walks.
        - The ending_at dictionary groups these walks taking O(n * d^t) space.
        - The returned G_pow graph contains exactly n variables and 
          n * d^(2t) constraint edges, each storing a reference to a custom function.
    """
    G_pow = ConstraintGraph()
    walks = get_walks(G, t)
    edge_lookup = build_edge_lookup(G)
    
    W_orig = max(dom for _, dom in G.variables.items()) if G.variables else 0
    
    # 1. Variables
    for var_id in G.variables:
        paths = walks[var_id]
        total_digits = sum(len(p) for p in paths)
        domain_size = W_orig ** total_digits
        G_pow.add_variable(var_id, domain_size=domain_size)
        
    # 2. Constraints (Edges of G^t)
    # Pre-group walks by their endpoint z.
    ending_at = {}
    for start_node, node_walks in walks.items():
        for walk in node_walks:
            z = walk[-1]
            if z not in ending_at:
                ending_at[z] = []
            ending_at[z].append((start_node, walk))
            
    # Now, for every node z, pair up all walks that end at z.
    for z, converging_walks in ending_at.items():
        for (u, walk_u) in converging_walks:
            for (v, walk_v) in converging_walks:
                check_fn = check_power_constraint(walk_u, walk_v, edge_lookup, walks[u], walks[v], W_orig)
                G_pow.add_constraint(u, v, check_fn)
                
    return G_pow

def calculate_acceptance_probability(G, t, assignment):
    """
    Efficiently calculate the acceptance probability of G^t using Matrix Power.
    Complexity: O(N^3 * log t)
    
    This function leverages the property that (A^t)_{ij} counts the number 
    of walks of length t between nodes i and j.
    """
    import numpy as np
    nodes = list(G.variables.keys())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    edge_lookup = build_edge_lookup(G)
    
    # 1. Adjacency Matrices
    adj_matrix = np.zeros((n, n), dtype=np.float64)
    valid_adj = np.zeros((n, n), dtype=np.float64)
    
    for u, v, _ in G.constraints:
        i, j = node_to_idx[u], node_to_idx[v]
        val_u, val_v = assignment[u], assignment[v]
        
        adj_matrix[i, j] += 1
        adj_matrix[j, i] += 1
        
        def is_valid(a, b, va, vb):
            constraints = edge_lookup.get((a, b), [])
            for cf, rev in constraints:
                if rev:
                    if not cf(vb, va): return False
                else:
                    if not cf(va, vb): return False
            return True

        if is_valid(u, v, val_u, val_v):
            valid_adj[i, j] += 1
        if is_valid(v, u, val_v, val_u):
            valid_adj[j, i] += 1
            
    # Normalize to prevent overflow at high t
    d_max = np.max(np.sum(adj_matrix, axis=0))
    if d_max > 0:
        adj_matrix /= d_max
        valid_adj /= d_max
    
    total_pow = np.linalg.matrix_power(adj_matrix, t)
    valid_pow = np.linalg.matrix_power(valid_adj, t)
    
    col_sums_valid = np.sum(valid_pow, axis=0)
    col_sums_total = np.sum(total_pow, axis=0)
    
    num = np.sum(col_sums_valid**2)
    den = np.sum(col_sums_total**2)
    
    return num / den if den > 0 else 0.0
