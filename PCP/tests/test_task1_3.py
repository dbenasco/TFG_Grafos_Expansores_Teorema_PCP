"""
Test for Task 1.3: Degree Reduction (Regularization).

Verifies that degree_reduction() produces a 3-regular graph
(counting self-loops as +1, not +2) and preserves satisfiability.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nice_transformation import (
    arity_reduction, degree_reduction,
    _compute_degrees, _satisfying_assignments,
)


def count_incident_edges(H):
    """
    Count incident edges per variable, treating self-loops as +1.

    In graph theory, a self-loop (u, u) contributes 2 to the degree.
    But in our CSP context, each constraint is one "edge" incident
    to a node, so self-loops count as 1 incident constraint.
    """
    counts = {v: 0 for v in H.variables}
    for u, v, _ in H.constraints:
        if u == v:
            counts[u] += 1   # self-loop: 1 incident edge
        else:
            counts[u] += 1
            counts[v] += 1
    return counts


def test_regularity():
    """After degree reduction, every node has 3 incident edges."""
    print("=" * 55)
    print("Test 1: 3 incident edges per node")
    print("=" * 55)

    clauses = [[1, 2, 3], [1, -2, 4]]
    G = arity_reduction(clauses)
    H = degree_reduction(G)

    print(f"  Before: {G}")
    print(f"  After:  {H}")

    inc = count_incident_edges(H)
    for var_id, count in inc.items():
        assert count == 3, f"Variable {var_id} has {count} incident edges, expected 3"

    print(f"  All {H.num_variables} nodes have 3 incident edges ✓")
    print()


def test_satisfying_through_pipeline():
    """Satisfying assignment → val = 1.0 after degree reduction."""
    print("=" * 55)
    print("Test 2: Satisfying assignment through pipeline")
    print("=" * 55)

    clauses = [[1, -2, 3]]
    G = arity_reduction(clauses)
    H = degree_reduction(G)

    sat = _satisfying_assignments([1, -2, 3])
    target = (1, 0, 1)
    w_val = sat.index(target)

    # Build assignment: all copies get the same value.
    assignment = {}
    for var_id in H.variables:
        orig_id = var_id[0]
        if orig_id == ("x", 1):
            assignment[var_id] = 1
        elif orig_id == ("x", 2):
            assignment[var_id] = 0
        elif orig_id == ("x", 3):
            assignment[var_id] = 1
        elif orig_id == ("w", 0):
            assignment[var_id] = w_val
        else:
            assignment[var_id] = 0

    val = H.verify_assignment(assignment)
    print(f"  val = {val}")
    assert val == 1.0, f"Expected 1.0, got {val}"
    print("  ✓ PASSED")
    print()


def test_inconsistent_copies():
    """If copies of the same variable disagree, val < 1.0."""
    print("=" * 55)
    print("Test 3: Inconsistent copies break cycle edges")
    print("=" * 55)

    clauses = [[1, 2, 3], [1, -2, 4]]
    G = arity_reduction(clauses)
    H = degree_reduction(G)

    assignment = {var_id: 0 for var_id in H.variables}

    x1_copies = [v for v in H.variables if v[0] == ("x", 1)]
    print(f"  x₁ copies: {x1_copies}")
    assignment[x1_copies[0]] = 1

    val = H.verify_assignment(assignment)
    print(f"  val with inconsistent x₁ copies = {val:.4f}")
    assert val < 1.0, f"Expected < 1.0, got {val}"
    print("  ✓ PASSED")
    print()


def test_various_formulas():
    """Test regularity on several formulas."""
    print("=" * 55)
    print("Test 4: Various formulas")
    print("=" * 55)

    formulas = [
        [[1, 2, 3]],
        [[1, 2, 3], [-1, -2, -3]],
        [[1, 2, 3], [1, 2, 4], [1, 3, 4]],
        [[1, 2, 3], [4, 5, 6]],
    ]

    for i, clauses in enumerate(formulas):
        G = arity_reduction(clauses)
        H = degree_reduction(G)
        inc = count_incident_edges(H)
        ok = all(c == 3 for c in inc.values())
        print(f"  Formula {i}: {clauses} → {H}, 3-regular: {'✓' if ok else '✗'}")
        assert ok, f"Formula {i} not 3-regular!"

    print("  ✓ All pass")
    print()


if __name__ == "__main__":
    test_regularity()
    test_satisfying_through_pipeline()
    test_inconsistent_copies()
    test_various_formulas()

    print("=" * 55)
    print("All Task 1.3 tests passed! ✓")
    print("=" * 55)
