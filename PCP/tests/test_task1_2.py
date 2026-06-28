"""
Test for Task 1.2: Arity Reduction (3SAT → 2CSP).

Tests the conversion of 3SAT clauses into a binary ConstraintGraph,
verifying that:
  - Satisfying assignments give val = 1.0
  - Unsatisfying assignments give val < 1.0
  - Graph dimensions match expected sizes
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nice_transformation import arity_reduction, _satisfying_assignments


def test_satisfying_assignments():
    """Verify the helper that enumerates satisfying variable-value tuples."""
    print("=" * 55)
    print("Test 0: _satisfying_assignments helper")
    print("=" * 55)

    # Clause (x₁ ∨ x₂ ∨ x₃) = [1, 2, 3]  (all positive)
    # Only fails when all variables = 0: (0,0,0)
    sat = _satisfying_assignments([1, 2, 3])
    print(f"  Clause [1, 2, 3]: {len(sat)} satisfying assignments")
    assert len(sat) == 7, f"Expected 7, got {len(sat)}"
    assert (0, 0, 0) not in sat, "(0,0,0) should NOT satisfy [1,2,3]"
    print("  ✓ Correct: 7 assignments, (0,0,0) excluded")

    # Clause (x₁ ∨ ¬x₂ ∨ x₃) = [1, -2, 3]
    # Fails when: x₁=0, x₂=1, x₃=0 → bits = (0, 1, 0)
    # (because all positive lits are 0, and ¬x₂ needs x₂=0 but x₂=1)
    sat2 = _satisfying_assignments([1, -2, 3])
    print(f"  Clause [1, -2, 3]: {len(sat2)} satisfying assignments")
    assert len(sat2) == 7, f"Expected 7, got {len(sat2)}"
    assert (0, 1, 0) not in sat2, "(0,1,0) should NOT satisfy [1,-2,3]"
    print("  ✓ Correct: 7 assignments, (0,1,0) excluded")

    print()


def test_single_clause_satisfying():
    """Single clause, satisfying assignment → val = 1.0"""
    print("=" * 55)
    print("Test 1: Single clause, SATISFYING assignment")
    print("=" * 55)

    # Formula: (x₁ ∨ ¬x₂ ∨ x₃)
    clauses = [[1, -2, 3]]
    G = arity_reduction(clauses)

    print(f"  Graph: {G}")
    print(f"  Variables: {list(G.variables.keys())}")
    assert G.num_variables == 4, f"Expected 4 vars, got {G.num_variables}"
    assert G.num_constraints == 3, f"Expected 3 constraints, got {G.num_constraints}"

    # Satisfying assignment: x₁=1, x₂=0, x₃=1
    # Clause check: (1 ∨ ¬0 ∨ 1) = (T ∨ T ∨ T) = True ✓
    #
    # sat_assignments stores VARIABLE VALUES, so we need
    # the tuple (1, 0, 1) — matching x₁=1, x₂=0, x₃=1 directly.
    sat = _satisfying_assignments([1, -2, 3])
    target = (1, 0, 1)
    w_val = sat.index(target)
    print(f"  Sat assignments: {sat}")
    print(f"  Target variable values {target} → W₀ = {w_val}")

    assignment = {
        ("x", 1): 1,      # x₁ = 1
        ("x", 2): 0,      # x₂ = 0
        ("x", 3): 1,      # x₃ = 1
        ("w", 0): w_val,   # W₀ indexes the matching entry
    }

    val = G.verify_assignment(assignment)
    print(f"  val = {val}")
    assert val == 1.0, f"Expected 1.0, got {val}"
    print("  ✓ PASSED")
    print()


def test_single_clause_unsatisfying():
    """Single clause, unsatisfying variable assignment → val < 1.0"""
    print("=" * 55)
    print("Test 2: Single clause, UNSATISFYING assignment")
    print("=" * 55)

    # Formula: (x₁ ∨ ¬x₂ ∨ x₃)
    clauses = [[1, -2, 3]]
    G = arity_reduction(clauses)

    # Unsatisfying assignment: x₁=0, x₂=1, x₃=0
    # Clause: (0 ∨ ¬1 ∨ 0) = (F ∨ F ∨ F) = False
    #
    # The variable-value tuple (0, 1, 0) is NOT in sat_assignments.
    # So no W₀ can produce exact match on all 3 positions.
    sat = _satisfying_assignments([1, -2, 3])
    print(f"  Sat assignments: {sat}")
    print(f"  Looking for (0,1,0) — should NOT be present: {(0,1,0) not in sat}")

    best_val = 0.0
    for w_val in range(len(sat)):
        assignment = {
            ("x", 1): 0,
            ("x", 2): 1,
            ("x", 3): 0,
            ("w", 0): w_val,
        }
        val = G.verify_assignment(assignment)
        best_val = max(best_val, val)
        bits = sat[w_val]
        matches = sum(1 for j, xv in enumerate([0, 1, 0]) if bits[j] == xv)
        print(f"    W₀={w_val} bits={bits} matches={matches}/3 val={val:.4f}")

    print(f"  Best val over all W₀ choices = {best_val:.4f}")
    assert best_val < 1.0, f"Expected < 1.0, got {best_val}"
    print("  ✓ PASSED (no W₀ can make all 3 edges consistent)")
    print()


def test_multi_clause():
    """Two clauses, satisfying assignment → val = 1.0"""
    print("=" * 55)
    print("Test 3: Two clauses, SATISFYING assignment")
    print("=" * 55)

    # Formula: (x₁ ∨ ¬x₂ ∨ x₃) ∧ (¬x₁ ∨ x₂ ∨ ¬x₃)
    clauses = [[1, -2, 3], [-1, 2, -3]]
    G = arity_reduction(clauses)

    print(f"  Graph: {G}")
    assert G.num_variables == 5, f"Expected 5 vars, got {G.num_variables}"
    assert G.num_constraints == 6, f"Expected 6 constraints, got {G.num_constraints}"

    # Assignment: x₁=1, x₂=1, x₃=0
    # Clause 0: (1 ∨ ¬1 ∨ 0) = (T ∨ F ∨ F) = True ✓
    # Clause 1: (¬1 ∨ 1 ∨ ¬0) = (F ∨ T ∨ T) = True ✓

    # Since sat_assignments stores VARIABLE VALUES:
    # Clause 0 [1, -2, 3]: we need variable values (1, 1, 0)
    sat0 = _satisfying_assignments([1, -2, 3])
    w0 = sat0.index((1, 1, 0))

    # Clause 1 [-1, 2, -3]: we need variable values (1, 1, 0)
    sat1 = _satisfying_assignments([-1, 2, -3])
    w1 = sat1.index((1, 1, 0))

    assignment = {
        ("x", 1): 1,
        ("x", 2): 1,
        ("x", 3): 0,
        ("w", 0): w0,
        ("w", 1): w1,
    }
    val = G.verify_assignment(assignment)
    print(f"  val = {val}")
    assert val == 1.0, f"Expected 1.0, got {val}"
    print("  ✓ PASSED")
    print()


if __name__ == "__main__":
    test_satisfying_assignments()
    test_single_clause_satisfying()
    test_single_clause_unsatisfying()
    test_multi_clause()

    print("=" * 55)
    print("All Task 1.2 tests passed! ✓")
    print("=" * 55)
