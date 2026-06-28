"""
Test for Task 1.1: ConstraintGraph class.

Builds a tiny hand-crafted CSP and checks verify_assignment()
under three scenarios: fully satisfied, half satisfied, none satisfied.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraint_graph import ConstraintGraph


def test_constraint_graph():
    # ── Build a small graph with 3 boolean variables ─────────
    G = ConstraintGraph()
    G.add_variable("x", domain_size=2)   # boolean
    G.add_variable("y", domain_size=2)   # boolean
    G.add_variable("z", domain_size=2)   # boolean

    print(f"Graph after adding variables: {G}")
    # → ConstraintGraph(variables=3, constraints=0)

    # ── Add two constraints ──────────────────────────────────
    # Constraint 1:  x == y   (equality)
    G.add_constraint("x", "y", lambda vx, vy: vx == vy)

    # Constraint 2:  y != z   (disequality)
    G.add_constraint("y", "z", lambda vy, vz: vy != vz)

    print(f"Graph after adding constraints: {G}")
    # → ConstraintGraph(variables=3, constraints=2)

    # ── Test 1: Fully satisfying assignment ──────────────────
    # x=1, y=1, z=0  →  (x==y)=True, (y!=z)=True  →  2/2 = 1.0
    assignment_full = {"x": 1, "y": 1, "z": 0}
    val = G.verify_assignment(assignment_full)
    print(f"\nTest 1 — Fully satisfying:  val = {val}")
    assert val == 1.0, f"Expected 1.0, got {val}"
    print("  ✓ PASSED")

    # ── Test 2: Half satisfying assignment ───────────────────
    # x=1, y=1, z=1  →  (x==y)=True, (y!=z)=False  →  1/2 = 0.5
    assignment_half = {"x": 1, "y": 1, "z": 1}
    val = G.verify_assignment(assignment_half)
    print(f"\nTest 2 — Half satisfying:   val = {val}")
    assert val == 0.5, f"Expected 0.5, got {val}"
    print("  ✓ PASSED")

    # ── Test 3: Fully unsatisfying assignment ────────────────
    # x=0, y=1, z=1  →  (x==y)=False, (y!=z)=False  →  0/2 = 0.0
    assignment_none = {"x": 0, "y": 1, "z": 1}
    val = G.verify_assignment(assignment_none)
    print(f"\nTest 3 — Fully unsatisfying: val = {val}")
    assert val == 0.0, f"Expected 0.0, got {val}"
    print("  ✓ PASSED")

    # ── Test 4: Empty graph (vacuous truth) ──────────────────
    G_empty = ConstraintGraph()
    G_empty.add_variable("a", 5)
    val = G_empty.verify_assignment({"a": 3})
    print(f"\nTest 4 — Empty constraints:  val = {val}")
    assert val == 1.0, f"Expected 1.0, got {val}"
    print("  ✓ PASSED")

    print("\n" + "=" * 50)
    print("All Task 1.1 tests passed! ✓")
    print("=" * 50)


if __name__ == "__main__":
    test_constraint_graph()
