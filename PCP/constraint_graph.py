"""
PCP Transformer — Step-by-step implementation of Dinur's PCP proof.

Based on: Arora & Barak, "Computational Complexity: A Modern Approach"
          Chapter 22: The PCP Theorem (Section 22.A: The "Nice Transformation")

This module will grow incrementally:
  Task 1.1  →  ConstraintGraph class
  Task 1.2  →  Arity Reduction  (3SAT → 2CSP)
  Task 1.3  →  Degree Reduction (Regularization)
  Task 1.4  →  Expansion Overlay
  ...and so on through Phases 2–4.
"""

from itertools import product


# ──────────────────────────────────────────────────────────────
# Task 1.1: The ConstraintGraph Class
# ──────────────────────────────────────────────────────────────
class ConstraintGraph:
    """
    A generic Constraint Satisfaction Problem (CSP) represented as a graph.

    - Each **node** is a variable with a finite domain of possible values.
    - Each **edge** is a binary constraint between two variables.

    This is the central data structure used by every phase of the
    PCP transformation pipeline.

    Attributes
    ----------
    variables : dict
        Maps variable ID → domain size.
        Example: {"x": 2, "y": 2} means x and y are boolean.

    constraints : list of tuples
        Each entry is (u, v, check_fn) where:
          - u, v     : variable IDs (must exist in self.variables)
          - check_fn : callable(val_u, val_v) → bool
                       Returns True if the constraint is satisfied.
    """

    def __init__(self):
        """Initialize an empty constraint graph with no variables or edges."""
        self.variables = {}      # {var_id: domain_size}
        self.constraints = []    # [(u, v, check_fn), ...]

    # ── Variable management ───────────────────────────────────

    def add_variable(self, var_id, domain_size):
        """
        Register a new variable in the graph.

        Parameters
        ----------
        var_id : hashable
            Unique identifier for this variable (e.g., "x", 0, ("w", 3)).
        domain_size : int
            Number of possible values this variable can take.
            For a boolean variable, domain_size = 2 (values 0 or 1).
            For an auxiliary clause variable (Task 1.2), domain_size = 7.

        Raises
        ------
        ValueError
            If var_id already exists or domain_size < 1.
        """
        if var_id in self.variables:
            raise ValueError(f"Variable '{var_id}' already exists.")
        if domain_size < 1:
            raise ValueError(f"Domain size must be ≥ 1, got {domain_size}.")
        self.variables[var_id] = domain_size

    # ── Constraint management ─────────────────────────────────

    def add_constraint(self, u, v, check_fn):
        """
        Add a binary constraint (edge) between two variables.

        Parameters
        ----------
        u, v : hashable
            Variable IDs. Both must have been added via add_variable().
        check_fn : callable(val_u, val_v) → bool
            A function that takes the values assigned to u and v,
            and returns True if the constraint is satisfied.

        Raises
        ------
        KeyError
            If u or v has not been registered as a variable.
        """
        if u not in self.variables:
            raise KeyError(f"Variable '{u}' not found. Add it first.")
        if v not in self.variables:
            raise KeyError(f"Variable '{v}' not found. Add it first.")
        self.constraints.append((u, v, check_fn))

    # ── Evaluation ────────────────────────────────────────────

    def verify_assignment(self, assignment):
        """
        Evaluate what fraction of constraints are satisfied by an assignment.

        This computes val(φ, σ) — the "value" of the CSP instance φ
        under the assignment σ. It is the central quantity tracked
        through the entire PCP transformation.

        Parameters
        ----------
        assignment : dict
            Maps variable ID → assigned value.
            Example: {"x": 1, "y": 0, "z": 1}

        Returns
        -------
        float
            Fraction of satisfied constraints, in [0.0, 1.0].
            Returns 1.0 if there are no constraints (vacuously true).

        Raises
        ------
        KeyError
            If assignment is missing a value for some variable
            that appears in a constraint.
        """
        if not self.constraints:
            return 1.0

        satisfied = 0
        for u, v, check_fn in self.constraints:
            val_u = assignment[u]   # will raise KeyError if missing
            val_v = assignment[v]
            if check_fn(val_u, val_v):
                satisfied += 1

        return satisfied / len(self.constraints)

    # ── Properties ────────────────────────────────────────────

    @property
    def num_variables(self):
        """Return the number of registered variables."""
        return len(self.variables)

    @property
    def num_constraints(self):
        """Return the number of constraints (edges)."""
        return len(self.constraints)

    def __repr__(self):
        return (
            f"ConstraintGraph("
            f"variables={self.num_variables}, "
            f"constraints={self.num_constraints})"
        )
