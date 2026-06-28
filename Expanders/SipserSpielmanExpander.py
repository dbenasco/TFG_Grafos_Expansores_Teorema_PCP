"""
Sipser-Spielman Expander Codes over GF(2)

Based on: Spielman, "Linear-time encodable and decodable error-correcting codes"
          http://cs-www.cs.yale.edu/homes/spielman/PAPERS/expandersIT.pdf

This module implements:
  - InnerCode:        small linear code used at each constraint node
  - SSExpanderCode:   full expander code C(B, S) built from a (c,d)-regular
                      bipartite graph B and an inner code S
"""

import numpy as np
from typing import Optional
import math


# ──────────────────────────────────────────────────────────────
# Inner Code  (small code applied at each constraint node)
# ──────────────────────────────────────────────────────────────
class InnerCode:
    """
    A small binary linear code defined by its parity-check matrix Hs.

    Attributes
    ----------
    d  : int
        Block length of the inner code.
    Hs : np.ndarray, shape (r, d), dtype uint8
        Parity-check matrix of the inner code (r local constraints).
    """

    def __init__(self, d: int, Hs: np.ndarray):
        """
        Parameters
        ----------
        d  : Block length (number of columns of Hs).
        Hs : 2-D array (r × d) with entries in {0, 1}.
        """
        self.d = d
        self.Hs = np.asarray(Hs, dtype=np.uint8)
        assert self.Hs.ndim == 2 and self.Hs.shape[1] == d

    # ---------- public helpers ----------
    def check(self, block: np.ndarray) -> bool:
        """Return True iff *block* satisfies every local parity constraint."""
        block = np.asarray(block, dtype=np.uint8)
        assert block.shape == (self.d,)
        # syndrome = Hs @ block  (mod 2)
        syn = self.Hs @ block % 2          # shape (r,)
        return np.all(syn == 0)

    def syndrome(self, block: np.ndarray) -> np.ndarray:
        """Compute the syndrome vector Hs @ block (mod 2)."""
        block = np.asarray(block, dtype=np.uint8)
        assert block.shape == (self.d,)
        return (self.Hs @ block) % 2        # shape (r,)


# ──────────────────────────────────────────────────────────────
# Sipser–Spielman Expander Code
# ──────────────────────────────────────────────────────────────
class SSExpanderCode:
    """
    Expander code C(B, S) built from a (c,d)-regular bipartite graph B
    and a small inner code S.

    The set of valid codewords is:
        C(B, S) = { x ∈ GF(2)^n : H x^T = 0 }
    where H is constructed by "lifting" the inner code's parity-check
    equations onto the bipartite graph.

    Parameters
    ----------
    n     : int          – number of variable nodes (code length)
    c     : int          – degree of every variable node
    d     : int          – degree of every constraint node
    inner : InnerCode    – inner code applied at each constraint node
    seed  : int          – RNG seed for the random bipartite graph
    """

    def __init__(self, n: int, c: int, d: int,
                 inner: InnerCode, seed: int = 12345, allow_multi_edges: bool = False):
        assert (c * n) % d == 0, "c*n must be divisible by d"

        self.n = n
        self.c = c
        self.d = d
        self.m = c * n // d          # number of constraint nodes
        self.S = inner

        # Adjacency lists
        self.var_nbh: list[list[int]] = []   # var_nbh[v] = list of check indices
        self.chk_nbh: list[list[int]] = []   # chk_nbh[u] = list of variable indices

        # Parity-check matrix  (m*r_S  ×  n)
        self.H: Optional[np.ndarray] = None

        if allow_multi_edges or n > 500:
            self._build_regular_bipartite(seed)
        else:
            self._build_regular_bipartite_no_duplicates(seed)
        
        # Performance: skip full H generation for very large graphs unless needed
        if n < 2000:
            self._build_parity_check_matrix()

    # ──────────────────────────────────────────────
    # Graph construction
    # ──────────────────────────────────────────────
    def _has_duplicate_edges(self) -> bool:
        """Check whether the bipartite graph has any multi-edges."""
        for v in range(self.n):
            if len(set(self.var_nbh[v])) != self.c:
                return True
        return False

    def _build_regular_bipartite(self, seed: int) -> None:
        rng = np.random.default_rng(seed)

        # Create stubs
        left_stubs = np.repeat(np.arange(self.n), self.c)
        right_stubs = np.repeat(np.arange(self.m), self.d)

        # Shuffle
        rng.shuffle(left_stubs)
        rng.shuffle(right_stubs)

        # Faster adjacency list building for large n
        self.var_nbh = [[] for _ in range(self.n)]
        self.chk_nbh = [[] for _ in range(self.m)]

        # Vectorized pair processing isn't easy with lists, but we can do:
        for v, u in zip(left_stubs, right_stubs):
            self.var_nbh[v].append(u)
            self.chk_nbh[u].append(v)

        # Sanity check: degrees must be exactly (c, d)
        for v in range(self.n):
            if len(self.var_nbh[v]) != self.c:
                raise RuntimeError("Bad degree on variable node")
        for u in range(self.m):
            if len(self.chk_nbh[u]) != self.d:
                raise RuntimeError("Bad degree on check node")

    def _build_regular_bipartite_no_duplicates(self, seed: int) -> None:
        """
        Repeatedly build a random (c,d)-regular bipartite graph until
        one without multi-edges is obtained.
        """
        # REVISE WHY IS THAT THE EXPECTED NUMBER OF ATTEMPTS
        max_attempts = int(5 * math.exp((self.c - 1) * (self.d - 1) / 2))
        rng = np.random.default_rng(seed)

        for attempt in range(1, max_attempts + 1):
            new_seed = int(rng.integers(0, 2**63))
            self._build_regular_bipartite(new_seed)
            if not self._has_duplicate_edges():
                print(f"Graph built with no duplicates after {attempt} attempt(s)")
                return

        raise RuntimeError(
            "Failed to build simple bipartite graph after many retries."
        )

    # ──────────────────────────────────────────────
    # Parity-check matrix
    # ──────────────────────────────────────────────
    def _build_parity_check_matrix(self) -> None:
        """
        Build the global parity-check matrix H by "lifting" the inner
        code's parity-check rows onto the positions dictated by the
        bipartite graph.

        For every constraint node u and every local parity row r of S,
        we create a global row whose non-zero columns are the variable
        neighbours of u weighted by the coefficients in S.Hs[r].
        """
        rs = self.S.Hs.shape[0]                 # number of local parity rows
        H = np.zeros((self.m * rs, self.n), dtype=np.uint8)

        # For each constraint node u
        for u in range(self.m):
            #For each local parity row r of the inner code S
            for r in range(rs):
                global_row = u * rs + r # Calculate the index of the row relevant to u
                row_Hs = self.S.Hs[r]
                #For each variable node v connected to u
                for j in range(self.d):
                    v = self.chk_nbh[u][j]
                    #If the j-th local parity check is 1
                    if row_Hs[j]:
                        H[global_row, v] = 1

        self.H = H

    # ──────────────────────────────────────────────
    # Generator matrix  (basis of ker H over GF(2))
    # ──────────────────────────────────────────────
    def generator(self) -> tuple[np.ndarray, int]:
        """
        Compute X = generator matrix G whose rows span ker(H) over GF(2).

        Returns
        -------
        G : np.ndarray, shape (k, n), dtype uint8
            Generator matrix.
        k : int
            Code dimension  (= n − rank(H)).
        """
        m_rows, n_cols = self.H.shape
        A = self.H.copy()          # work on a copy
        pivot_col = [-1] * m_rows
        row = 0

        # --- Gaussian elimination mod 2 ---
        for col in range(n_cols):
            if row >= m_rows:
                break
            # Find pivot
            sel = -1
            for r in range(row, m_rows):
                if A[r, col]:
                    sel = r
                    break
            if sel == -1:
                continue

            # Swap rows
            A[[row, sel]] = A[[sel, row]]
            pivot_col[row] = col

            # Eliminate column from all other rows
            for r in range(m_rows):
                if r != row and A[r, col]:
                    A[r] ^= A[row]      # XOR (addition mod 2)
            row += 1

        rank = sum(1 for p in pivot_col if p != -1)
        k = n_cols - rank

        is_pivot = set(p for p in pivot_col if p != -1)
        free_cols = [c for c in range(n_cols) if c not in is_pivot]

        # --- Build G from free columns ---
        G = np.zeros((k, n_cols), dtype=np.uint8)
        for idx, f in enumerate(free_cols):
            x = np.zeros(n_cols, dtype=np.uint8)
            x[f] = 1

            # Back-substitution for pivot variables
            for r in range(m_rows):
                pc = pivot_col[r]
                if pc == -1:
                    continue
                s = 0
                for j in range(pc + 1, n_cols):
                    if A[r, j] and x[j]:
                        s ^= 1
                x[pc] = s

            G[idx] = x

        return G, k

    # ──────────────────────────────────────────────
    # Encoding
    # ──────────────────────────────────────────────
    def encode(self, msg: np.ndarray, G: np.ndarray) -> np.ndarray:
        """
        Encode a message using the generator matrix.

            codeword = msg · G   (over GF(2))

        Parameters
        ----------
        msg : 1-D array of length k  (message bits).
        G   : 2-D array of shape (k, n) (generator matrix).

        Returns
        -------
        codeword : 1-D array of length n.
        """
        msg = np.asarray(msg, dtype=np.uint8)
        k = G.shape[0]
        assert msg.shape == (k,), f"Message length {msg.shape} != k={k}"

        # codeword = msg @ G  (mod 2)
        codeword = (msg @ G) % 2
        return codeword.astype(np.uint8)

    # ──────────────────────────────────────────────
    # Syndrome decoding
    # ──────────────────────────────────────────────
    def syndrome_decoding(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the syndrome  s = H · x^T  (mod 2).

        https://en.wikipedia.org/wiki/Decoding_methods (Syndrome decoding)

        The syndrome of a received word y = x + e satisfies:
            s = H y^T = H(x+e)^T = H e^T
        """
        x = np.asarray(x, dtype=np.uint8)
        assert x.shape == (self.n,)

        rs = self.S.Hs.shape[0]
        s = np.zeros(self.m * rs, dtype=np.uint8)

        for u in range(self.m):
            block = np.array([x[v] for v in self.chk_nbh[u]], dtype=np.uint8)
            local_s = self.S.syndrome(block)
            for r in range(rs):
                s[u * rs + r] = local_s[r]

        return s

    # ──────────────────────────────────────────────
    # Parallel bit-flip decoder
    # ──────────────────────────────────────────────
    def decode_parallel(self, y: np.ndarray,
                        max_rounds: int = 100,
                        threshold_majority: int = -1) -> tuple[np.ndarray, int]:
        """
        Parallel bit-flip decoding for C(B, S).

        Each iteration:
          1. Each constraint node checks if its d-bit block ∈ S.
             If not, it votes for all its neighbour variables to flip.
          2. Each variable flips if it gets more than *threshold_majority* votes.
          3. Repeat until syndrome = 0 or *max_rounds* reached.

        Parameters
        ----------
        y                   : received word (modified in-place).
        max_rounds          : maximum number of decoding rounds.
        threshold_majority  : vote threshold; defaults to c/2.

        Returns
        -------
        y      : decoded word.
        rounds : number of rounds used, or -1 on failure.
        """
        y = np.array(y, dtype=np.uint8)   # work on a copy
        assert y.shape == (self.n,)

        if threshold_majority == -1:
            threshold_majority = self.c // 2

        for rnd in range(1, max_rounds + 1):
            votes = np.zeros(self.n, dtype=np.int32)
            unsatisfied = 0

            # 1) Constraint check & voting
            for u in range(self.m):
                block = np.array([y[v] for v in self.chk_nbh[u]], dtype=np.uint8)
                if not self.S.check(block):
                    unsatisfied += 1
                    for v in self.chk_nbh[u]:
                        votes[v] += 1

            # Already satisfied?
            if unsatisfied == 0:
                print(f"Decoded in {rnd - 1} rounds.")
                return y, rnd - 1

            # 2) Flip variables with enough votes
            to_flip = np.where(votes > threshold_majority)[0]

            if to_flip.size == 0:
                print(f"Decoder stuck after {rnd} rounds "
                      f"(unsatisfied={unsatisfied}).")
                return y, -1

            y[to_flip] ^= 1

            # Recompute syndrome to check early termination
            syn = self.syndrome_decoding(y)
            if np.all(syn == 0):
                print(f"Syndrome cleared after {rnd} rounds.")
                return y, rnd

        print(f"Failed to decode within {max_rounds} rounds.")
        return y, -1

    # ──────────────────────────────────────────────
    # Local (PCP-style) verifier
    # ──────────────────────────────────────────────
    def local_verify(self, y: np.ndarray,
                     num_samples: int = 10,
                     seed: int = 12345) -> bool:
        """
        Local PCP-style / Sipser–Spielman verifier.

        Randomly selects *num_samples* constraint nodes and checks
        whether the corresponding d-bit block satisfies the inner code.

        Complexity: O(num_samples * d)
        Bits read:  O(num_samples * d)
        If num_samples is constant the test is *local*.

        Returns True if all sampled checks pass; False otherwise.
        """
        y = np.asarray(y, dtype=np.uint8)
        assert y.shape == (self.n,)

        rng = np.random.default_rng(seed)

        for i in range(num_samples):
            u = rng.integers(0, self.m)
            block = np.array([y[v] for v in self.chk_nbh[u]], dtype=np.uint8)

            if not self.S.check(block):
                print(f"Failure detected at check {u} "
                      f"(sample {i + 1}/{num_samples})")
                return False

        print(f"All {num_samples} local checks passed.")
        return True


# ──────────────────────────────────────────────────────────────
# Demo / smoke test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ----- 1. Define a small inner code (single parity-check of length d) -----
    d = 4
    # Single parity-check: all-ones row  →  x1 ⊕ x2 ⊕ x3 ⊕ x4 = 0
    Hs = np.array([[1, 1, 1, 1]], dtype=np.uint8)
    inner = InnerCode(d, Hs)

    # ----- 2. Build the expander code -----
    n = 20   # number of variable nodes
    c = 4    # variable degree
    # m = c*n/d = 4*20/4 = 20 constraint nodes
    code = SSExpanderCode(n, c, d, inner, seed=42)

    print(f"Code parameters: n={code.n}, m={code.m}, c={code.c}, d={code.d}")
    print(f"H shape: {code.H.shape}")

    # ----- 3. Generator matrix & encoding -----
    G, k = code.generator()
    print(f"Code dimension k={k}  (rate = {k/n:.3f})")

    # Random message
    rng = np.random.default_rng(123)
    msg = rng.integers(0, 2, size=k, dtype=np.uint8)
    codeword = code.encode(msg, G)
    print(f"Message:  {msg}")
    print(f"Codeword: {codeword}")

    # Verify codeword satisfies H
    syn = code.syndrome_decoding(codeword)
    assert np.all(syn == 0), "Codeword does not satisfy parity checks!"
    print("✓ Codeword passes all parity checks.")

    # ----- 4. Introduce errors & decode -----
    noisy = codeword.copy()
    error_positions = rng.choice(n, size=2, replace=False)
    noisy[error_positions] ^= 1
    print(f"\nIntroduced errors at positions {error_positions}")
    print(f"Noisy word: {noisy}")

    decoded, rounds = code.decode_parallel(noisy, max_rounds=50)
    if rounds >= 0:
        if np.array_equal(decoded, codeword):
            print("✓ Decoding recovered the original codeword!")
        else:
            print("✗ Decoding converged but to a different codeword.")
    else:
        print("✗ Decoding failed.")

    # ----- 5. Local verification -----
    print("\n--- Local verification on codeword ---")
    code.local_verify(codeword, num_samples=10)

    print("\n--- Local verification on noisy word ---")
    code.local_verify(noisy, num_samples=10)
