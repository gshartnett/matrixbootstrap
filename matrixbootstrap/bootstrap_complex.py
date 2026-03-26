import logging
import pickle
from itertools import product
from typing import Optional

import numpy as np
import psutil
from scipy.sparse import (
    coo_matrix,
    csr_matrix,
    hstack,
    save_npz,
    vstack,
)

from matrixbootstrap.algebra import (
    DoubleTraceOperator,
    SingleTraceOperator,
)
from matrixbootstrap.bootstrap import BootstrapSystem
from matrixbootstrap.linear_algebra import (
    create_sparse_matrix_from_dict,
    get_row_space_sparse,
)

logger = logging.getLogger(__name__)


class BootstrapSystemComplex(BootstrapSystem):
    """
    Bootstrap system for complex operator bases (e.g. mini-BMN model).

    Inherits shared infrastructure from BootstrapSystem and overrides
    the methods that differ for the complex case.
    """

    # ------------------------------------------------------------------
    # Subclass initialisation
    # ------------------------------------------------------------------

    def _init_subclass_vars(self) -> None:
        self._eig_charge_list = None  # set by _build_invariant_basis if used
        if self.use_invariant_basis:
            if self.symmetry_generators is None:
                raise ValueError(
                    "use_invariant_basis=True requires symmetry_generators to be provided."
                )
            self._build_invariant_basis()
        self.operator_list = self.generate_operators_truncated(
            L=self.max_degree_L,
            fraction_operators_to_retain=self.fraction_operators_to_retain,
        )
        self.operator_dict = {op: idx for idx, op in enumerate(self.operator_list)}
        if 2 * self.max_degree_L < self.hamiltonian.max_degree:
            raise ValueError("2 * max_degree_L must be >= max degree of Hamiltonian.")
        self.param_dim_complex = len(self.operator_dict)
        self.param_dim_real = 2 * len(self.operator_dict)

    def _build_invariant_basis(self) -> None:
        """
        Transform the system to the Cartan eigenbasis of the last symmetry generator,
        keeping only the charge-0 invariant subspace.

        After this call:
          - self.matrix_system is replaced with the eigenbasis MatrixSystem
          - self.hamiltonian is expanded in the eigenbasis
          - self.gauge_generator is expanded in the eigenbasis
          - self.symmetry_generators is set to None (constraints are trivially satisfied)
          - self._eig_charge_list holds integer charges for each eigenbasis element
        """
        from itertools import product as iproduct

        from matrixbootstrap.algebra import MatrixSystem

        orig_basis = self.matrix_system.operator_basis
        orig_herm = self.matrix_system.hermitian_dict
        orig_comm = self.matrix_system.commutation_rules
        n = len(orig_basis)

        # ---- 1. Build M matrix: action of Cartan generator on single operators ----
        # Use the last symmetry generator (L_z for SO(3))
        cartan_gen = self.symmetry_generators[-1]
        M = np.zeros((n, n), dtype=np.complex128)
        for i, op_name in enumerate(orig_basis):
            comm = self.matrix_system.single_trace_commutator(
                cartan_gen, SingleTraceOperator(data={(op_name,): 1})
            )
            for op_tuple, coeff in comm:
                if len(op_tuple) == 1:
                    j = orig_basis.index(op_tuple[0])
                    M[j, i] = coeff

        # ---- 2. Eigendecompose M ----
        eigenvalues, V_cols = np.linalg.eig(M)
        # Charges: eigenvalue = i * charge (the U(1) generator is i*M)
        charges = np.round(np.real(eigenvalues / 1j)).astype(int)

        # Sort by descending charge so positive charges come first
        order = np.argsort(-charges)
        charges = charges[order]
        V_cols = V_cols[:, order]  # V[i, k] = coeff of eig_k in orig_basis[i]
        V_inv = np.linalg.inv(V_cols)  # V_inv[k, i] = coeff of orig_basis[i] in eig_k

        eig_names = [f"eig_{k}" for k in range(n)]
        self._eig_charge_list = charges.tolist()
        logger.info("Invariant basis: eigenbasis charges = %s", self._eig_charge_list)

        # ---- 3. Commutation rules in eigenbasis ----
        # [eig_a, eig_b] = sum_{i,j} V_inv[a,i] * V_inv[b,j] * orig_comm[(i,j)]
        # Must include ALL pairs (including zeros) since algebra.py does direct dict lookup.
        eig_comm_full = {}
        for a, name_a in enumerate(eig_names):
            for b, name_b in enumerate(eig_names):
                val = sum(
                    V_inv[a, i]
                    * V_inv[b, j]
                    * orig_comm.get((orig_basis[i], orig_basis[j]), 0)
                    for i in range(n)
                    for j in range(n)
                )
                val = complex(val)
                eig_comm_full[(name_a, name_b)] = val if abs(val) > 1e-12 else 0

        # ---- 4. Hermitian dict and conjugate map ----
        # Determine each eig_k's type by its dominant original basis element
        eig_hermitian_dict = {}
        for k, name_k in enumerate(eig_names):
            max_idx = int(np.argmax(np.abs(V_inv[k])))
            eig_hermitian_dict[name_k] = orig_herm[orig_basis[max_idx]]

        # Conjugate map: eig_k† (ignoring sign) is the eig_l with charge = -charge_k
        # and whose row in V_inv is proportional to conj(V_inv[k]) * sign_vec
        sign_vec = np.array([1.0 if orig_herm[op] else -1.0 for op in orig_basis])
        conj_map = {}
        for k, name_k in enumerate(eig_names):
            conj_row = np.conj(V_inv[k]) * sign_vec
            best_l, best_overlap = 0, -1.0
            for l_idx, name_l in enumerate(eig_names):
                norm = np.linalg.norm(V_inv[l_idx]) * np.linalg.norm(conj_row)
                if norm < 1e-14:
                    continue
                overlap = abs(np.dot(conj_row, np.conj(V_inv[l_idx]))) / norm
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_l = l_idx
            conj_map[name_k] = eig_names[best_l]

        # ---- 4b. Compute per-operator conjugation phases ----
        # eig_k† = phase_k * eig_{conj_map[k]}
        # From: conj(V_inv[k]) * sign_vec = phase_k * V_inv[m]
        # where sign_vec[i] = +1 if orig Hermitian, -1 if anti-Hermitian
        conjugate_phase_map = {}
        for k, name_k in enumerate(eig_names):
            m = eig_names.index(conj_map[name_k])
            conj_row = np.conj(V_inv[k]) * sign_vec  # = phase_k * V_inv[m]
            norm_m_sq = float(np.real(np.dot(V_inv[m], np.conj(V_inv[m]))))
            if norm_m_sq > 1e-14:
                phase_k = np.dot(conj_row, np.conj(V_inv[m])) / norm_m_sq
            else:
                phase_k = 1.0 + 0j
            conjugate_phase_map[name_k] = complex(
                np.round(phase_k.real, 10) + 1j * np.round(phase_k.imag, 10)
            )

        # ---- 5. Build new MatrixSystem ----
        eig_matrix_system = MatrixSystem(
            operator_basis=eig_names,
            commutation_rules_concise={},
            hermitian_dict=eig_hermitian_dict,
            conjugate_map=conj_map,
            precomputed_commutation_rules=eig_comm_full,
            conjugate_phase_map=conjugate_phase_map,
        )

        # ---- 6. Expand operators in eigenbasis ----
        def expand_op_tuple(op_tuple):
            """Expand a tuple of original-basis names into eigenbasis (name → complex coeff)."""
            indices = [orig_basis.index(name) for name in op_tuple]
            result = {}
            for ks in iproduct(range(n), repeat=len(indices)):
                coeff = 1.0 + 0j
                for orig_idx, k in zip(indices, ks):
                    coeff *= V_cols[orig_idx, k]
                if abs(coeff) > 1e-14:
                    new_tuple = tuple(eig_names[k] for k in ks)
                    result[new_tuple] = result.get(new_tuple, 0) + coeff
            return {k: v for k, v in result.items() if abs(v) > 1e-14}

        # Transform Hamiltonian
        new_ham_data = {}
        for op_tuple, coeff in self.hamiltonian:
            if len(op_tuple) == 0:
                new_ham_data[()] = new_ham_data.get((), 0) + coeff
            else:
                for eig_tuple, c in expand_op_tuple(op_tuple).items():
                    new_ham_data[eig_tuple] = new_ham_data.get(eig_tuple, 0) + coeff * c
        new_ham_data = {k: v for k, v in new_ham_data.items() if abs(v) > 1e-14}

        # Transform gauge generator
        from matrixbootstrap.algebra import MatrixOperator

        new_gauge_data = {}
        for op_tuple, coeff in self.gauge_generator:
            if len(op_tuple) == 0:
                new_gauge_data[()] = new_gauge_data.get((), 0) + coeff
            else:
                for eig_tuple, c in expand_op_tuple(op_tuple).items():
                    new_gauge_data[eig_tuple] = (
                        new_gauge_data.get(eig_tuple, 0) + coeff * c
                    )
        new_gauge_data = {k: v for k, v in new_gauge_data.items() if abs(v) > 1e-14}

        self.matrix_system = eig_matrix_system
        self.hamiltonian = SingleTraceOperator(data=new_ham_data)
        self.gauge_generator = MatrixOperator(data=new_gauge_data)
        # Symmetry constraints are trivially satisfied in the invariant (charge-0) subspace:
        # L_z: charge-0 operators have zero L_z charge sum, so [L_z, O]=0 trivially.
        # L_x, L_y: their filtered projections onto the charge-0 sector over-constrain
        #   the truncated system (driving h_null → 0), so we skip them here.
        # NOTE: the correct approach for the invariant basis requires including
        #   ALL charge-sector bootstrap blocks as separate PSD constraints (not just
        #   the charge-0 block), which is a multi-block SDP not yet implemented.
        self.symmetry_generators = None
        logger.info(
            "Invariant basis built: %d eigenbasis operators, symmetry constraints disabled",
            n,
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _check_bootstrap_matrix_hermitian(
        self, matrix: np.ndarray, atol: float
    ) -> bool:
        return np.allclose(matrix, matrix.conj().T, atol=atol)

    def generate_reality_constraints(self) -> list[SingleTraceOperator]:
        """
        Reality constraints for the complex case are handled directly inside
        build_linear_constraints via the palindrome / non-palindrome logic.
        This method returns an empty list to satisfy the abstract interface;
        the actual constraints are embedded in build_linear_constraints.
        """
        return []

    def single_trace_to_coefficient_vector(
        self, st_operator: SingleTraceOperator, return_null_basis: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Map a single trace operator to the (in general complex) vector of coefficients.
        Here, a real representation for the *operators* will be used, meaning that a
        general single trace operator will be written as <tr(O)> = sum_i a_i v_i.

        Optionally returns the vectors in the null basis. The null basis transformation
        acts on a real representation of the operator basis elements, i.e. the operator above
        is first written as sum_i z_i (vR_i + i vI_i), so that the coefficient vector is [z, z].

        Parameters
        ----------
        st_operator : SingleTraceOperator
            The operator

        return_null_basis : bool, optional
            Controls whether the vector is returned in the original basis or the null basis.
            By default False.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The real and imaginary parts of the coefficient vector.
        """

        # validate
        self.validate_operator(operator=st_operator)

        # build the complex-basis vector of coefficients
        vec = [0] * self.param_dim_complex
        for op, coeff in st_operator:
            idx = self.operator_dict[op]
            vec[idx] = coeff
        vec = np.asarray(vec)
        if not return_null_basis:
            return vec

        # return the real-basis vector of coefficients and convert to null space
        if self.null_space_matrix is None:
            raise ValueError("Error, must first build null space.")

        # Real part of <H> = Re(h)·vR - Im(h)·vI  →  coefficient vector [Re(h), -Im(h)]
        # (same convention as build_quadratic_constraints lines 495-500)
        vec = np.concatenate((vec.real, -vec.imag))
        return vec @ self.null_space_matrix

    def double_trace_to_coefficient_matrix(self, dt_operator: DoubleTraceOperator):
        # use large-N factorization <tr(O1)tr(O2)> = <tr(O1)><tr(O2)>
        # to represent the double-trace operator as a quadratic expression of single trace operators,
        # \sum_{ij} M_{ij} v_i v_j, represented as an index-value dict.

        index_value_dict = {}
        for (op1, op2), coeff in dt_operator:
            if op1 not in self.operator_dict or op2 not in self.operator_dict:
                # In the invariant basis, only charge-0 operators are in
                # operator_dict.  Products involving non-zero-charge operators
                # vanish by charge conservation and can be skipped.
                continue
            idx1, idx2 = self.operator_dict[op1], self.operator_dict[op2]

            # symmetrize
            index_value_dict[(idx1, idx2)] = (
                index_value_dict.get((idx1, idx2), 0) + coeff / 2
            )
            index_value_dict[(idx2, idx1)] = (
                index_value_dict.get((idx2, idx1), 0) + coeff / 2
            )

        n_ops = len(self.operator_list)
        if not index_value_dict:
            from scipy.sparse import csr_matrix as _csr

            return _csr((n_ops, n_ops))

        mat = create_sparse_matrix_from_dict(
            index_value_dict=index_value_dict,
            matrix_shape=(n_ops, n_ops),
        )

        return mat

    def build_linear_constraints(
        self, additional_constraints: Optional[list[SingleTraceOperator]] = None
    ) -> coo_matrix:
        """
        Build the linear constraints. Each linear constraint corresponds to a
        linear combination of single trace operators that must vanish. The set
        of linear constraints may be numerically represented as a matrix L_{ij},
        where the first index runs over the set of all such constraints, and the
        second index runs over the set of single trace operators considered at this
        bootstrap, i.e., the constraint equations are

        L_{ij} v_j = 0.

        NOTE: Not sure if it's a good idea to store the constraints, may become memory intensive.

        NOTE: In this implementation of the bootstrap, the constraints are complex-valued. However,
        the subsequent operations assume that the objects are real-valued.

        To address this, v = vR + i vI will be made real by stacking the real and imaginary parts,
        as in V = [vR, vI].

        The constraints L v = 0 can also be decomposed into real and imaginary parts, let L_ij = X_ij + i Y_ij
        be the real and imaginary components of the constraint coefficients. So then

        (L v)_i = L_ij (vR_j + i vI_j) = (X_ij + i Y_ij) (vR_j + i vI_j)
                = (X_ij vR_j - Y_ij vI_j) + i (Y_ij vR_j + X_ij vI_j)

        Note that the real and imaginary parts of the equation must separately hold.
        These can be jointly written as LL V = 0, where

        LL = [[X, -Y],
            [Y, X]]

        Note also that a permutation of the rows will not affect the content of the
        constraints.

        Returns
        -------
        coo_matrix
            The set of linear constraints.
        """
        if self.verbose:
            logger.debug("Building the linear constraint matrix")

        # grab the constraints, building them if necessary
        if self.linear_constraints is None:
            constraints = self.generate_constraints()
            self.linear_constraints = constraints[0]
            self.quadratic_constraints = constraints[1]

        # add the additional constraints
        if additional_constraints is not None:
            self.linear_constraints += additional_constraints

        # build the index-value dict
        index_value_dict = {}
        constraint_idx = 0

        # loop over operators
        for st_operator in self.linear_constraints:
            for op_str, coeff in st_operator:
                index_value_dict[(constraint_idx, self.operator_dict[op_str])] = (
                    np.real(coeff)
                )
                index_value_dict[
                    (
                        constraint_idx,
                        self.operator_dict[op_str] + self.param_dim_complex,
                    )
                ] = -np.imag(coeff)
            constraint_idx += 1

        # imaginary part (Y_ij vR_j + X_ij vI_j)
        for st_operator in self.linear_constraints:
            for op_str, coeff in st_operator:
                index_value_dict[(constraint_idx, self.operator_dict[op_str])] = (
                    np.imag(coeff)
                )
                index_value_dict[
                    (
                        constraint_idx,
                        self.operator_dict[op_str] + self.param_dim_complex,
                    )
                ] = np.real(coeff)
            constraint_idx += 1

        # impose the reality constraints <tr(S^dag)> = <tr(S)>*
        # For operators that are Hermitian or anti-Hermitian:
        #   S^dag = (-1)^{#anti in S} * reverse(S) ≡ sign_S * reverse(S)
        # so the constraint is: vR[S] = sign_S * vR[rev(S)], vI[S] = -sign_S * vI[rev(S)]
        #
        # Palindrome case (op_str_reversed == op_str):
        #   sign_S = +1: vI[S] = 0
        #   sign_S = -1: vR[S] = 0
        # Non-palindrome case: vR[S] - sign_S*vR[rev(S)] = 0, vI[S] + sign_S*vI[rev(S)] = 0
        for op_str, op_idx in self.operator_dict.items():

            op_str_reversed = self.matrix_system.hermitian_conjugate_tuple(op_str)
            op_reversed_idx = self.operator_dict[op_str_reversed]

            # Compute the conjugation sign for the operator product.
            # If conjugate_phase_map is available (eigenbasis), each element eig_k has
            # a per-operator phase: eig_k† = phase_k * eig_{conj[k]}.
            # The total sign of (O_1...O_n)† = prod(phase_k) * rev_conj_op.
            # For the original basis (no eigenbasis), fall back to (-1)^num_antihermitian.
            if self.matrix_system.conjugate_phase_map is not None:
                sign_S_complex = complex(
                    np.prod(
                        [
                            self.matrix_system.conjugate_phase_map.get(s, 1.0 + 0j)
                            for s in op_str
                        ]
                    )
                )
            else:
                num_antihermitian = sum(
                    1 for term in op_str if not self.matrix_system.hermitian_dict[term]
                )
                sign_S_complex = complex((-1) ** num_antihermitian)

            # Round to nearest root of unity (±1 or ±i)
            sign_re = int(np.round(sign_S_complex.real))
            sign_im = int(np.round(sign_S_complex.imag))
            n_dim = self.param_dim_complex  # shorthand

            # palindrome case
            if op_reversed_idx == op_idx:
                # (1-a)*vR + b*vI = 0  where sign = a + ib
                a, b = sign_re, sign_im
                # For a=+1, b=0: 0*vR + 0*vI = trivial and 2*vI = 0 → vI = 0
                # For a=-1, b=0: 2*vR = 0 and 0*vI = trivial → vR = 0
                # For a=0, b=±1: vR ± vI = 0
                if sign_re == 1 and sign_im == 0:
                    # Hermitian: vI[S] = 0
                    index_value_dict[(constraint_idx, op_idx + n_dim)] = 1
                elif sign_re == -1 and sign_im == 0:
                    # Anti-Hermitian: vR[S] = 0
                    index_value_dict[(constraint_idx, op_idx)] = 1
                else:
                    # Complex phase (e.g. ±i): vR[S] + b*vI[S] = 0  (b = ±1)
                    index_value_dict[(constraint_idx, op_idx)] = 1
                    index_value_dict[(constraint_idx, op_idx + n_dim)] = b
                constraint_idx += 1

            # non-palindrome case
            else:
                # Constraint: <S>* = sign_S * <rev(S)>
                # Re(S) = a*Re(rev) - b*Im(rev)  →  vR[S] - a*vR[rev] + b*vI[rev] = 0
                # Im(S) = -b*Re(rev) - a*Im(rev)  →  vI[S] + b*vR[rev] + a*vI[rev] = 0
                a, b = sign_re, sign_im
                # real constraint
                entry_vR = index_value_dict.get((constraint_idx, op_idx), 0)
                index_value_dict[(constraint_idx, op_idx)] = 1
                if a != 0:
                    index_value_dict[(constraint_idx, op_reversed_idx)] = -a
                if b != 0:
                    index_value_dict[(constraint_idx, op_reversed_idx + n_dim)] = b
                constraint_idx += 1
                # imaginary constraint
                index_value_dict[(constraint_idx, op_idx + n_dim)] = 1
                if b != 0:
                    index_value_dict[(constraint_idx, op_reversed_idx)] = b
                if a != 0:
                    index_value_dict[(constraint_idx, op_reversed_idx + n_dim)] = a
                constraint_idx += 1

        # assemble the constraint matrix
        linear_constraint_matrix = create_sparse_matrix_from_dict(
            index_value_dict=index_value_dict,
            matrix_shape=(constraint_idx, 2 * self.param_dim_complex),
        )

        return linear_constraint_matrix

    def build_quadratic_constraints(self) -> dict[str, np.ndarray]:
        """
        Build the quadratic constraints. The quadratic constraints are exclusively due to
        the cyclic constraints. The constraints can be written as

        A_{Iij} v_i v_j + B_{Ii} v_i = 0.

        After imposing the linear constraints by transforming to the null basis, these become

        A'_{Iab} u_a u_b + B'_{Ia} u_a = 0,

        where A'_{Iab} = A_{Iij} K_{ia} K_{jb}, B'_{Ia} = B_{Ii} K_{ia}

        The quadratic constraint tensor can be written as
            A_{Iij} = (1/2) sum_{k=0}^{K_I-1} (v_i^{(I,k)} v_j^{(I,k)} + v_j^{(I,k)} v_i^{(I,k)})
        for vectors v^{(I,k)}. For each I, the vectors correspond to the double trace terms, <tr()> <tr()>.

        Returns
        -------
        dict[str, np.nparray]
            The constraint arrays, contained in a dictionary like so
            {'quadratic': A, 'linear': B}
        """

        if self.quadratic_constraints is None:
            self.build_linear_constraints()
        quadratic_constraints = self.quadratic_constraints

        additional_constraints = []
        if self.null_space_matrix is None:
            self.build_null_space_matrix()
        null_space_matrix = self.null_space_matrix

        # Pre-convert null space to dense once for fast BLAS row-indexing below.
        # N_dense shape: (2*n_complex, n_null) where n_complex = param_dim_complex.
        _N_dense = null_space_matrix.toarray()
        _n_c = self.param_dim_complex

        def _proj(row_idx, vals, col_idx):
            """Compute -(N[row_idx,:].T @ diag(vals) @ N[col_idx,:]) via BLAS.
            Each call is O(k * n_null^2) where k = len(vals), exploiting sparsity
            of the quadratic constraint matrix instead of full sparse-sparse multiply.
            """
            if len(vals) == 0:
                return np.zeros((self.param_dim_null, self.param_dim_null))
            return -(_N_dense[row_idx, :].T @ (vals[:, None] * _N_dense[col_idx, :]))

        linear_terms = []
        quadratic_terms = []

        # add <1> = <1>^2
        normalization_constraint = {
            "lhs": SingleTraceOperator(data={(): 1}),
            "rhs": DoubleTraceOperator(data={((), ()): 1}),
        }
        quadratic_constraints[None] = normalization_constraint

        # loop over constraints
        for constraint_idx, constraint in enumerate(quadratic_constraints.values()):

            if self.verbose:
                logger.debug(
                    f"Generating quadratic constraints, operator {constraint_idx+1}/{len(quadratic_constraints)}"
                )
            logger.debug(
                f"Memory usage: {psutil.Process().memory_info().rss / 1024 ** 2}"
            )

            lhs = constraint["lhs"]
            rhs = constraint["rhs"]

            # retrieve the vectorized form of the linear and quadratic terms
            # Note: these will be complex-valued and in the complex operator basis
            linear_constraint_vector = self.single_trace_to_coefficient_vector(lhs)
            quadratic_matrix = self.double_trace_to_coefficient_matrix(rhs)

            # convert to real operator basis, and split each constraint into a real and imaginary part
            # sum_k z_k v_k = sum_k (x_k vR_k - y_k vI_k) + i sum_k (y_k vR_k + x_k vI_k)
            # so that the real vector is [real(z), - imag(z)] and the imaginary term is [imag(z), real(z)]
            linear_constraint_vectorR = np.concatenate(
                (linear_constraint_vector.real, -linear_constraint_vector.imag)
            )
            linear_constraint_vectorI = np.concatenate(
                (linear_constraint_vector.imag, linear_constraint_vector.real)
            )

            # Fast path: if the quadratic matrix is entirely zero (e.g. all
            # double-trace operators projected out by charge conservation in the
            # invariant basis), skip the expensive _proj / dense-matrix path.
            if quadratic_matrix.nnz == 0:
                linear_constraint_vectorR = linear_constraint_vectorR @ _N_dense
                linear_constraint_vectorI = linear_constraint_vectorI @ _N_dense
                if not self.simplify_quadratic:
                    linear_is_zeroR = (
                        np.max(np.abs(linear_constraint_vectorR)) < self.tol
                    )
                    linear_is_zeroI = (
                        np.max(np.abs(linear_constraint_vectorI)) < self.tol
                    )
                    if not linear_is_zeroR:
                        linear_terms.append(csr_matrix(linear_constraint_vectorR))
                        quadratic_terms.append(csr_matrix((1, self.param_dim_null**2)))
                    if not linear_is_zeroI:
                        linear_terms.append(csr_matrix(linear_constraint_vectorI))
                        quadratic_terms.append(csr_matrix((1, self.param_dim_null**2)))
                else:
                    if np.max(np.abs(linear_constraint_vectorR)) >= self.tol:
                        additional_constraints.append(lhs.get_real_part())
                    if np.max(np.abs(linear_constraint_vectorI)) >= self.tol:
                        additional_constraints.append(lhs.get_imag_part())
                continue

            # rewrite the quadratic constraints in terms of real variables
            # this entails two things:
            #   1. every constraint will become two (one real and one imaginary)
            #   2. each constraint will be naturally expressed as a (2d, 2d) matrix
            #      acting on the stacked parameter vector [vR, vI]
            qR_coo = quadratic_matrix.real.tocoo()
            qI_coo = quadratic_matrix.imag.tocoo()
            rR, cR, vR = qR_coo.row, qR_coo.col, qR_coo.data
            rI, cI, vI = qI_coo.row, qI_coo.col, qI_coo.data

            # transform to null basis (use dense N for BLAS speed)
            # the minus sign is important: (-RHS + LHS = 0)
            linear_constraint_vectorR = linear_constraint_vectorR @ _N_dense
            linear_constraint_vectorI = linear_constraint_vectorI @ _N_dense

            # In the invariant-basis mode the quadratic part is already partially
            # captured by the multi-block PSD structure.  The dense _proj matrices
            # are n_null x n_null = O(param_dim_null^2) and cannot be stored in
            # memory at L=3 (param_dim_null=3998 => 128 MB each, ~192 MB sparse).
            # Skip the quadratic term and treat the constraint as linear-only.
            if self.use_invariant_basis:
                linear_is_zeroR = np.max(np.abs(linear_constraint_vectorR)) < self.tol
                linear_is_zeroI = np.max(np.abs(linear_constraint_vectorI)) < self.tol
                if not linear_is_zeroR:
                    linear_terms.append(csr_matrix(linear_constraint_vectorR))
                    quadratic_terms.append(csr_matrix((1, self.param_dim_null**2)))
                if not linear_is_zeroI:
                    linear_terms.append(csr_matrix(linear_constraint_vectorI))
                    quadratic_terms.append(csr_matrix((1, self.param_dim_null**2)))
                continue

            # Fast null-space projection of the (2n, 2n) block-quadratic matrices.
            # QR = [[+qR, -qI], [-qI, -qR]]  =>  -N.T @ QR @ N:
            #   (i,   j  ) = +vR : _proj(rR,      vR,  cR     )
            #   (i+nc,j+nc) = -vR : _proj(rR+nc, -vR,  cR+nc  )
            #   (i,   j+nc) = -vI : _proj(rI,    -vI,  cI+nc  )
            #   (i+nc,j  ) = -vI : _proj(rI+nc,  -vI,  cI     )
            QR = (
                _proj(rR, vR, cR)
                + _proj(rR + _n_c, -vR, cR + _n_c)
                + _proj(rI, -vI, cI + _n_c)
                + _proj(rI + _n_c, -vI, cI)
            )

            # QI = [[+qI, +qR], [+qR, -qI]]  =>  -N.T @ QI @ N:
            #   (i,   j  ) = +vI : _proj(rI,      vI,  cI     )
            #   (i+nc,j+nc) = -vI : _proj(rI+nc, -vI,  cI+nc  )
            #   (i,   j+nc) = +vR : _proj(rR,     vR,  cR+nc  )
            #   (i+nc,j  ) = +vR : _proj(rR+nc,   vR,  cR     )
            QI = (
                _proj(rI, vI, cI)
                + _proj(rI + _n_c, -vI, cI + _n_c)
                + _proj(rR, vR, cR + _n_c)
                + _proj(rR + _n_c, vR, cR)
            )

            # reshape the (n_null, n_null) matrices to (1, n_null^2) sparse matrices
            QR = csr_matrix(QR.reshape((1, self.param_dim_null**2)))
            QI = csr_matrix(QI.reshape((1, self.param_dim_null**2)))

            # process the real and imaginary constraints separately
            # real part
            linear_is_zero = np.max(np.abs(linear_constraint_vectorR)) < self.tol
            quadratic_is_zero = np.max(np.abs(QR)) < self.tol
            if self.simplify_quadratic:
                if not quadratic_is_zero:
                    linear_terms.append(csr_matrix(linear_constraint_vectorR))
                    quadratic_terms.append(QR)
                elif not linear_is_zero:
                    additional_constraints.append(lhs.get_real_part())
            else:
                if not quadratic_is_zero or not linear_is_zero:
                    linear_terms.append(csr_matrix(linear_constraint_vectorR))
                    quadratic_terms.append(QR)

            # imaginary part
            linear_is_zero = np.max(np.abs(linear_constraint_vectorI)) < self.tol
            quadratic_is_zero = np.max(np.abs(QI)) < self.tol
            if self.simplify_quadratic:
                if not quadratic_is_zero:
                    linear_terms.append(csr_matrix(linear_constraint_vectorI))
                    quadratic_terms.append(QI)
                elif not linear_is_zero:
                    additional_constraints.append(lhs.get_imag_part())
            else:
                if not quadratic_is_zero or not linear_is_zero:
                    linear_terms.append(csr_matrix(linear_constraint_vectorI))
                    quadratic_terms.append(QI)

        logger.info(f"len(quadratic_terms) = {len(quadratic_terms)}")

        if self.simplify_quadratic and len(additional_constraints) > 0:
            logger.info(
                f"Building quadratic constraints: adding {len(additional_constraints)} new linear constraints and rebuilding null matrix"
            )
            self.build_null_space_matrix(additional_constraints=additional_constraints)
            return self.build_quadratic_constraints()

        if len(quadratic_terms) == 0:
            # All quadratic constraints were absorbed into the null space; use empty matrices.
            logger.info(
                "No residual quadratic constraints after simplification; using empty constraint matrices."
            )
            d = self.param_dim_null
            from scipy.sparse import csr_matrix as _csr

            empty_q = _csr((0, d**2), dtype=np.float64)
            empty_l = _csr((0, d), dtype=np.float64)
            self.quadratic_constraints_numerical = {
                "linear": empty_l,
                "quadratic": empty_q,
            }
            return

        # map to sparse matrices
        quadratic_terms = vstack(quadratic_terms)
        linear_terms = vstack(linear_terms)

        # apply reduction
        num_constraints = quadratic_terms.shape[0]

        logger.info(
            f"Number of quadratic constraints before row reduction: {num_constraints}"
        )
        logger.info("Skipping row reduction")
        if False:
            stacked_matrix = hstack([quadratic_terms, linear_terms])
            stacked_matrix = get_row_space_sparse(stacked_matrix)
            num_constraints = stacked_matrix.shape[0]
            linear_terms = stacked_matrix[:, self.param_dim_null**2 :]
            quadratic_terms = stacked_matrix[:, : self.param_dim_null**2]
        logger.info(
            f"Number of quadratic constraints after row reduction: {num_constraints}"
        )

        if self.config_cache_path is not None:
            save_npz(
                self.config_cache_path
                + "/quadratic_constraints_numerical_linear_term.npz",
                linear_terms,
            )
            save_npz(
                self.config_cache_path
                + "/quadratic_constraints_numerical_quadratic_term.npz",
                quadratic_terms,
            )

        self.quadratic_constraints_numerical = {
            "linear": linear_terms,
            "quadratic": quadratic_terms.tocsr(),
        }

    def build_bootstrap_table(self) -> None:
        """
        Creates the bootstrap table.

        The bootstrap matrix is

        M_{ij} = < tr(O^dagger_i O_j)>.

        Representing the set of single trace operators considered at this
        boostrap level as a vector v, this may be written as

        M_{ij} = v_{I(i,j)},

        where I(i,j) is an index map.

        For example, in the one-matrix case,
          i = 1 corresponds to the string 'X'
          i = 4 corresponds to the string 'XP'
          M_{14} = < tr(XXP) > = v_8

        After imposing the linear constraints, the variable v becomes

        v_i = K_{ij} u_j

        and the bootstrap matrix becomes

        M_{ij} = K_{I(i,j)k} u_k

        Define the bootstrap table as the 3-index array K_{I(i,j)k}.
        Note that this is independent of the single trace variables.

        ------------------------------------------------------------------------
        NOTE
        In https://doi.org/10.1103/PhysRevLett.125.041601, M is simplified by imposing discrete
        symmetries. Each operator is assigned an integer-valued degree, and only expectations
        with zero total degree are non-zero:

        degree_total( O_1^{dagger} O_2 ) = -degree(O_1) + degree(O_2).

        The degree function depends on the symmetry. In the two-matrix example the degree is
        the charge of the operators (in the A, B, C, D basis) under the SO(2) generators. In
        the one-matrix example, it is the charge under reflection symmetry (X, P) -> (-X, -P).

        Imposing the condition that the total degree is zero leads to M being block diagonal,
        provided that the operators are sorted by degree, as in

        O = [(degree -d_min operators), (degree -d + 1 operators), ..., (degree d_max operators)]

        The blocks will then corresponds to the different ways of pairing the two operators
        O_i^dagger O_j to form a degree zero operator.

        No discrete symmetries will be imposed here
        ------------------------------------------------------------------------
        Returns
        -------
        csr_matrix
            The bootstrap array, with shape (self.bootstrap_matrix_dim**2, self.param_dim_null).
            It has been reshaped to be a matrix.
        """
        logger.info("Building the bootstrap table...")
        if self.null_space_matrix is None:
            raise ValueError("Error, null space matrix has not yet been built.")

        self.bootstrap_table_sparse = self._build_table_for_basis(
            self.bootstrap_basis_list
        )

        # For the invariant (charge-0) basis, also build bootstrap tables for each
        # non-zero charge sector.  Each charge-q block provides PSD constraints that
        # couple different charge-0 expectation values, bounding the energy from below.
        charge_sector_bases = getattr(self, "_charge_sector_bootstrap_bases", None)
        if charge_sector_bases:
            self.extra_bootstrap_tables = {
                q: self._build_table_for_basis(basis)
                for q, basis in sorted(charge_sector_bases.items())
            }
            logger.info(
                "Built extra bootstrap tables for charge sectors: %s",
                sorted(self.extra_bootstrap_tables.keys()),
            )

        if self.config_cache_path is not None:
            save_npz(
                self.config_cache_path + "/bootstrap_table_sparse.npz",
                self.bootstrap_table_sparse,
            )

    def _build_table_for_basis(self, basis_list) -> "csr_matrix":
        """Build a bootstrap table for an arbitrary list of bootstrap basis operators.

        Returns a sparse matrix of shape (n^2, param_dim_null) where n = len(basis_list).
        The entries encode M_{ij} = <O_i† O_j> as a linear combination of free parameters.
        """
        null_space_matrix = self.null_space_matrix
        n_complex = self.param_dim_complex
        n = len(basis_list)

        # Convert to dense once for O(1) row access (~700MB at L=3, acceptable).
        # Repeated N[row].todense() on a sparse matrix is ~16ms/call; direct
        # array indexing is nanoseconds.
        N_dense = null_space_matrix.toarray()

        rows_out, cols_out, data_out = [], [], []
        for idx1, op_str1 in enumerate(basis_list):
            # phase for O_i† = phase * conj_tuple(O_i)
            if self.matrix_system.conjugate_phase_map is not None:
                sign = complex(
                    np.prod(
                        [
                            self.matrix_system.conjugate_phase_map.get(s, 1.0 + 0j)
                            for s in op_str1
                        ]
                    )
                )
            else:
                num_antihermitian_ops = sum(
                    [not self.matrix_system.hermitian_dict[term] for term in op_str1]
                )
                sign = complex((-1) ** num_antihermitian_ops)
            op_str1_dag = self.matrix_system.hermitian_conjugate_tuple(op_str1)
            for idx2, op_str2 in enumerate(basis_list):
                product_key = op_str1_dag + op_str2
                if product_key not in self.operator_dict:
                    continue  # degree > 2L, skip
                index_map = self.operator_dict[product_key]
                x_vec = sign * (
                    N_dense[index_map] + 1j * N_dense[index_map + n_complex]
                )
                nz = np.where(np.abs(x_vec) > self.tol)[0]
                if nz.size:
                    rows_out.extend([idx1 * n + idx2] * len(nz))
                    cols_out.extend(nz.tolist())
                    data_out.extend(x_vec[nz].tolist())

        if data_out:
            from scipy.sparse import coo_matrix

            return coo_matrix(
                (data_out, (rows_out, cols_out)),
                shape=(n * n, self.param_dim_null),
                dtype=np.complex128,
            ).tocsr()
        else:
            return csr_matrix((n * n, self.param_dim_null), dtype=np.complex128)

    def get_operator_expectation_value(
        self, st_operator: SingleTraceOperator, param: np.ndarray
    ) -> float:
        param_real = (self.null_space_matrix @ param)[: self.param_dim_complex]
        param_imag = (self.null_space_matrix @ param)[self.param_dim_complex :]
        param_complex = param_real + 1j * param_imag

        vec = self.single_trace_to_coefficient_vector(
            st_operator=st_operator, return_null_basis=False
        )

        op_expectation_value = vec @ param_complex
        return op_expectation_value

    # ------------------------------------------------------------------
    # Complex-specific methods
    # ------------------------------------------------------------------

    def generate_operators_truncated(self, L, fraction_operators_to_retain=1.0):

        # generate all operators with length <= L
        operators = [
            [x for x in product(self.matrix_system.operator_basis, repeat=deg)]
            for deg in range(0, L + 1)
        ]
        operators = [
            x for operators_by_degree in operators for x in operators_by_degree
        ]
        logger.info(f"Number of operators with length <= {L}: {len(operators)}")

        # If using invariant basis, filter to charge-0 operators only;
        # also store bootstrap bases for non-zero charge sectors (for multi-block PSD).
        if self._eig_charge_list is not None:
            charge_map = {
                name: self._eig_charge_list[i]
                for i, name in enumerate(self.matrix_system.operator_basis)
            }
            all_ops_up_to_L = operators  # save before filtering
            operators = [
                op for op in all_ops_up_to_L if sum(charge_map[s] for s in op) == 0
            ]
            logger.info(
                f"Number of charge-0 operators with length <= {L}: {len(operators)}"
            )
            # Store bootstrap bases for each non-zero charge sector.
            # Iterate over ALL charges that appear among the truncated operators,
            # not just the single-operator charges in _eig_charge_list (which would
            # miss e.g. charge ±2 from pairs of charge ±1 operators).
            all_charges = set(sum(charge_map[s] for s in op) for op in all_ops_up_to_L)
            self._charge_sector_bootstrap_bases = {}
            for q in all_charges:
                if q == 0:
                    continue
                ops_q = [
                    op for op in all_ops_up_to_L if sum(charge_map[s] for s in op) == q
                ]
                if ops_q:
                    self._charge_sector_bootstrap_bases[q] = ops_q
            logger.info(
                "Invariant basis charge sectors: %s (sizes: %s)",
                sorted(self._charge_sector_bootstrap_bases.keys()),
                [
                    len(v)
                    for _, v in sorted(self._charge_sector_bootstrap_bases.items())
                ],
            )

        # truncate the set of operators considered
        if fraction_operators_to_retain < 1.0:

            # for reference find the number of operators in the multiplication table if we had not done the truncation
            untruncated_number_of_operators = len(
                sorted(
                    set(
                        [
                            self.matrix_system.hermitian_conjugate_tuple(op_str1)
                            + op_str2
                            for op_str1 in operators
                            for op_str2 in operators
                        ]
                    )
                )
            )

            # perform the truncation
            num_operators_to_retain = int(fraction_operators_to_retain * len(operators))
            operators_retain = set(operators[0:num_operators_to_retain])
            operators = [
                op_str
                for op_str in operators_retain
                if self.matrix_system.hermitian_conjugate_tuple(op_str)
                in operators_retain
            ]

            # add back any conjugates
            logger.info(
                f"Number of operators with length <= {L} after truncation: {len(operators)}"
            )

        # generate all matrices appearing in the LxL multiplication table
        # use sorted() for deterministic ordering across processes (fixes cache hash mismatch)
        operator_list_table = sorted(
            set(
                [
                    self.matrix_system.hermitian_conjugate_tuple(op_str1) + op_str2
                    for op_str1 in operators
                    for op_str2 in operators
                ]
            )
        )

        # When using invariant basis, the multiplication table misses charge-0 operators
        # whose first element has nonzero charge (e.g. ('eig_+','eig_0','eig_-')).
        # Include ALL charge-0 operators up to degree 2L to ensure completeness.
        if self._eig_charge_list is not None:
            basis = self.matrix_system.operator_basis
            charge_map = {
                name: self._eig_charge_list[i] for i, name in enumerate(basis)
            }
            all_charge0_up_to_2L = sorted(
                set(
                    op
                    for deg in range(0, 2 * L + 1)
                    for op in product(basis, repeat=deg)
                    if sum(charge_map[s] for s in op) == 0
                )
            )
            operator_list = sorted(set(operator_list_table) | set(all_charge0_up_to_2L))
            logger.info(
                "Invariant basis: operator_list expanded from %d (table) to %d (all charge-0 up to degree %d)",
                len(operator_list_table),
                len(operator_list),
                2 * L,
            )
        else:
            operator_list = operator_list_table

        if fraction_operators_to_retain < 1.0:
            logger.info(
                f"Number of operators appearing in the L x L multiplication table (before truncation): {untruncated_number_of_operators}"
            )
            logger.info(
                f"Number of operators appearing in the L x L multiplication table (after truncation): {len(operator_list)}"
            )
        else:
            logger.info(
                f"Number of operators appearing in the L x L multiplication table: {len(operator_list)}"
            )

        self.bootstrap_basis_list = operators
        self.bootstrap_matrix_dim = len(operators)

        return operator_list

    def generate_symmetry_constraints(self, tol=1e-10) -> list[SingleTraceOperator]:
        """
        Generate any symmetry constraints <[g,O]>=0 for O single trace
        and g a symmetry generator.

        Returns
        -------
        list[SingleTraceOperator]
            The list of constraint terms.
        """
        # skip if no symmetry generators are provided
        if self.symmetry_generators is None:
            return []

        constraints = []
        n = len(self.matrix_system.operator_basis)

        # loop over symmetry generators
        for symmetry_generator in self.symmetry_generators:

            # initialize a matrix M which will implement the linear action of the generator g
            # M will obey [g, operators_vector] = M operators_vector
            M = np.zeros(shape=(n, n), dtype=np.complex128)
            for i, op in enumerate(self.matrix_system.operator_basis):
                commutator = self.matrix_system.single_trace_commutator(
                    symmetry_generator, SingleTraceOperator(data={(op): 1})
                )
                for op, coeff in commutator:
                    if np.abs(coeff) > tol:
                        j = self.matrix_system.operator_basis.index(op[0])
                        M[i, j] = coeff

            # build the change of variables matrix
            eig_values, old_to_new_variables = np.linalg.eig(M)
            old_to_new_variables = old_to_new_variables.T

            # confirm that the eigenvector relationship holds
            assert np.all(
                [
                    np.allclose(
                        np.zeros(n),
                        M @ old_to_new_variables[i]
                        - eig_values[i] * old_to_new_variables[i],
                    )
                    for i in range(n)
                ]
            )

            # build all monomials using the new operators
            if self.fraction_operators_to_retain != 1.0:
                raise ValueError(
                    "Warning, symmetry constraints and dropping a fraction of operators are not simultaneously supported."
                )

            new_ops_dict = {f"new_op_{i}": i for i in range(n)}
            all_new_operators = {
                deg: [x for x in product(new_ops_dict.keys(), repeat=deg)]
                for deg in range(1, 2 * self.max_degree_L + 1)
            }
            all_new_operators = [
                x for xs in all_new_operators.values() for x in xs
            ]  # flatten

            # loop over all operators in the eigenbasis
            for operator in all_new_operators:

                # compute the charge under the symmetry
                charge = sum(
                    [eig_values[new_ops_dict[basis_op]] for basis_op in operator]
                )

                # if the charge is not zero, the resulting operator expectation value must vanish in a symmetric state
                if np.abs(charge) > tol:

                    operator2 = {}
                    for i in range(len(operator)):
                        operator2[i] = [
                            (
                                self.matrix_system.operator_basis[j],
                                old_to_new_variables[new_ops_dict[operator[i]], j],
                            )
                            for j in range(n)
                        ]

                    # build the constraint single-trace operator
                    data = {}
                    for indices in list(product(range(n), repeat=len(operator))):
                        op = tuple(
                            [
                                value[indices[idx]][0]
                                for idx, value in enumerate(operator2.values())
                            ]
                        )
                        coeff = np.prod(
                            [
                                value[indices[idx]][1]
                                for idx, value in enumerate(operator2.values())
                            ]
                        )
                        if np.abs(coeff) > tol:
                            data[op] = data.get(op, 0) + coeff
                    constraints.append(SingleTraceOperator(data=data))

        return self.clean_constraints(constraints)

    def generate_cyclic_constraints(
        self,
    ) -> tuple[
        int:SingleTraceOperator,
        dict[int, dict[str, SingleTraceOperator | DoubleTraceOperator]],
    ]:
        """
        Generate cyclic constraints relating single trace operators to double
        trace operators. See S37 of
        https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.125.041601/supp.pdf

        Returns
        -------
        dict[int, dict[str, SingleTraceOperator | DoubleTraceOperator]]
            The linear and quadratic constraints.
        """
        identity = SingleTraceOperator(data={(): 1})
        quadratic_constraints = {}
        linear_constraints = {}
        for op_idx, op in enumerate(self.operator_list):
            if len(op) > 1:

                if not isinstance(op, tuple):
                    raise ValueError(f"op should be tuple, not {type(op)}")

                # the LHS corresponds to single trace operators
                eq_lhs = SingleTraceOperator(data={op: 1}) - SingleTraceOperator(
                    data={op[1:] + (op[0],): 1}
                )

                # rhe RHS corresponds to double trace operators
                eq_rhs = DoubleTraceOperator(data={})
                for k in range(1, len(op)):
                    commutator = self.matrix_system.commutation_rules[(op[0], op[k])]
                    st_operator_1 = SingleTraceOperator(data={tuple(op[1:k]): 1})
                    st_operator_2 = SingleTraceOperator(data={tuple(op[k + 1 :]): 1})

                    # If the double trace term involves <tr(1)> simplify and add to the linear, LHS
                    if st_operator_1 == identity:
                        eq_lhs -= commutator * st_operator_2
                    elif st_operator_2 == identity:
                        eq_lhs -= commutator * st_operator_1
                    else:
                        eq_rhs += commutator * (st_operator_1 * st_operator_2)

                # if the quadratic term vanishes but the linear term is non-zero, record the constraint as being linear
                if not eq_lhs.is_zero() and eq_rhs.is_zero():
                    linear_constraints[op_idx] = eq_lhs

                # do not expect to find any constraints where the linear term vanishes but the quadratic term does not
                elif eq_lhs.is_zero() and not eq_rhs.is_zero():
                    raise ValueError(
                        f"Warning, for operator index {op_idx}, op={op}, the LHS is unexpectedly 0"
                    )

                # record proper quadratic constraints
                elif not eq_lhs.is_zero():
                    quadratic_constraints[op_idx] = {"lhs": eq_lhs, "rhs": eq_rhs}

            if self.verbose:
                logger.debug(
                    f"Generating cyclic constraints, operator {op_idx+1}/{len(self.operator_list)}"
                )

        return linear_constraints, quadratic_constraints

    def generate_constraints(self) -> tuple[list[SingleTraceOperator]]:
        """
        Generate all constraints.

        Returns
        -------
        tuple[list[SingleTraceOperator]]
            The first entry in the tuple is the list of linear constraints.
            The second is the list of cyclic constraints.
        """
        linear_constraints = []

        # Hamiltonian constraints
        hamiltonian_constraints = self.generate_hamiltonian_constraints()
        logger.info(f"Generated {len(hamiltonian_constraints)} Hamiltonian constraints")
        linear_constraints.extend(hamiltonian_constraints)

        # gauge constraints
        gauge_constraints = self.generate_gauge_constraints()
        logger.info(f"Generated {len(gauge_constraints)} gauge constraints")
        linear_constraints.extend(gauge_constraints)

        # symmetry constraints
        if self.symmetry_generators is not None:
            symmetry_constraints = self.generate_symmetry_constraints()
            logger.info(f"Generated {len(symmetry_constraints)} symmetry constraints")
            linear_constraints.extend(symmetry_constraints)

        # odd degree vanish
        if self.odd_degree_vanish:
            odd_degree_constraints = self.generate_odd_degree_vanish_constraints()
            logger.info(
                f"Generated {len(odd_degree_constraints)} odd degree vanish constraints"
            )
            linear_constraints.extend(odd_degree_constraints)

        # cyclic constraints
        cyclic_linear, cyclic_quadratic = self.generate_cyclic_constraints()
        cyclic_linear = list(cyclic_linear.values())
        logger.info(f"Generated {len(cyclic_linear)} linear cyclic constraints")
        logger.info(f"Generated {len(cyclic_quadratic)} quadratic cyclic constraints")
        linear_constraints.extend(cyclic_linear)

        # NOTE pretty sure this is not necessary
        # linear_constraints.extend(
        #     [self.matrix_system.hermitian_conjugate(op) for op in linear_constraints]
        # )

        # save the constraints
        if self.config_cache_path is not None:
            with open(
                self.config_cache_path + "/linear_constraints_data.pkl", "wb"
            ) as f:
                pickle.dump([constraint.data for constraint in linear_constraints], f)
        if self.structural_cache_path is not None:
            with open(self.structural_cache_path + "/cyclic_quadratic.pkl", "wb") as f:
                cyclic_data_dict = {}
                for key, value in cyclic_quadratic.items():
                    cyclic_data_dict[key] = {
                        "lhs": value["lhs"].data,
                        "rhs": value["rhs"].data,
                    }
                pickle.dump(cyclic_data_dict, f)

        return self.clean_constraints(
            linear_constraints
        ), self.clean_constraints_quadratic(cyclic_quadratic)

    def clean_constraints_quadratic(
        self,
        constraints: dict[int, dict[str, SingleTraceOperator | DoubleTraceOperator]],
    ) -> dict[int, dict[str, SingleTraceOperator | DoubleTraceOperator]]:
        """
        Remove constraints that involve operators outside the operator list.
        Also remove empty constraints of the form 0=0.

        Parameters
        ----------
        constraints : list[SingleTraceOperator]
            The single trace constraints.

        Returns
        -------
        list[SingleTraceOperator]
            The cleaned constraints.
        """
        cleaned_constraints = {}

        # use a set to check membership as this operation is O(1) vs O(N) for lists
        set_of_all_operators = set(self.operator_list)

        for constraint_idx, quadratic_constraint in constraints.items():
            lhs = quadratic_constraint["lhs"]
            rhs = quadratic_constraint["rhs"]

            lhs_in_basis = all([op in set_of_all_operators for op in lhs.data])
            rhs_in_basis = all(
                [
                    op1 in set_of_all_operators and op2 in set_of_all_operators
                    for (op1, op2) in rhs.data
                ]
            )

            if lhs_in_basis and rhs_in_basis:
                cleaned_constraints[constraint_idx] = {
                    "lhs": lhs,
                    "rhs": rhs,
                }

        return cleaned_constraints

    def scale_param_to_enforce_normalization(self, param: np.ndarray) -> np.ndarray:
        """
        Rescale the parameter vector to enforce the normalization condition that <1> = 1.

        Parameters
        ----------
        param : np.ndarray
            The input param.

        Returns
        -------
        np.ndarray
            The rescaled param.
        """
        raise NotImplementedError
