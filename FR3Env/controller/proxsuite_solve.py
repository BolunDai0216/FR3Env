import numpy as np
import proxsuite


class ProxSuiteSolver:
    def __init__(self, nq):
        self.n = nq
        self.n_eq = 0
        self.n_ieq = nq
        self.qp = proxsuite.proxqp.dense.QP(self.n, self.n_eq, self.n_ieq)
        self.initialized = False

    def solve(self, params):
        H, g, A, b, C, u, l = self.compute_params(params)

        if not self.initialized:
            self.qp.init(H, g, A, b, C, l, u)
            self.qp.settings.eps_abs = 1.0e-6
            self.initialized = True
        else:
            self.qp.update(H, g, A, b, C, l, u)

        self.qp.solve()

    def compute_params(self, params):
        H = 2 * (
            params["Jacobian"].T @ params["Jacobian"]
            + params["nullspace_proj"].T @ params["nullspace_proj"]
        )

        g = (
            -2
            * (
                params["q_measured"].T @ params["Jacobian"].T @ params["Jacobian"]
                + params["p_error"].T @ params["Jacobian"]
                + params["q_nominal"].T
                @ params["nullspace_proj"].T
                @ params["nullspace_proj"]
            )[0, :]
        )

        A = np.zeros((self.n_eq, 9))
        b = np.zeros((self.n_eq,))
        C = np.eye(self.n_ieq)
        u = params["q_max"][:, 0]
        l = params["q_min"][:, 0]

        return H, g, A, b, C, l, u
