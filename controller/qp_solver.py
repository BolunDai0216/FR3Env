import casadi as ca


class QPSolver:
    def __init__(self, nq):
        self.nq = nq  # number of joints

        self.opti = ca.Opti()
        self.variables = {}
        self.parameters = {}
        self.costs = {}

        self.setup_opt()

    def setup_opt(self):
        self.set_parameters()
        self.set_variables()

        self.set_constraints()
        self.set_cost()

        # options to make it not output solver stats
        sol_options = {
            "verbose": False,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.acceptable_constr_viol_tol": 1e-8,
            "ipopt.constr_viol_tol": 1e-8,
        }
        self.opti.solver("ipopt", sol_options)

    def solve(self, params, warm_start_guess=None):
        self.set_parameters_value(params)
        sol = self.opti.solve()

        return sol

    def set_constraints(self):
        self.set_joint_limit_constraint()

    def set_cost(self):
        self.tracking_cost()
        self.joint_centering_cost()

        # Sum individual costs into final cost
        cost = 0
        for _cost_name in self.costs:
            cost += self.costs[_cost_name]

        # set cost to optimization problem
        self.opti.minimize(cost)

    def set_parameters(self):
        self.parameters["Jacobian"] = self.opti.parameter(6, self.nq)
        self.parameters["q_measured"] = self.opti.parameter(self.nq, 1)
        self.parameters["p_error"] = self.opti.parameter(6, 1)
        self.parameters["q_min"] = self.opti.parameter(self.nq, 1)
        self.parameters["q_max"] = self.opti.parameter(self.nq, 1)
        self.parameters["nullspace_proj"] = self.opti.parameter(self.nq, self.nq)
        self.parameters["q_nominal"] = self.opti.parameter(self.nq, 1)

    def set_variables(self):
        self.variables["q_target"] = self.opti.variable(self.nq, 1)

    def set_parameters_value(self, params):
        """
        set the value of parameters
        ---------------------------
        params: a dict storing values of the parameters
        """
        for param_name in params:
            self.opti.set_value(self.parameters[param_name], params[param_name])

    def tracking_cost(self):
        self.costs["tracking_cost"] = ca.mtimes(
            (
                self.parameters["Jacobian"]
                @ (self.variables["q_target"] - self.parameters["q_measured"])
                - self.parameters["p_error"]
            ).T,
            (
                self.parameters["Jacobian"]
                @ (self.variables["q_target"] - self.parameters["q_measured"])
                - self.parameters["p_error"]
            ),
        )

    def joint_centering_cost(self):
        self.costs["joint_centering"] = ca.mtimes(
            (
                self.parameters["nullspace_proj"]
                @ (self.variables["q_target"] - self.parameters["q_nominal"])
            ).T,
            (
                self.parameters["nullspace_proj"]
                @ (self.variables["q_target"] - self.parameters["q_nominal"])
            ),
        )

    def set_joint_limit_constraint(self):
        self.opti.subject_to(self.parameters["q_min"] <= self.variables["q_target"])
        self.opti.subject_to(self.variables["q_target"] <= self.parameters["q_max"])
