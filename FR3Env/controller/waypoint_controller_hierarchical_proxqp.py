import numpy as np
import numpy.linalg as LA
from scipy.spatial.transform import Rotation as R

from FR3Env.controller.proxsuite_solve import ProxSuiteSolver


def axis_angle_from_rot_mat(rot_mat):
    rotation = R.from_matrix(rot_mat)
    axis_angle = rotation.as_rotvec()
    angle = LA.norm(axis_angle)
    axis = axis_angle / angle

    return axis, angle


def alpha_func(t, T=5.0):
    if t <= T:
        alpha = np.sin((np.pi / 4) * (1 - np.cos(np.pi * t / T)))
        dalpha = (
            ((np.pi ** 2) / (4 * T))
            * np.cos((np.pi / 4) * (1 - np.cos(np.pi * t / T)))
            * np.sin(np.pi * t / T)
        )
    else:
        alpha = 1.0
        dalpha = 0.0

    return alpha, dalpha


class WaypointController:
    START = 1
    UPDATE = 2
    WAIT = 3

    def __init__(self) -> None:
        self.p_start = None
        self.p_end = None
        self.R_start = None
        self.R_end = None
        self.movement_duration = None
        self.clock = 0.0
        self.qp_solver = ProxSuiteSolver(9)  # 9 represents there are 9 joints

        self.status = None

    def start(self, p_start, p_end, R_start, R_end, movement_duration):
        # Set status to START
        self.status = self.START

        # Start time counter
        self.clock = 0.0

        # Get p_start, R_start
        self.p_start = p_start
        self.R_start = R_start

        # Set p_end, R_end
        self.p_end = p_end
        self.R_end = R_end
        self.movement_duration = movement_duration

        # Compute R_error, ω_error, θ_error
        self.R_error = R_end @ R_start.T
        self.ω_error, self.θ_error = axis_angle_from_rot_mat(self.R_error)

        # Set status to UPDATE
        self.status = self.UPDATE

    def update(
        self,
        q,
        dq,
        p_current,
        R_current,
        pinv_jac,
        jacobian,
        G,
        duration,
        q_min,
        q_max,
        q_nominal,
    ):
        self.clock += duration

        # Compute end-effector pose error
        _R_err = self.R_end @ R_current.T
        _rotvec_err = R.from_matrix(_R_err).as_rotvec()
        error_vec = np.concatenate([_rotvec_err, (self.p_end - p_current)[:, 0]])
        error = LA.norm(error_vec)

        if error <= 1e-3:
            self.status = self.WAIT

        if self.status == self.WAIT:
            p_target = self.p_end
            R_target = self.R_end
        elif self.status == self.UPDATE:
            # Compute α and dα
            alpha, dalpha = alpha_func(self.clock, T=self.movement_duration)
            # Compute p_target
            p_target = self.p_start + alpha * (self.p_end - self.p_start)
            # Compute R_target
            θ_t = alpha * self.θ_error
            R_target = R.from_rotvec(self.ω_error * θ_t).as_matrix() @ self.R_start

        # Compute error rotation matrix
        R_err = R_target @ R_current.T

        # Orientation error in axis-angle form
        rotvec_err = R.from_matrix(R_err).as_rotvec()

        # Compute controller
        Δx = np.zeros((6, 1))
        Δx[:3] = p_target - p_current
        Δx[3:] = rotvec_err[:, np.newaxis]

        # Solve for q_target
        params = {
            "Jacobian": jacobian,
            "q_measured": q[:, np.newaxis],
            "p_error": Δx,
            "q_min": q_min,
            "q_max": q_max,
            "nullspace_proj": np.eye(9) - pinv_jac @ jacobian,
            "q_nominal": q_nominal,
        }

        self.qp_solver.solve(params)
        q_target = self.qp_solver.qp.results.z

        return q_target, error
