import numpy as np
import numpy.linalg as LA
from scipy.spatial.transform import Rotation as R


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

    def update(self, dq, p_current, R_current, pinv_jac, G, duration):
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
            v_target = np.zeros((3, 1))
            R_target = self.R_end
            ω_target = np.zeros((3,))
        elif self.status == self.UPDATE:
            # Compute α and dα
            alpha, dalpha = alpha_func(self.clock, T=self.movement_duration)
            # Compute p_target
            p_target = self.p_start + alpha * (self.p_end - self.p_start)
            # Compute v_target
            v_target = dalpha * (self.p_end - self.p_start)
            # Compute R_target
            θ_t = alpha * self.θ_error
            R_target = R.from_rotvec(self.ω_error * θ_t).as_matrix() @ self.R_start
            # Compute ω_target
            ω_target = dalpha * self.ω_error * self.θ_error

        # Compute error rotation matrix
        R_err = R_target @ R_current.T

        # Orientation error in axis-angle form
        rotvec_err = R.from_matrix(R_err).as_rotvec()

        # Compute controller
        Δx = np.zeros((6, 1))
        Δx[:3] = p_target - p_current
        Δx[3:] = rotvec_err[:, np.newaxis]
        Δq = pinv_jac @ Δx

        dx = np.zeros((6, 1))
        dx[:3] = v_target
        dx[3:] = ω_target[:, np.newaxis]
        Δdq = pinv_jac @ dx - dq[:, np.newaxis]

        Kp = 10 * np.eye(9)
        Kd = 0.1 * np.eye(9)

        τ = Kp @ Δq + Kd @ Δdq + G

        # Set control for the two fingers to zero
        τ[-1] = 0.0
        τ[-2] = 0.0

        return τ, error
