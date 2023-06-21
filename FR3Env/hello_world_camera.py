import numpy as np
import numpy.linalg as LA
from scipy.spatial.transform import Rotation as R

from FR3Env.fr3_with_camera_env import FR3CameraSim


def alpha_func(t, T=5.0):
    if t <= T:
        alpha = np.sin((np.pi / 4) * (1 - np.cos(np.pi * t / T)))
        dalpha = (
            ((np.pi**2) / (4 * T))
            * np.cos((np.pi / 4) * (1 - np.cos(np.pi * t / T)))
            * np.sin(np.pi * t / T)
        )
    else:
        alpha = 1.0
        dalpha = 0.0

    return alpha, dalpha


def axis_angle_from_rot_mat(rot_mat):
    rotation = R.from_matrix(rot_mat)
    axis_angle = rotation.as_rotvec()
    angle = LA.norm(axis_angle)
    axis = axis_angle / angle

    return axis, angle


def main():
    env = FR3CameraSim(render_mode="human")
    info = env.reset()

    p_end = np.array([[0.2], [-0.4], [0.2]])

    # get initial rotation and position
    q, dq, R_start, _p_start = info["q"], info["dq"], info["R_EE"], info["P_EE"]
    p_start = _p_start[:, np.newaxis]

    # Get target orientation based on initial orientation
    _R_end = (
        R.from_euler("x", 0, degrees=True).as_matrix()
        @ R.from_euler("z", 90, degrees=True).as_matrix()
        @ R_start
    )
    R_end = R.from_matrix(_R_end).as_matrix()
    R_error = R_end @ R_start.T
    axis_error, angle_error = axis_angle_from_rot_mat(R_error)

    for i in range(10000):
        # Get simulation time
        sim_time = i * (1 / 240)

        # Compute α and dα
        alpha, dalpha = alpha_func(sim_time, T=30.0)

        # Compute p_target
        p_target = p_start + alpha * (p_end - p_start)

        # Compute v_target
        v_target = dalpha * (p_end - p_start)

        # Compute R_target
        theta_t = alpha * angle_error
        R_target = R.from_rotvec(axis_error * theta_t).as_matrix() @ R_start

        # Compute ω_target
        ω_target = dalpha * axis_error * angle_error

        # Get end-effector position
        p_current = info["P_EE"][:, np.newaxis]

        # Get end-effector orientation
        R_current = info["R_EE"]

        # Error rotation matrix
        R_err = R_target @ R_current.T

        # Orientation error in axis-angle form
        rotvec_err = R.from_matrix(R_err).as_rotvec()

        # Get pseudo-inverse of frame Jacobian
        pinv_jac = info["pJ_EE"]

        # Compute controller
        delta_x = np.zeros((6, 1))
        delta_x[:3] = p_target - p_current
        delta_x[3:] = rotvec_err[:, np.newaxis]
        delta_q = pinv_jac @ delta_x

        dx = np.zeros((6, 1))
        dx[:3] = v_target
        dx[3:] = ω_target[:, np.newaxis]
        delta_dq = pinv_jac @ dx - dq[:, np.newaxis]

        Kp = 10 * np.eye(9)
        Kd = 0.1 * np.eye(9)

        tau = Kp @ delta_q + Kd @ delta_dq + info["G"][:, np.newaxis]

        # Set control for the two fingers to zero
        tau[-1] = 0.0
        tau[-2] = 0.0

        # Send joint commands to motor
        if i % 50 == 0:
            info = env.step(tau, return_image=True)
        else:
            info = env.step(tau)

        q, dq = info["q"], info["dq"]

        if i % 500 == 0:
            print(
                "Iter {:.2e} \t ǁeₒǁ₂: {:.2e} \t ǁeₚǁ₂: {:.2e}".format(
                    i, LA.norm(rotvec_err), LA.norm(p_target - p_current)
                ),
            )

    env.close()


if __name__ == "__main__":
    main()
