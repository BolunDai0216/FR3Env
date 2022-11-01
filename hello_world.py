from pdb import set_trace

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

from fr3_env import FR3Env


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
    env = FR3Env(render_mode="human")
    info = env.reset()

    p_end = np.array([[0.2], [-0.4], [0.2]])

    # get initial rotation and position
    dq, R_start, _p_start = info["dq"], info["R_EE"], info["P_EE"]
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

    v_targets = []
    ω_targets = []
    kd_taus = []

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

        # Get frame ID for grasp target
        jacobian_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

        # Get Jacobian from grasp target frame
        # preprocessing is done in get_state_update_pinocchio()
        jacobian = env.robot.getFrameJacobian(env.EE_FRAME_ID, jacobian_frame)

        # Get pseudo-inverse of frame Jacobian
        pinv_jac = np.linalg.pinv(jacobian)

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
        Kd = 0.0 * np.eye(9)

        tau = Kp @ delta_q + Kd @ delta_dq + info["G"][:, np.newaxis]

        # Set control for the two fingers to zero
        tau[-1] = 0.0
        tau[-2] = 0.0

        # Send joint commands to motor
        info = env.step(tau)
        dq = info["q"]

        if i % 500 == 0:
            print(
                "Iter {:.2e} \t ǁeₒǁ₂: {:.2e} \t ǁeₚǁ₂: {:.2e}".format(
                    i, LA.norm(rotvec_err), LA.norm(p_target - p_current)
                ),
            )

        v_targets.append(v_target)
        ω_targets.append(ω_target[:, np.newaxis])
        kd_taus.append(Kd @ delta_dq)

    env.close()

    # v_arr = np.concatenate(v_targets, axis=1)
    # ω_arr = np.concatenate(ω_targets, axis=1)
    # kd_arr = np.concatenate(kd_taus, axis=1)

    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # for i in range(3):
    #     axs[0].plot(v_arr[i, :])
    #     axs[1].plot(ω_arr[i, :])

    # for i in range(9):
    #     axs[2].plot(kd_arr[i, :])

    # plt.show()


if __name__ == "__main__":
    main()
