import numpy as np
import numpy.linalg as LA
from scipy.spatial.transform import Rotation as R

from FR3Env.controller.qp_solver import QPSolver
from FR3Env.fr3_env import FR3Sim
from FR3Env.hello_world import alpha_func, axis_angle_from_rot_mat


def main():
    env = FR3Sim(render_mode="human")
    qp_solver = QPSolver(9)  # 9 represents there are 9 joints

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

    for i in range(20000):
        # Get simulation time
        sim_time = i * (1 / 240)

        if i % 12 == 0:
            # Compute α and dα
            alpha, dalpha = alpha_func(sim_time, T=30.0)

            # Compute p_target
            p_target = p_start + alpha * (p_end - p_start)

            # Compute R_target
            theta_t = alpha * angle_error
            R_target = R.from_rotvec(axis_error * theta_t).as_matrix() @ R_start

            # Get end-effector position
            p_current = info["P_EE"][:, np.newaxis]

            # Get end-effector orientation
            R_current = info["R_EE"]

            # Error rotation matrix
            R_err = R_target @ R_current.T

            # Orientation error in axis-angle form
            rotvec_err = R.from_matrix(R_err).as_rotvec()

            # Get Jacobian from grasp target frame
            jacobian = info["J_EE"]

            # Get pseudo-inverse of frame Jacobian
            pinv_jac = info["pJ_EE"]

            # Compute controller
            p_error = np.zeros((6, 1))
            p_error[:3] = p_target - p_current
            p_error[3:] = rotvec_err[:, np.newaxis]

            # Solve for q_target
            params = {
                "Jacobian": jacobian,
                "q_measured": q[:, np.newaxis],
                "p_error": p_error,
                "q_min": env.observation_space.low[:9][:, np.newaxis],
                "q_max": env.observation_space.high[:9][:, np.newaxis],
                "nullspace_proj": np.eye(9) - pinv_jac @ jacobian,
                "q_nominal": env.q_nominal[:, np.newaxis],
            }

            sol = qp_solver.solve(params)
            q_target = sol.value(qp_solver.variables["q_target"])

        # Compute delta_q
        delta_q = (q_target - q)[:, np.newaxis]

        # Computer tau
        Kp = 10 * np.eye(9)
        tau = Kp @ delta_q - 0.1 * dq[:, np.newaxis] + info["G"][:, np.newaxis]

        # Set control for the two fingers to zero
        tau[-1] = 0.0
        tau[-2] = 0.0

        # Send joint commands to motor
        info = env.step(tau)
        q, dq = info["q"], info["dq"]

        # compute _rotvec_err
        _R_err = R_end @ R_current.T
        _rotvec_err = R.from_matrix(_R_err).as_rotvec()

        if i % 500 == 0:
            print(
                "Iter {:.2e} \t ǁeₒǁ₂: {:.2e} \t ǁeₚǁ₂: {:.2e}".format(
                    i, LA.norm(_rotvec_err), LA.norm(p_end - p_current)
                ),
            )

    env.close()


if __name__ == "__main__":
    main()
