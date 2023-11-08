import argparse

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

from FR3Env.controller.waypoint_controller_hierarchical_proxqp import WaypointController
from FR3Env.fr3_env import FR3Sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--recordPath",
        help="path where the recording is saved",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-i",
        "--iterationNum",
        help="number of iterations of the simulation",
        type=int,
        default=100000,
    )
    args = parser.parse_args()

    p_ends = [
        np.array([[0.4], [0.4], [0.2]]),
        np.array([[0.4], [-0.4], [0.2]]),
        np.array([[0.3], [0.0], [0.5]]),
    ]
    p_end_id = 0

    dt = 1 / 1000
    env = FR3Sim(render_mode="human", record_path=args.recordPath)
    p.setTimeStep(dt)

    controller = WaypointController()
    info = env.reset()

    p_end = p_ends[p_end_id]
    p_end_id = (p_end_id + 1) % len(p_ends)

    # get initial rotation and position
    q, dq, R_start, _p_start = info["q"], info["dq"], info["R_EE"], info["P_EE"]
    p_start = _p_start[:, np.newaxis]

    # Get target orientation based on initial orientation
    _R_end = (
        R.from_euler("x", 0, degrees=True).as_matrix()
        @ R.from_euler("z", 0, degrees=True).as_matrix()
        @ R_start
    )
    R_end = R.from_matrix(_R_end).as_matrix()

    controller.start(p_start, p_end, R_start, R_end, 30.0)
    q_min = env.observation_space.low[:9][:, np.newaxis]
    q_max = env.observation_space.high[:9][:, np.newaxis]
    q_nominal = env.q_nominal[:, np.newaxis]

    for i in range(args.iterationNum):
        # Get end-effector position
        p_current = info["P_EE"][:, np.newaxis]

        # Get end-effector orientation
        R_current = info["R_EE"]

        # Get Jacobian from grasp target frame
        jacobian = info["J_EE"]

        # Get pseudo-inverse of frame Jacobian
        pinv_jac = info["pJ_EE"]

        # Get gravitational vector
        G = info["G"][:, np.newaxis]

        if i == 0:
            _dt = 0
        else:
            _dt = dt

        q_target, error = controller.update(
            q,
            dq,
            p_current,
            R_current,
            pinv_jac,
            jacobian,
            G,
            _dt,
            q_min,
            q_max,
            q_nominal,
        )

        if controller.status == controller.WAIT:
            p_end = p_ends[p_end_id]
            p_end_id = (p_end_id + 1) % len(p_ends)

            # get initial rotation and position
            dq, R_start, _p_start = info["dq"], info["R_EE"], info["P_EE"]
            p_start = _p_start[:, np.newaxis]

            # Get target orientation based on initial orientation
            _R_end = (
                R.from_euler("x", 0, degrees=True).as_matrix()
                @ R.from_euler("z", 0, degrees=True).as_matrix()
                @ R_start
            )
            R_end = R.from_matrix(_R_end).as_matrix()

            controller.start(p_start, p_end, R_start, R_end, 30.0)

        if i % 500 == 0:
            print("Iter {:.2e} \t error: {:.2e}".format(i, error))

        # Compute controller
        Δq = (q_target - q)[:, np.newaxis]
        Kp = 10 * np.eye(9)
        τ = Kp @ Δq - 1.0 * dq[:, np.newaxis] + G

        # Set control for the two fingers to zero
        τ[-1] = 1.0 * (0.01 - q[-1]) + 0.1 * (0 - dq[-1])
        τ[-2] = 1.0 * (0.01 - q[-2]) + 0.1 * (0 - dq[-2])

        # Send joint commands to motor
        info = env.step(τ)
        q, dq = info["q"], info["dq"]

    env.close()


if __name__ == "__main__":
    main()
