import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

from controller.waypoint_controller_hierarchical import WaypointController
from fr3_env import FR3Env


def main():
    p_ends = [
        np.array([[0.4], [0.4], [0.2]]),
        np.array([[0.4], [-0.4], [0.2]]),
        np.array([[0.3], [0.0], [0.5]]),
    ]
    p_end_id = 0

    env = FR3Env(render_mode="human")
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

    update_interval = 10

    for i in range(60000):
        if i % update_interval == 0:
            # Get end-effector position
            p_current = info["P_EE"][:, np.newaxis]

            # Get end-effector orientation
            R_current = info["R_EE"]

            # Get frame ID for grasp target
            jacobian_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

            # Get Jacobian from grasp target frame
            # preprocessing is done in get_state_update_pinocchio()
            jacobian = env.robot.getFrameJacobian(env.EE_FRAME_ID, jacobian_frame)

            # Get pseudo-inverse of frame Jacobian
            pinv_jac = np.linalg.pinv(jacobian)

            # Get gravitational vector
            G = info["G"][:, np.newaxis]

            if i == 0:
                dt = 0
            else:
                dt = 1 / (240 / update_interval)

            q_target, error = controller.update(
                q,
                dq,
                p_current,
                R_current,
                pinv_jac,
                jacobian,
                G,
                dt,
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
