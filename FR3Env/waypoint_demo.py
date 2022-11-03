import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

from FR3Env.controller.waypoint_controller import WaypointController
from FR3Env.fr3_env import FR3Sim


def main():
    p_ends = [
        np.array([[0.4], [0.4], [0.2]]),
        np.array([[0.4], [-0.4], [0.2]]),
    ]
    p_end_id = 0

    env = FR3Sim(render_mode="human")
    controller = WaypointController()
    info = env.reset()

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

    for i in range(60000):
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
            dt = 1 / 240

        tau, error = controller.update(dq, p_current, R_current, pinv_jac, G, dt)

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

        # Send joint commands to motor
        info = env.step(tau)
        dq = info["dq"]

    env.close()


if __name__ == "__main__":
    main()
