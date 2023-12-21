import numpy as np
import pinocchio as pin
from ndcurves import SE3Curve

from FR3Env.controller import DiffIK
from FR3Env.fr3_mj_env import FR3MuJocoEnv
import time


def main():
    env = FR3MuJocoEnv(xml_name="fr3_with_block")
    info = env.reset()

    controller = DiffIK()

    FSM = {
        "PRE-GRASP": (np.array([[0.4], [0.4], [0.1], [0.04]]), 5.0),
        "GRASP": (np.array([[0.4], [0.4], [0.0125], [0.04]]), 2.0),
        "CLOSE-GRASP": (np.array([[0.4], [0.4], [0.013], [0.000]]), 2.0),
        "POST-GRASP": (np.array([[0.4], [0.4], [0.1], [0.000]]), 2.0),
        "PRE-PLACE": (np.array([[0.4], [-0.4], [0.1], [0.000]]), 5.0),
        "PLACE": (np.array([[0.4], [-0.4], [0.013], [0.000]]), 2.0),
        "OPEN-GRASP": (np.array([[0.4], [-0.4], [0.013], [0.04]]), 2.0),
        "POST-PLACE": (np.array([[0.4], [-0.4], [0.1], [0.04]]), 2.0),
        "RETURN": (np.array([[0.3], [0.0], [0.5], [0.00]]), 10.0),
    }

    task_list = [
        "PRE-GRASP",
        "GRASP",
        "CLOSE-GRASP",
        "POST-GRASP",
        "PRE-PLACE",
        "PLACE",
        "OPEN-GRASP",
        "POST-PLACE",
        "RETURN",
    ]
    task_id = 0
    target, duration = FSM[task_list[task_id]]
    t = 0.0

    T_init = pin.SE3(info["R_EE"].copy(), info["P_EE"].copy())
    R_end = np.diag([1.0, -1.0, -1.0])
    p_end = target[:3, :]
    T_end = pin.SE3(R_end, p_end)
    t_init = t
    t_end = t + duration
    curve = SE3Curve(T_init, T_end, t_init, t_end)

    for i in range(40000):
        if not env.viewer.is_running():
            break

        T = pin.SE3(info["R_EE"].copy(), info["P_EE"].copy())

        t = np.clip(i * 2e-3, 0.0, t_end)

        v_error = pin.log6(curve(t) @ T.inverse().homogeneous).vector
        v_des = curve.derivate(t, 1) + 20.0 * v_error
        Δdq = controller(v_des, info["q"], info["J_EE"])

        _tau = 10.0 * Δdq + info["G"] - 20.0 * info["dq"]
        tau, finger_pos = _tau[:7], target[3, 0]

        info = env.step(tau, finger_pos)

        error = np.linalg.norm(target[:3, 0] - info["P_EE"])

        if t >= t_end and error <= 0.005:
            task_id = task_id + 1

            if task_id >= len(task_list):
                break

            target, duration = FSM[task_list[task_id]]

            T_init = pin.SE3(info["R_EE"].copy(), info["P_EE"].copy())
            R_end = np.diag([1.0, -1.0, -1.0])
            p_end = target[:3, 0]
            T_end = pin.SE3(R_end, p_end)
            t_init = t
            t_end = t + duration
            curve = SE3Curve(T_init, T_end, t_init, t_end)

    env.close()


if __name__ == "__main__":
    main()
