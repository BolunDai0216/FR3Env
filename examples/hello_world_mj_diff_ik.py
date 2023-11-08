import time

import numpy as np
from ndcurves import SE3Curve
import pinocchio as pin

from FR3Env.controller import DiffIK
from FR3Env.fr3_mj_env import FR3MuJocoEnv


def main():
    env = FR3MuJocoEnv()
    info = env.reset()

    T_init = pin.SE3(info["R_EE"], info["P_EE"])
    # R_end = pin.rpy.rpyToMatrix(0.0, 0.0, 0.0)
    # p_end = np.array([[0.3], [0.0], [0.5]])
    # T_end = pin.SE3(R_end, p_end)
    T_end = pin.SE3(info["R_EE"], info["P_EE"])
    t_init = 0.0
    t_end = 10.0
    curve = SE3Curve(T_init, T_end, t_init, t_end)

    controller = DiffIK()

    for i in range(10000):
        if not env.viewer.is_running():
            break

        T = pin.SE3(info["R_EE"], info["P_EE"])

        t = i * env.model.opt.timestep

        if t <= t_end:
            V = pin.log6(T.inverse().homogeneous @ curve(t)).vector
            V_des = 1.0 * V + curve.derivate(t, 1)
        else:
            # V = pin.log6(T.inverse().homogeneous @ T_end).vector
            # V_des = 1.0 * V + np.zeros((6,))
            V_des = np.zeros((6,))

        dq_des = controller(V_des, info["q"], info["J_EE"])
        q_des = info["q"] + dq_des * env.model.opt.timestep

        # print(dq_des[:7])

        _tau = (
            1.0 * (q_des[:7] - info["q"][:7])
            + 0.1 * (dq_des[:7] - info["dq"][:7])
            + info["G"][:7]
        )
        tau, finger_pos = _tau[:7], 0.0

        info = env.step(tau, finger_pos)
        time.sleep(1e-3)

    env.close()


if __name__ == "__main__":
    main()
