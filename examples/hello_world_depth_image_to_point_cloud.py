import numpy as np
import pinocchio as pin
from ndcurves import SE3Curve
import open3d as o3d

from FR3Env.controller import DiffIK
from FR3Env.fr3_mj_env import FR3MuJocoEnv


def main():
    env = FR3MuJocoEnv(xml_name="fr3_on_table")
    info = env.reset()

    T_init = pin.SE3(info["R_EE"].copy(), info["P_EE"].copy())
    R_end = np.diag([1.0, -1.0, -1.0])
    p_end = np.array([[0.4], [0.3], [0.3]])
    T_end = pin.SE3(R_end, p_end)
    t_init = 0.0
    t_end = 5.0
    curve = SE3Curve(T_init, T_end, t_init, t_end)

    controller = DiffIK()

    for i in range(12000):
        if not env.viewer.is_running():
            break

        T = pin.SE3(info["R_EE"].copy(), info["P_EE"].copy())

        t = np.clip(i * 2e-3, 0.0, t_end)

        v_error = pin.log6(curve(t) @ T.inverse().homogeneous).vector
        v_des = curve.derivate(t, 1) + 1.0 * v_error
        Δdq = controller(v_des, info["q"], info["J_EE"])

        _tau = 10.0 * Δdq + info["G"] - 6.0 * info["dq"]
        tau, finger_pos = _tau[:7], 0.0

        info = env.step(tau, finger_pos)

        # pixels = env.get_depth_image(camera="franka_camera")
        # depth = env.renderer.render()

        if i == 4000:
            T_init = pin.SE3(info["R_EE"].copy(), info["P_EE"].copy())
            p_end = np.array([[0.4], [-0.3], [0.3]])
            T_end = pin.SE3(R_end, p_end)
            t_init = i * 2e-3
            t_end = i * 2e-3 + 5.0
            curve = SE3Curve(T_init, T_end, t_init, t_end)

        if i == 8000:
            T_init = pin.SE3(info["R_EE"].copy(), info["P_EE"].copy())
            p_end = np.array([[0.3], [0.0], [0.5]])
            T_end = pin.SE3(R_end, p_end)
            t_init = i * 2e-3
            t_end = i * 2e-3 + 5.0
            curve = SE3Curve(T_init, T_end, t_init, t_end)

    env.close()


if __name__ == "__main__":
    main()
