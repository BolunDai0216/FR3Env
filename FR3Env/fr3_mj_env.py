import time

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

from FR3Env import getDataPath


class FR3MuJocoEnv:
    def __init__(self, render=True, urdf_name="fr3"):
        package_directory = getDataPath()

        self.model = mujoco.MjModel.from_xml_path(package_directory + "/robots/fr3.xml")
        self.data = mujoco.MjData(self.model)

        robot_URDF = package_directory + "/robots/{}.urdf".format(urdf_name)
        self.pin_robot = RobotWrapper.BuildFromURDF(robot_URDF, package_directory)

        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.render = True
        else:
            self.render = False

    def reset(self):
        self.q_nominal = target = np.array(
            [0.0, -0.785398163, 0.0, -2.35619449, 0.0, 1.57079632679, 0.785398163397]
        )

        for i in range(7):
            self.data.qpos[i] = target[i]

        self.data.qpos[7] = 0.0
        self.data.qpos[8] = 0.0

        q, dq = self.data.qpos.copy(), self.data.qvel.copy()
        self.update_pinocchio(q, dq)
        info = self.get_info(q, dq)

        return info

    def step(self, tau, finger_pos):
        finger_control = finger_pos * np.ones(2) - self.data.qpos[7:]
        frc_applied = np.append(tau, finger_control)

        self.data.qfrc_applied = frc_applied
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

        q, dq = self.data.qpos.copy(), self.data.qvel.copy()
        self.update_pinocchio(q, dq)
        info = self.get_info(q, dq)

        return info

    def close(self):
        self.viewer.close()

    def sleep(self, start_time):
        time_until_next_step = self.model.opt.timestep - (time.time() - start_time)

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    def update_pinocchio(self, q, dq):
        self.pin_robot.computeJointJacobians(q)
        self.pin_robot.framesForwardKinematics(q)
        self.pin_robot.centroidalMomentum(q, dq)

    def get_info(self, q, dq):
        M, Minv, nle = self.get_dynamics(q, dq)

        info = {
            "q": q,
            "dq": dq,
            "M(q)": M,
            "M(q)^{-1}": Minv,
            "nle": nle,
            "G": self.pin_robot.gravity(q),
        }

        return info

    def get_dynamics(self, q, dq):
        """
        f.shape = (18, 1), g.shape = (18, 9)
        """
        Minv = pin.computeMinverse(self.pin_robot.model, self.pin_robot.data, q)
        M = self.pin_robot.mass(q)
        nle = self.pin_robot.nle(q, dq)

        return M, Minv, nle
