import time
from copy import deepcopy

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

from FR3Env import getDataPath


class FR3MuJocoEnv:
    def __init__(self, render=True, xml_name="fr3", urdf_name="fr3"):
        package_directory = getDataPath()

        self.model = mujoco.MjModel.from_xml_path(
            package_directory + f"/robots/{xml_name}.xml"
        )
        self.data = mujoco.MjData(self.model)

        robot_URDF = package_directory + "/robots/{}.urdf".format(urdf_name)
        self.pin_robot = RobotWrapper.BuildFromURDF(robot_URDF, package_directory)

        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.render = True
        else:
            self.render = False

        self.model.opt.gravity[2] = -9.81
        self.jacobian_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        self.EE_FRAME_ID = self.pin_robot.model.getFrameId("fr3_hand_tcp")
        self.renderer = None

    def reset(self):
        self.q_nominal = np.array(
            [0.0, -0.785398163, 0.0, -2.35619449, 0.0, 1.57079632679, 0.785398163397]
        )

        for i in range(7):
            self.data.qpos[i] = self.q_nominal[i]

        self.data.qpos[7] = 0.0
        self.data.qpos[8] = 0.0

        q, dq = self.data.qpos[:9].copy(), self.data.qvel[:9].copy()
        self.update_pinocchio(q, dq)
        info = self.get_info(q, dq)

        return info

    def step(self, tau, finger_pos):
        frc_applied = np.append(tau, finger_pos)

        self.data.ctrl[:] = frc_applied
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

        q, dq = self.data.qpos[:9].copy(), self.data.qvel[:9].copy()
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

        # preprocessing is done in update_pinocchio()
        jacobian = self.pin_robot.getFrameJacobian(
            self.EE_FRAME_ID, self.jacobian_frame
        )

        # Get pseudo-inverse of frame Jacobian
        pinv_jac = np.linalg.pinv(jacobian)

        info = {
            "q": q,
            "dq": dq,
            "M(q)": M,
            "M(q)^{-1}": Minv,
            "nle": nle,
            "G": self.pin_robot.gravity(q),
            "J_EE": jacobian,
            "pJ_EE": pinv_jac,
            "R_EE": deepcopy(self.pin_robot.data.oMf[self.EE_FRAME_ID].rotation),
            "P_EE": deepcopy(self.pin_robot.data.oMf[self.EE_FRAME_ID].translation),
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

    def get_depth_image(self, camera=-1, scene_option=None):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model)
            self.renderer.enable_depth_rendering()

        self.renderer.update_scene(self.data, camera=camera, scene_option=scene_option)
        depth = self.renderer.render()

        # process depth image
        depth -= depth.min()
        depth = np.clip(depth, 0.0, 4.0) / 4.0
        # depth /= 2 * depth[depth <= 1].mean()
        pixels = 255 * np.clip(depth, 0, 1)

        return pixels

    def show_depth_img(self, pixels):
        plt.imshow(pixels.astype(np.uint8))
        plt.show()
