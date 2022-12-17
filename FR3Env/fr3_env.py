import copy
from typing import Optional

import numpy as np
import pinocchio as pin
import pybullet as p
import pybullet_data
from gymnasium import Env, spaces
from pinocchio.robot_wrapper import RobotWrapper

from FR3Env import getDataPath


class FR3Sim(Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(
        self, render_mode: Optional[str] = None, record_path=None, crude_model=False
    ):
        if render_mode == "human":
            self.client = p.connect(p.GUI)
            # Improves rendering performance on M1 Macs
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.client = p.connect(p.DIRECT)

        self.record_path = record_path

        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)

        # Load plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        # Load Franka Research 3 Robot
        if crude_model:
            model_name = "fr3_crude"
        else:
            model_name = "fr3"
        package_directory = getDataPath()
        robot_URDF = package_directory + "/robots/{}.urdf".format(model_name)
        urdf_search_path = package_directory + "/robots"
        p.setAdditionalSearchPath(urdf_search_path)
        self.robotID = p.loadURDF("{}.urdf".format(model_name), useFixedBase=True)

        # Build pin_robot
        self.robot = RobotWrapper.BuildFromURDF(robot_URDF, package_directory)

        # Get active joint ids
        self.active_joint_ids = [0, 1, 2, 3, 4, 5, 6, 10, 11]

        # Disable the velocity control on the joints as we use torque control.
        p.setJointMotorControlArray(
            self.robotID,
            self.active_joint_ids,
            p.VELOCITY_CONTROL,
            forces=np.zeros(9),
        )

        # Get number of joints
        self.n_j = p.getNumJoints(self.robotID)

        # End-effector frame id
        self.EE_FRAME_ID = 26

        # Get frame ID for grasp target
        self.jacobian_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

        # Set observation and action space
        obs_low_q = []
        obs_low_dq = []
        obs_high_q = []
        obs_high_dq = []
        _act_low = []
        _act_high = []

        for i in range(self.n_j):
            _joint_infos = p.getJointInfo(self.robotID, i)  # get info of each joint

            if _joint_infos[2] != p.JOINT_FIXED:
                obs_low_q.append(_joint_infos[8])
                obs_high_q.append(_joint_infos[9])
                obs_low_dq.append(-_joint_infos[11])
                obs_high_dq.append(_joint_infos[11])
                _act_low.append(-_joint_infos[10])
                _act_high.append(_joint_infos[10])

        obs_low = np.array(obs_low_q + obs_low_dq, dtype=np.float32)
        obs_high = np.array(obs_high_q + obs_high_dq, dtype=np.float32)
        act_low = np.array(_act_low, dtype=np.float32)
        act_high = np.array(_act_high, dtype=np.float32)

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        cameraDistance=1.4,
        cameraYaw=66.4,
        cameraPitch=-16.2
    ):
        super().reset(seed=seed)

        target_joint_angles = [
            0.0,
            -0.785398163,
            0.0,
            -2.35619449,
            0.0,
            1.57079632679,
            0.785398163397,
            0.001,
            0.001,
        ]

        self.q_nominal = np.array(target_joint_angles)

        for i, joint_ang in enumerate(target_joint_angles):
            p.resetJointState(self.robotID, self.active_joint_ids[i], joint_ang, 0.0)

        q, dq = self.get_state_update_pinocchio()
        info = self.get_info(q, dq)

        if self.record_path is not None:
            p.resetDebugVisualizerCamera(
                cameraDistance, cameraYaw, cameraPitch, [0.0, 0.0, 0.0]
            )

            self.loggingId = p.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4, self.record_path
            )

        return info

    def get_info(self, q, dq):
        """
        info contains:
        -------------------------------------
        q: joint position
        dq: joint velocity
        f(x): drift
        g(x): control influence matrix
        G: gravitational vector
        J_EE: end-effector Jacobian
        dJ_EE: time derivative of end-effector Jacobian
        pJ_EE: pseudo-inverse of end-effector Jacobian
        R_EE: end-effector rotation matrix
        P_EE: end-effector position vector
        """
        # Get Jacobian from grasp target frame
        # preprocessing is done in get_state_update_pinocchio()
        jacobian = self.robot.getFrameJacobian(self.EE_FRAME_ID, self.jacobian_frame)

        # Get pseudo-inverse of frame Jacobian
        pinv_jac = np.linalg.pinv(jacobian)

        dJ = pin.getFrameJacobianTimeVariation(
            self.robot.model, self.robot.data, self.EE_FRAME_ID, self.jacobian_frame
        )

        f, g, M, Minv, nle = self.get_dynamics(q, dq)

        info = {
            "q": q,
            "dq": dq,
            "f(x)": f,
            "g(x)": g,
            "M(q)": M,
            "M(q)^{-1}": Minv,
            "nle": nle,
            "G": self.robot.gravity(q),
            "J_EE": jacobian,
            "dJ_EE": dJ,
            "pJ_EE": pinv_jac,
            "R_EE": copy.deepcopy(self.robot.data.oMf[self.EE_FRAME_ID].rotation),
            "P_EE": copy.deepcopy(self.robot.data.oMf[self.EE_FRAME_ID].translation),
        }

        return info

    def get_dynamics(self, q, dq):
        """
        f.shape = (18, 1), g.shape = (18, 9)
        """
        Minv = pin.computeMinverse(self.robot.model, self.robot.data, q)
        M = self.robot.mass(q)
        nle = self.robot.nle(q, dq)

        f = np.vstack((dq[:, np.newaxis], -Minv @ nle[:, np.newaxis]))
        g = np.vstack((np.zeros((9, 9)), Minv))

        return f, g, M, Minv, nle

    def step(self, action):
        self.send_joint_command(action)
        p.stepSimulation()

        q, dq = self.get_state_update_pinocchio()
        info = self.get_info(q, dq)

        return info

    def close(self):
        if self.record_path is not None:
            p.stopStateLogging(self.loggingId)
        p.disconnect()

    def get_state(self):
        q = np.zeros(9)
        dq = np.zeros(9)

        for i, id in enumerate(self.active_joint_ids):
            _joint_state = p.getJointState(self.robotID, id)
            q[i], dq[i] = _joint_state[0], _joint_state[1]

        return q, dq

    def update_pinocchio(self, q, dq):
        self.robot.computeJointJacobians(q)
        self.robot.framesForwardKinematics(q)
        self.robot.centroidalMomentum(q, dq)

    def get_state_update_pinocchio(self):
        q, dq = self.get_state()
        self.update_pinocchio(q, dq)

        return q, dq

    def send_joint_command(self, tau):
        zeroGains = tau.shape[0] * (0.0,)

        p.setJointMotorControlArray(
            self.robotID,
            self.active_joint_ids,
            p.TORQUE_CONTROL,
            forces=tau,
            positionGains=zeroGains,
            velocityGains=zeroGains,
        )
