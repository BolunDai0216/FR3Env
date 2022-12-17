import copy
from typing import Optional

import numpy as np
import pinocchio as pin
import pybullet as p
import pybullet_data
from gymnasium import Env, spaces
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation

from FR3Env import getDataPath


class FR3CrudeSim(Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(
        self, render_mode: Optional[str] = None, record_path=None, crude_model=True
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

        # frame ids
        self.FR3_LINK5_FRAME_ID = 12
        self.FR3_LINK6_FRAME_ID = 14
        self.FR3_LINK7_FRAME_ID = 16
        self.FR3_HAND_FRAME_ID = 20
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
        cameraPitch=-16.2,
        lookat=[0.0, 0.0, 0.0]
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
            p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, lookat)

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
        R_LINK5_1: rotation matrix of the first part of the 5th link
        P_LINK5_1: postion vector of the first part of the 5th link
        q_LINK5_1: quaternion of the first part of the 5th link [x, y, z, w]
        R_LINK5_2: rotation matrix of the second part of the 5th link
        P_LINK5_2: postion vector of the second part of the 5th link
        q_LINK5_2: quaternion of the second part of the 5th link [x, y, z, w]
        R_LINK6: rotation matrix of the 6th link
        P_LINK6: postion vector of the 6th link
        q_LINK6: quaternion of the 6th link [x, y, z, w]
        R_LINK7: rotation matrix of the 7th link
        P_LINK7: postion vector of the 7th link
        q_LINK7: quaternion of the 7th link [x, y, z, w]
        R_HAND: rotation matrix of the hand
        P_HAND: position vector of the hand
        q_HAND: quaternion of the hand [x, y, z, w]
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

        # compute the position and rotation of the crude models
        p_link5_1, R_link5_1, q_LINK5_1 = self.compute_crude_location(
            np.eye(3), np.array(([0.0], [0.0], [-0.26])), self.FR3_LINK5_FRAME_ID
        )

        p_link5_2, R_link5_2, q_LINK5_2 = self.compute_crude_location(
            np.eye(3), np.array(([0.0], [0.08], [-0.13])), self.FR3_LINK5_FRAME_ID
        )

        p_link6, R_link6, q_LINK6 = self.compute_crude_location(
            np.eye(3), np.array(([0.0], [0.0], [-0.03])), self.FR3_LINK6_FRAME_ID
        )

        p_link7, R_link7, q_LINK7 = self.compute_crude_location(
            np.eye(3), np.array(([0.0], [0.0], [0.01])), self.FR3_LINK7_FRAME_ID
        )

        p_hand, R_hand, q_HAND = self.compute_crude_location(
            np.eye(3), np.array(([0.0], [0.0], [0.06])), self.FR3_HAND_FRAME_ID
        )

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
            "R_LINK5_1": copy.deepcopy(R_link5_1),
            "P_LINK5_1": copy.deepcopy(p_link5_1),
            "q_LINK5_1": copy.deepcopy(q_LINK5_1),
            "R_LINK5_2": copy.deepcopy(R_link5_2),
            "P_LINK5_2": copy.deepcopy(p_link5_2),
            "q_LINK5_2": copy.deepcopy(q_LINK5_2),
            "R_LINK6": copy.deepcopy(R_link6),
            "P_LINK6": copy.deepcopy(p_link6),
            "q_LINK6": copy.deepcopy(q_LINK6),
            "R_LINK7": copy.deepcopy(R_link7),
            "P_LINK7": copy.deepcopy(p_link7),
            "q_LINK7": copy.deepcopy(q_LINK7),
            "R_HAND": copy.deepcopy(R_hand),
            "P_HAND": copy.deepcopy(p_hand),
            "q_HAND": copy.deepcopy(q_HAND),
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

    def compute_crude_location(self, R_offset, p_offset, frame_id):
        # get link orientation and position
        _p = self.robot.data.oMf[frame_id].translation
        _Rot = self.robot.data.oMf[frame_id].rotation

        # compute link transformation matrix
        _T = np.hstack((_Rot, _p[:, np.newaxis]))
        T = np.vstack((_T, np.array([[0.0, 0.0, 0.0, 1.0]])))

        # compute link offset transformation matrix
        _TB = np.hstack((R_offset, p_offset))
        TB = np.vstack((_TB, np.array([[0.0, 0.0, 0.0, 1.0]])))

        # get transformation matrix
        T_mat = T @ TB

        # compute crude model location
        p = (T_mat @ np.array([[0.0], [0.0], [0.0], [1.0]]))[:3, 0]

        # compute crude model orientation
        Rot = T_mat[:3, :3]

        # quaternion
        q = Rotation.from_matrix(Rot).as_quat()

        return p, Rot, q
