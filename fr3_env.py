import pathlib
from typing import Optional

import numpy as np
import pybullet as p
import pybullet_data
from pinocchio.robot_wrapper import RobotWrapper


class FR3Env:
    def __init__(self, render=True):
        if render:
            self.client = p.connect(p.GUI)
            # Improves rendering performance on M1 Macs
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.client = p.connect(p.DIRECT)

        p.setGravity(0, 0, 0)
        p.setTimeStep(1 / 240)

        # Load plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        # Load Franka Panda Robot
        file_directory = str(pathlib.Path(__file__).parent.resolve())
        robot_URDF = file_directory + "/robots/fr3.urdf"
        self.robotID = p.loadURDF(robot_URDF, useFixedBase=True)

        # Build pin_robot
        self.robot = RobotWrapper.BuildFromURDF(robot_URDF)

    def reset(self):
        print("reset")

    def step(self, action):
        p.stepSimulation()

    def close(self):
        p.disconnect()
