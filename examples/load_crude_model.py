import pybullet as p
import pybullet_data

from FR3Env import getDataPath


def main():
    p.connect(p.GUI)

    # Improves rendering performance on M1 Macs
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # Load plane
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    # Load crude model of Franka Research 3 Robot
    package_directory = getDataPath()
    robot_URDF = package_directory + "/robots/fr3_crude.urdf"
    urdf_search_path = package_directory + "/robots"
    p.setAdditionalSearchPath(urdf_search_path)
    robotID = p.loadURDF("fr3_crude.urdf", useFixedBase=True)

    active_joint_ids = [0, 1, 2, 3, 4, 5, 6, 10, 11]

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

    for i, joint_ang in enumerate(target_joint_angles):
        p.resetJointState(robotID, active_joint_ids[i], joint_ang, 0.0)

    while True:
        p.stepSimulation()


if __name__ == "__main__":
    main()
