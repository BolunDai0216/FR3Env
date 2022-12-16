from pdb import set_trace

import pybullet as p
import pybullet_data

from FR3Env import getDataPath


def main():
    client = p.connect(p.GUI)

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

    set_trace()

    print("-------")


if __name__ == "__main__":
    main()
