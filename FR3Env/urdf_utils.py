import pybullet as p


def get_joint_infos(robot_id):
    num_joints = p.getNumJoints(robot_id)

    for joint_index in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_index)
        joint_id = joint_info[0]
        joint_name = joint_info[1].decode("utf-8")
        print(f"Joint ID: {joint_id}, Joint Name: {joint_name}")
