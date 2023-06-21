from pyquaternion import Quaternion
import numpy as np


def cvK2BulletP(K, w, h, near, far):
    """
    cvKtoPulletP converst the K interinsic matrix as calibrated using Opencv
    and ROS to the projection matrix used in openGL and Pybullet.

    :param K:  OpenCV 3x3 camera intrinsic matrix
    :param w:  Image width
    :param h:  Image height
    :param near:     The nearest objects to be included in the render
    :param far:      The furthest objects to be included in the render
    :return:   4x4 projection matrix as used in openGL and pybullet
    """
    f_x = K[0, 0]
    f_y = K[1, 1]
    c_x = K[0, 2]
    c_y = K[1, 2]
    A = (near + far) / (near - far)
    B = 2 * near * far / (near - far)

    projection_matrix = [
        [2 / w * f_x, 0, (w - 2 * c_x) / w, 0],
        [0, 2 / h * f_y, (2 * c_y - h) / h, 0],
        [0, 0, A, B],
        [0, 0, -1, 0],
    ]
    # The transpose is needed for respecting the array structure of the OpenGL
    return np.array(projection_matrix).T.reshape(16).tolist()


def cvPose2BulletView(q, t):
    """
    cvPose2BulletView gets orientation and position as used
    in ROS-TF and opencv and coverts it to the view matrix used
    in openGL and pyBullet.

    :param q: ROS orientation expressed as quaternion [qx, qy, qz, qw]
    :param t: ROS postion expressed as [tx, ty, tz]
    :return:  4x4 view matrix as used in pybullet and openGL

    """
    q = Quaternion([q[3], q[0], q[1], q[2]])
    R = q.rotation_matrix

    _T = np.vstack([np.hstack([R, np.array(t).reshape(3, 1)]), np.array([0, 0, 0, 1])])
    # Convert opencv convention to python convention
    # By a 180 degrees rotation along X
    Tc = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).reshape(
        4, 4
    )

    # pybullet pse is the inverse of the pose from the ROS-TF
    _T = Tc @ np.linalg.inv(_T)
    # The transpose is needed for respecting the array structure of the OpenGL
    viewMatrix = _T.T.reshape(16)

    return viewMatrix
