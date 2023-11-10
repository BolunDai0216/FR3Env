import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pinocchio as pin


def main():
    depth = np.load("./data/depth.npy")
    _depth = np.clip(depth, 0.0, 4.0)

    # _depth_img = (_depth / 4.0 * 255).astype(np.uint8)
    # plt.imshow(_depth_img)
    # plt.show()

    FX_DEPTH = 250
    FY_DEPTH = 250
    CX_DEPTH = 160
    CY_DEPTH = 120

    pcd = []
    height, width = _depth.shape
    for i in range(height):
        for j in range(width):
            z = _depth[i][j]

            if z >= 0.6:
                continue
            x = (j - CX_DEPTH) * z / FX_DEPTH
            y = (i - CY_DEPTH) * z / FY_DEPTH
            pcd.append([x, y, z])

    R = pin.rpy.rpyToMatrix(-1.5708, 0, 3.1416)
    p = np.array([0, -0.4, 0.8])
    T = pin.SE3(R, p)

    pcd_arr = np.array(pcd)
    pcd_arr_homogeneous = np.hstack((pcd_arr, np.ones((pcd_arr.shape[0], 1))))
    pcd_w = (T.homogeneous @ pcd_arr_homogeneous.T)[:3, :].T

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(pcd_w[:, 0], pcd_w[:, 1], pcd_w[:, 2], marker="o")
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    plt.show()

    # pcd_o3d = o3d.geometry.PointCloud()
    # pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    # o3d.visualization.draw_geometries([pcd_o3d])

    # depth = o3d.geometry.Image(_depth)
    # pin_cam = o3d.camera.PinholeCameraIntrinsic(
    #     width=320, height=240, fx=250, fy=250, cx=0.0, cy=0.0
    # )
    # pcd = o3d.geometry.PointCloud()
    # pcd = pcd.create_from_depth_image(depth, pin_cam, depth_scale=1.0, depth_trunc=1.5)
    # o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
