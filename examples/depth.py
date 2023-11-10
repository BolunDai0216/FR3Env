import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def main():
    depth = np.load("./depth.npy")
    depth = o3d.geometry.Image(depth)
    pin_cam = o3d.camera.PinholeCameraIntrinsic(
        width=240, height=320, fx=0.01, fy=0.01, cx=0.0, cy=0.0
    )
    pcd = o3d.geometry.PointCloud()
    pcd = pcd.create_from_depth_image(depth, pin_cam, depth_scale=1.0, depth_trunc=4.0)
    o3d.visualization.draw_geometries([pcd])

    # pixels = np.load("./pixels.npy")
    # plt.imshow(pixels.astype(np.uint8))
    # plt.show()


if __name__ == "__main__":
    main()
