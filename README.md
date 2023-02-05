# FR3Env: A Simulator for Franka Research 3

## Dependencies

- Core dependencies: `NumPy`, `SciPy`, `PyBullet`, `Pinocchio`, `Gymnasium`, `pyquternion`
- Dependencies to run the demos: `CasADi`, `ProxSuite`, 

## Installation

To install this package, first install the dependencies. `NumPy`, `SciPy`, `PyBullet`, and `Gymnasium` can be installed while running the setup code below. It is recommended to install `Pinocchio`, `CasADi`, and `ProxSuite` using conda. Then, run the following commands

```console
git clone https://github.com/BolunDai0216/FR3Env.git
cd FR3Env
python3 -m pip install .
```

It is suggested to install this package within a `conda` virtual environment, since it is easier to install the dependencies (especially `Pinocchio` and `CasADi`) using `conda`.

## Run the demo

```console
fr3env-waypoint-hierarchical-proxqp-demo
```

https://user-images.githubusercontent.com/36321182/199758677-c325b83e-695f-4cad-b302-7c7ee6a30922.mp4

## Simulate Images From Calibrated Cameras

Cameras can be simulated and placed at various places in the environment. Our simulator provides two useful functions to make this task easiers. The first function converts the camera intrinsic paramters as found using ROS or OpenCV calibration procedure to a pybullet projection matrix and the other constructs the view matrix given the `[R|t]` paris:

``` python
from FR3Env.utils import render_utils

'''
K is the camera matrix as estimated using ROS/OpenCV, w,h are image width and height 
during calibration, and near, far indicate the interest depth to be rendered''' 
render_utils.projectionMatrix = cvK2BulletP(K, w, h, near, far)

''' q and t are two lists representing the orientation and translation 
as parameterized by ROS-TF conventions.'''
render_utils.viewMatrix = cvPose2BulletView(q, t)
```

Using the computed matrices above, you can grab images from the environment as follows:

``` python
import pybullet as b
_, _, rgb, depth, segmentation = b.getCameraImage(W, H, viewMatrix, projectionMatrix, shadow = True)
```
The function above returns the undistorted images, segmentation, and depth maps. 

## Lessons Learned

- To make `pinocchio` work all of the links have to have a specified mass and inertia.

## Credits

The FR3 URDF file is generated using [`fr3-urdf-pybullet`](https://github.com/RumailM/fr3-urdf-pybullet)
