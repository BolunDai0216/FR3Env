# FR3Env: A Simulator for Franka Research 3

## Dependencies

- Core dependencies: `NumPy`, `SciPy`, `PyBullet`, `Pinocchio`, `Gymnasium`, `pyquaternion`, `MuJoCo`
- Dependencies to run the demos: `ProxSuite`, 

## Installation

To install this package, first install the dependencies. `NumPy`, `MuJoCo`, `SciPy`, `PyBullet`, and `Gymnasium` can be installed while running the setup code below. It is recommended to install `Pinocchio`, and `ProxSuite` using conda. Then, run the following commands

```console
git clone https://github.com/BolunDai0216/FR3Env.git
cd FR3Env
python3 -m pip install .
```

It is suggested to install this package within a `conda` virtual environment, since it is easier to install the dependencies (especially `Pinocchio`) using `conda`.

## Run the demo

```console
mjpython examples/waypoint_hierarchical_proxqp_demo.py
```

https://user-images.githubusercontent.com/36321182/199758677-c325b83e-695f-4cad-b302-7c7ee6a30922.mp4

```console
python examples/hello_world_mj_diff_ik.py
```

https://github.com/BolunDai0216/FR3Env/assets/36321182/2aedb089-f50e-4658-8b45-e0b50ba16540

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
import pybullet as p
_, _, rgb, depth, segmentation = p.getCameraImage(W, H, viewMatrix, projectionMatrix, shadow = True)
```
The function above returns the undistorted images, segmentation, and depth maps. 

## Lessons Learned

- To make `pinocchio` work all of the links have to have a specified mass and inertia.

## Credits

The FR3 URDF file is generated using [`fr3-urdf-pybullet`](https://github.com/RumailM/fr3-urdf-pybullet)

## Citation

To cite FR3Env in your academic research, please use the following bibtex entry:

```
@article{DaiKKGTK23,
  author       = {Bolun Dai and Rooholla Khorrambakht and Prashanth Krishnamurthy and Vin{\'{\i}}cius Gon{\c{c}}alves and Anthony Tzes and Farshad Khorrami},
  title        = {Safe Navigation and Obstacle Avoidance Using Differentiable Optimization Based Control Barrier Functions},
  journal      = {{IEEE} Robotics and Automation Letters},
  year         = {2023},
  volume       = {8},
  number       = {9},
  pages        = {5376-5383},
}
```

## Known Issues

When running the MuJoCo viewer on MacOS, you may encounter the following error:

```bash
OSError: dlopen(/System/Library/OpenGL.framework/OpenGL, 0x0006): tried: '/System/Library/OpenGL.framework/OpenGL' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/System/Library/OpenGL.framework/OpenGL' (no such file), '/System/Library/OpenGL.framework/OpenGL' (no such file, not in dyld cache)
```

If you are using a conda environment, you can fix this by going to (replace `3.8` with your python version)

```bash
cd $CONDA_PREFIX/lib/python3.8/site-packages/OpenGL/platform
```

and change the line of `_loadLibraryWindows()` in `ctypesloader.py` from

```python
fullName = util.find_library( name )
```

to 

```python
fullName = '/System/Library/Frameworks/OpenGL.framework/OpenGL'
```

For more details, see [here](https://stackoverflow.com/a/64021312).
