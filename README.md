# FR3Env: A Simulator for Franka Research 3

## Dependencies

- Core dependencies: `NumPy`, `SciPy`, `PyBullet`, `Pinocchio`, `Gymnasium`
- Dependencies to run the demos: `CasADi`, 

## Installation

First change the mesh file path in the URDF file to `<README-Parent-Path>/FR3Env/robots/meshes/<Mesh-Path>`. Then run

```console
python3 -m pip install .
```

It is suggested to install this package within a `conda` virtual environment, since it is easier to install the dependencies (especially `Pinocchio` and `CasADi`) using `conda`.

## Run the demo

```console
fr3env-waypoint-demo
```

## Lessons Learned

- To make `pinocchio` work all of the links have to have a specified mass and inertia.

## Credits

The FR3 URDF file is generated using [`fr3-urdf-pybullet`](https://github.com/RumailM/fr3-urdf-pybullet)