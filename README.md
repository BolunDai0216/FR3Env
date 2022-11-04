# FR3Env: A Simulator for Franka Research 3

## Dependencies

- Core dependencies: `NumPy`, `SciPy`, `PyBullet`, `Pinocchio`, `Gymnasium`
- Dependencies to run the demos: `CasADi`, 

## Installation

To install this package, first install the dependencies, then, run the following commands

```console
git clone https://github.com/BolunDai0216/FR3Env.git
cd FR3Env
python3 -m pip install .
```

It is suggested to install this package within a `conda` virtual environment, since it is easier to install the dependencies (especially `Pinocchio` and `CasADi`) using `conda`.

## Run the demo

```console
fr3env-waypoint-demo
```

https://user-images.githubusercontent.com/36321182/199758677-c325b83e-695f-4cad-b302-7c7ee6a30922.mp4


## Lessons Learned

- To make `pinocchio` work all of the links have to have a specified mass and inertia.

## Credits

The FR3 URDF file is generated using [`fr3-urdf-pybullet`](https://github.com/RumailM/fr3-urdf-pybullet)
