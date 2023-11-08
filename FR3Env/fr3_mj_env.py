import mujoco
import numpy as np
import time
import mujoco.viewer


class FR3MuJocoEnv:
    def __init__(self, render=True) -> None:
        self.model = mujoco.MjModel.from_xml_path("./robots/fr3.xml")
        self.data = mujoco.MjData(self.model)

        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.render = True
        else:
            self.render = False

    def step(self, action):
        self.data.qfrc_applied = action
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()


def main():
    model = mujoco.MjModel.from_xml_path("./robots/fr3.xml")
    data = mujoco.MjData(model)

    model.opt.gravity[0] = 0.0
    model.opt.gravity[1] = 0.0
    model.opt.gravity[2] = 0.0

    target = np.array(
        [0.0, -0.785398163, 0.0, -2.35619449, 0.0, 1.57079632679, 0.785398163397]
    )

    viewer = mujoco.viewer.launch_passive(model, data)

    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()

    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(model, data)

        error = data.qpos[:7] - target
        error_norm = np.linalg.norm(error)

        if error_norm >= 0.3:
            Kp = 10.0
            finger_control = 0.0
        else:
            Kp = np.clip(2.0 / error_norm, 0, 100)
            finger_control = 0.03 - data.qpos[-1]

        _pd_control = Kp * (target - data.qpos[:7]) - 1.0 * data.qvel[:7]
        pd_control = np.append(_pd_control, finger_control)
        pd_control = np.append(pd_control, finger_control)
        data.qfrc_applied = pd_control

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    viewer.close()


if __name__ == "__main__":
    main()
