import time

import numpy as np

from FR3Env.fr3_mj_env import FR3MuJocoEnv


def main():
    env = FR3MuJocoEnv()
    info = env.reset()

    target = np.array(
        [np.pi / 2, -0.785398163, 0.0, -2.35619449, 0.0, 1.57079632679, 0.785398163397]
    )
    Kd = 10.0

    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()

    while env.viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        error = target - info["q"][:7]
        error_norm = np.linalg.norm(error)

        if error_norm >= 0.3:
            Kp = 10.0
            finger_pos = 0.0
        else:
            Kp = np.clip(1.0 / error_norm, 0, 100)
            finger_pos = 0.03

        tau = Kp * error + Kd * (0 - info["dq"][:7]) + info["G"][:7]

        info = env.step(tau, finger_pos)
        env.sleep(step_start)

    env.close()


if __name__ == "__main__":
    main()
