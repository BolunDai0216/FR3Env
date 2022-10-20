from pdb import set_trace

from fr3_env import FR3Env


def main():
    env = FR3Env()
    env.reset()

    for i in range(1000000):
        env.step(1)


if __name__ == "__main__":
    main()
