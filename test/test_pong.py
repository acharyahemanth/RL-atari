import gym
import numpy as np


def test_pong():
    env = gym.make("Pong-v0")
    initial_state = env.reset()
    print(
        f"is.shape : {initial_state.shape}, type(is) : {np.array(initial_state).dtype}"
    )
    done = False
    while not done:
        env.render()
        img, reward, done, info = env.step(
            env.action_space.sample()
        )  # take a random action
        print(f"reward : {reward}, done : {done}")
    env.close()


test_pong()
