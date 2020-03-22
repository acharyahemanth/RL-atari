import gym
import numpy as np
import skimage.io


def test_pong():
    env = gym.make("Pong-v0")
    initial_state = env.reset()
    print(
        f"is.shape : {initial_state.shape}, type(is) : {np.array(initial_state).dtype}"
    )
    done = False
    frame_cnt = 1
    while not done:
        env.render()
        img, reward, done, info = env.step(5)  # take a random action
        print(f"reward : {reward}, done : {done}")
        # skimage.io.imsave(f"img_{frame_cnt}.png", img)
        frame_cnt += 1
    env.close()


test_pong()
