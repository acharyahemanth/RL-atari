# the training manager trains the given network
import json
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from networks.net import Network
from replay_memory import ReplayMemory, ReplayMemoryConfig


class TrainingManager(object):
    def __init__(self, config_json: str):
        self.training_config = json.load(open(config_json, "r"))
        self.replay_memory = ReplayMemory(
            ReplayMemoryConfig(
                buffer_size=self.training_config["buffer_size"],
                state_size=self.training_config["state_size"],
                batch_size=self.training_config["batch_size"],
            )
        )

    def train(self, net: Network, env):
        """ trains the given network using RL """

        # create multiple episodes
        for epi_cnt in range(0, self.training_config["num_episodes"]):
            env.reset()
            done = False
            eps = self.training_config["epsilon"]

            # play a game
            while not done:
                # generate next state
                state = self.replay_memory.gen_next_state()

                # generate action using epsilon-greedy strategy
                if state is not None and np.random.rand() > eps:
                    action = net.predict(state)
                else:
                    action = env.action_space.sample()

                # generate reward for current action as well as the next state
                next_img, reward, done, info = env.step(action)

                # update replace memory with next image and reward
                self.replay_memory.update(action, reward, done, next_img)

                # generate batch of samples for training
                batch = self.replay_memory.generate_batch()
                if len(batch) < self.training_config["batch_size"]:
                    continue

                # update policy
                net.train(batch)

            # if(epi_cnt % self.training_config['checkpoint_dump_frequency']==0):
            #     net.save()

            # if(epi_cnt % self.training_config['eval_frequency']==0):
            #     net.evaluate(env)
