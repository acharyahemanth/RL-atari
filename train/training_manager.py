# the training manager trains the given network
import json
import numpy as np
import sys
import os
import skimage.io
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from networks.net import Network
from replay_memory import ReplayMemory, ReplayMemoryConfig


class TrainingManager(object):
    def __init__(self, config_json: str):
        self.training_config = json.load(open(config_json, "r"))
        self.training_config["input_shape"] = tuple(
            [int(v) for v in (self.training_config["input_shape"].split(","))]
        )
        self.training_config["enable_debug_mode"] = (
            self.training_config["enable_debug_mode"] == "True"
        )

        # create folder to dump out debug images
        self.dbg_folder = "./dbg_imgs"
        if self.training_config["enable_debug_mode"]:
            if os.path.exists(self.dbg_folder):
                print(f"deleting contents of {self.dbg_folder}...")
                shutil.rmtree(self.dbg_folder)
            os.mkdir(self.dbg_folder)

        # create replay memory
        self.replay_memory = ReplayMemory(
            ReplayMemoryConfig(
                buffer_size=self.training_config["buffer_size"],
                state_size=self.training_config["state_size"],
                batch_size=self.training_config["batch_size"],
                dbg_folder_path=self.dbg_folder
                if self.training_config["enable_debug_mode"]
                else None,
            )
        )

    def train(self, net, env):
        """ trains the given network using RL """
        # initialise the network
        net.init(
            self.training_config["input_shape"],
            2,
            self.training_config["discount_factor"],
        )

        # training batch index
        batch_idx = 0

        # create multiple episodes
        for epi_cnt in range(0, self.training_config["num_episodes"]):
            next_img = env.reset()
            done = False
            eps = self.training_config["epsilon"]

            # play a game
            while not done:
                # generate next state
                state = self.replay_memory.gen_next_state(next_img)

                # generate action using epsilon-greedy strategy
                if state is not None and np.random.rand() > eps:
                    action = net.predict(np.expand_dims(state, axis=0))
                else:
                    action = env.action_space.sample()

                # generate reward for current action as well as the next state
                next_img, reward, done, info = env.step(action)

                # update replace memory with next image and reward
                self.replay_memory.update(action, reward, done)

                # generate batch of samples for training
                batch = self.replay_memory.generate_batch(batch_idx)
                if len(batch) < self.training_config["batch_size"]:
                    continue
                batch_idx += 1

                # update policy
                net.train(batch)

            # if(epi_cnt % self.training_config['checkpoint_dump_frequency']==0):
            #     net.save()

            # if(epi_cnt % self.training_config['eval_frequency']==0):
            #     net.evaluate(env)
