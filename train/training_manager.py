# the training manager trains the given network
import json
import numpy as np
import sys
import os
import skimage.io
import shutil
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from networks.net import Network
from replay_memory import ReplayMemory, ReplayMemoryConfig


class TrainingManager(object):
    def __init__(self, config_json: str):
        self.training_config = json.load(open(config_json, "r"))
        self.training_config["replay_buffer"]["input_shape"] = tuple(
            [
                int(v)
                for v in (
                    self.training_config["replay_buffer"]["input_shape"].split(",")
                )
            ]
        )
        self.training_config["outputs"]["dump_training_samples"] = (
            self.training_config["outputs"]["dump_training_samples"] == "True"
        )
        self.training_config["outputs"]["save_weights"] = (
            self.training_config["outputs"]["save_weights"] == "True"
        )

        # create folder to dump out debug data
        if os.path.exists(self.training_config["outputs"]["output_folder"]):
            shutil.rmtree(self.training_config["outputs"]["output_folder"])
        os.mkdir(self.training_config["outputs"]["output_folder"])

        self.output_folders = {}
        if self.training_config["outputs"]["dump_training_samples"]:
            self.output_folders["training_samples"] = os.path.join(
                self.training_config["outputs"]["output_folder"], "training_samples"
            )
            os.mkdir(self.output_folders["training_samples"])

        # dump tensorboard logs
        self.output_folders["tb_logs"] = os.path.join(
            self.training_config["outputs"]["output_folder"], "tb_logs"
        )
        os.mkdir(self.output_folders["tb_logs"])
        self.writer = tf.summary.create_file_writer(self.output_folders["tb_logs"])

        # create folder to dump checkpoints
        if self.training_config["outputs"]["save_weights"]:
            self.output_folders["weights"] = os.path.join(
                self.training_config["outputs"]["output_folder"], "weights"
            )
            os.mkdir(self.output_folders["weights"])

        # create replay memory
        self.replay_memory = ReplayMemory(
            ReplayMemoryConfig(
                buffer_size=self.training_config["replay_buffer"]["buffer_size"],
                state_size=self.training_config["replay_buffer"]["state_size"],
                batch_size=self.training_config["training"]["batch_size"],
                dbg_folder_path=self.output_folders["training_samples"]
                if self.training_config["outputs"]["dump_training_samples"]
                else None,
            )
        )

    def train(self, net, env, env_action_space):
        """ trains the given network using RL """

        # initialise the network
        net.init(
            self.training_config["replay_buffer"]["input_shape"],
            2,
            self.training_config["training"]["discount_factor"],
            self.output_folders["tb_logs"],
            env_action_space,
            self.writer,
        )

        # training batch index
        batch_idx = 0

        def get_eps(batch_idx):
            """ given a batch index, returns the epsilon for greedy exploration"""
            ratio = min(
                1,
                batch_idx
                / self.training_config["training"]["epsilon_strategy"]["end_batch"],
            )
            eps = (1 - ratio) * self.training_config["training"]["epsilon_strategy"][
                "start_val"
            ] + ratio * self.training_config["training"]["epsilon_strategy"]["end_val"]
            with self.writer.as_default():
                tf.summary.scalar("epsilon", eps, step=batch_idx)
            return eps

        # create multiple episodes
        for epi_cnt in range(0, self.training_config["training"]["num_episodes"]):
            print(f"Playing episode {epi_cnt}............................")
            next_img = env.reset()
            done = False

            # play a game
            while not done:

                # generate next state
                state = self.replay_memory.gen_next_state(next_img)

                # generate action using epsilon-greedy strategy
                if state is not None and np.random.rand() > get_eps(batch_idx):
                    action = net.predict(np.expand_dims(state, axis=0))
                else:
                    action = np.random.choice(env_action_space)

                # generate regward for current action as well as the next state
                next_img, reward, done, info = env.step(action)

                # update replace memory with next image and reward
                self.replay_memory.update(action, reward, done)

                # generate batch of samples for training
                batch = self.replay_memory.generate_batch(batch_idx)
                if len(batch) < self.training_config["training"]["batch_size"]:
                    continue
                batch_idx += 1

                # update policy
                net.train(batch_idx, batch)

                with self.writer.as_default():
                    self.writer.flush()

            if (
                self.training_config["outputs"]["save_weights"]
                and epi_cnt
                % self.training_config["outputs"]["weights_dump_episode_frequency"]
                == 0
            ):
                net.save(self.output_folders["weights"], epi_cnt)

            # if(epi_cnt % self.training_config['eval_frequency']==0):
            #     net.evaluate(env)
