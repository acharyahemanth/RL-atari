import os
import sys
import gym
import numpy as np

dirpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(dirpath))


from training_manager import TrainingManager
from networks.keras_net import KerasNetwork


# create pong environment
pong_env = gym.make("Pong-v0")

# create network
net = KerasNetwork()

# create training manager
tm = TrainingManager(os.path.join(dirpath, "training_config.json"))

# train
tm.train(net, pong_env, [2, 3])  # 2-> paddle up, 3 -> paddle down
