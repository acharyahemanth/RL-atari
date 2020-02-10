# the training manager trains the given network
import json


class TrainingManager(object):
    def __init__(self):
        self.is_initialised = False

    def init(self, config_json: str):
        """ initialize the training manager """

        self.training_config = json.load(open(config_json, "r"))
        self.is_initialised = True

    def train(self, net, env):
        pass
