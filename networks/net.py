import numpy as np
from typing import NamedTuple, List
from abc import ABC, abstractmethod


class TrainingSample(NamedTuple):
    """ Sample for training """

    current_state: np.ndarray  # set of images concatenated along axis-2 (h x w x n)
    action: int  # action taken by network for current_state
    reward: float  # reward rxd for action
    next_state: np.ndarray  # next state after taking action


class Network(ABC):
    """ Base class for all networks """

    @abstractmethod
    def init(self, num_actions: int, discount_factor: float):
        assert False

    @abstractmethod
    def predict(self, state: List[np.ndarray]):
        """ returns np.ndarray of actions """  # TODO : mention shape here!
        assert False

    @abstractmethod
    def train(self, batch: List[TrainingSample]):
        assert False
