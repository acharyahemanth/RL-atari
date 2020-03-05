import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class TrainingSample:
    """ Sample for training """

    current_state: np.ndarray = np.array(
        []
    )  # set of images concatenated along axis-2 (h x w x n)
    action: int = -1  # action taken by network for current_state
    reward: float = 0  # reward rxd for action
    next_state: np.ndarray = np.array([])  # next state after taking action


class Network(ABC):
    """ Base class for all networks """

    @abstractmethod
    def init(self, input_shape: Tuple[int], num_actions: int, discount_factor: float):
        assert False

    @abstractmethod
    def predict(self, state: List[np.ndarray], predict_all_actions=False):
        """ if predict_all_actions is true, it returns the output of the network directly (np.ndarray(num_actions), else it returns a single number corresponding to the action with the largest Q value)"""
        assert False

    @abstractmethod
    def train(self, batch: List[TrainingSample]):
        assert False
