import numpy as np
from collections import deque
from typing import NamedTuple

from ..networks import TrainingSample


class ReplayMemoryConfig(NamedTuple):
    """ Configuration for ReplayMemory """

    buffer_size: int  # number of elements in the replay memory
    state_size: int  # last 'state_size' images are stacked together to make up a state
    batch_size: int  # batch size for training


class RingBufferElement(NamedTuple):
    """ Element of the internal ring buffer maintained by ReplayMemory """

    img: np.uint8  # image
    action: int = -1  # action
    reward: float = -1  # reward
    done: bool = False  # was terminal state


class ReplayMemory(object):
    def __init__(self, config: ReplayMemoryConfig):
        """ Accepts curr-states/rewards/next-states and returns batches for SGD """
        self._config = config
        self._ring_buffer = deque([], config.buffer_size)

    def update(self, action, reward, done, next_img):
        """
        Updates internal ring buffer with the data which is given
        Note : function assumes that action/reward/done correspond to the state which was last handed out by get_next_state()
        Args:
            action : action taken for last state 
            reward : reward for the last state
            done : is game done when action was taken
            next_img : next_img returned by the environment when action was taken
        """
        # update data for current image
        self._ring_buffer[-1].action = action
        self._ring_buffer[-1].reward = reward
        self._ring_buffer[-1].done = done
        # create new element for the next image
        self._ring_buffer.append(RingBufferElement(img=np.unit8(next_img)))

    def gen_next_state(self):
        """
        Returns the next state.
        State is defined as the past 'state_size' images stacked along each channel
        Returns 
            None if number of elements is lesser than whats required to compose a state
            Else np array of size -> (m x n x state_size)
        """
        if len(self._ring_buffer) < self._config.state_size:
            return None

        return np.dstack(self._ring_buffer[-self._config.state_size :])

    def generate_batch(self):
        """
        Returns a list of samples for training
        Returns None if insufficent samples
        """
        # first (n-1) elements dont have history, last element only contains image
        if (
            len(self._ring_buffer) - (self._config.state_size - 1) - 1
            < self._config.state_size * self._config.batch_size
        ):
            return None

        def make_sample(idx):
            """ makes a training sample combining images from idx -> idx-state_size """
            ts = TrainingSample()
            ts.current_state = np.dstack(
                [
                    item.img
                    for item in self._ring_buffer[idx - self._config.state_size : idx]
                ]
            )
            ts.action = self._ring_buffer[idx].action
            ts.reward = self._ring_buffer[idx].rewards
            ts.next_state = np.dstack(
                [
                    item.img
                    for item in self._ring_buffer[
                        idx + 1 - self._config.state_size : idx + 1
                    ]
                ]
            )
            return ts

        # generate random indices from state_size : buffersize-state_size
        batch_indices = np.random.randint(
            self._config.state_size,
            len(self._ring_buffer) - 2,
            size=self._config.batch_size - 1,
        )
        batch = [make_sample(idx) for idx in batch_indices]
        # always add the current state
        batch.append(make_sample(len(self._ring_buffer) - 2))

        assert len(batch) == self._config.batch_size

        return batch
