import numpy as np
from collections import deque
from dataclasses import dataclass
import sys, os
import cv2, skimage.io

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from networks.net import TrainingSample


@dataclass
class ReplayMemoryConfig:
    """ Configuration for ReplayMemory """

    buffer_size: int  # number of elements in the replay memory
    state_size: int  # last 'state_size' images are stacked together to make up a state
    batch_size: int  # batch size for training
    dbg_folder_path: str  # folder to dump out debug images


@dataclass
class RingBufferElement:
    """ Element of the internal ring buffer maintained by ReplayMemory """

    img: np.uint8  # image
    action: int = -1  # action
    reward: float = -1  # reward
    done: bool = False  # was terminal state


# TODO : clean this up : assumes image dimensions
def preprocess_image(img):
    gray_img = np.uint8(cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2GRAY))
    gray_img = gray_img[34:194, :]
    return cv2.resize(gray_img, (80, 80))


def save_batch(batch, batch_idx, op_folder, state_size):
    def save_training_sample(s, sample_idx):
        imgs = []
        for i in range(state_size):
            imgs.append(s[..., i])
        imgs = np.hstack(imgs)
        skimage.io.imsave(
            os.path.join(op_folder, f"./batch_{batch_idx}_sample_{sample_idx}.png"),
            imgs,
        )

    for i in range(len(batch)):
        save_training_sample(batch[i].current_state, i)


class ReplayMemory(object):
    def __init__(self, config: ReplayMemoryConfig):
        """ Accepts curr-states/rewards/next-states and returns batches for SGD """
        self._config = config
        self._ring_buffer = deque([], config.buffer_size)
        self.debug_ctr = 0

    def update(self, action, reward, done):
        """
        Updates internal ring buffer with the data which is given
        Note : function assumes that action/reward/done correspond to the img which was last submitted to get_next_state()
        Args:
            action : action taken for last state 
            reward : reward for the last state
            done : is game done when action was taken
        """
        # update data for current image
        self._ring_buffer[-1].action = action
        self._ring_buffer[-1].reward = reward
        self._ring_buffer[-1].done = done

    def gen_next_state(self, img):
        """
        Returns the next state.
        State is defined as the current + past 'state_size'-1 images stacked along axis 2
        Returns 
            None if number of elements is lesser than whats required to compose a state
            Else np array of size -> (m x n x state_size)
        """
        # create new element for the next image
        self._ring_buffer.append(RingBufferElement(preprocess_image(img)))
        if len(self._ring_buffer) < self._config.state_size:
            return None

        state = np.dstack(
            [self._ring_buffer[i].img for i in range(-self._config.state_size, 0)]
        )
        return state

    def generate_batch(self, batch_idx):
        """
        Returns a list of samples for training
        Returns None if insufficent samples
        """
        # first (n-1) elements dont have history, last element only contains image
        if (
            len(self._ring_buffer) - (self._config.state_size - 1) - 1
            < self._config.state_size * self._config.batch_size
        ):
            return []

        def make_sample(idx):
            """ makes a training sample combining images from idx -> idx-state_size """
            ts = TrainingSample()
            ts.current_state = np.dstack(
                [
                    self._ring_buffer[i].img
                    for i in range(idx - self._config.state_size, idx)
                ]
            )
            ts.action = self._ring_buffer[idx].action
            ts.reward = self._ring_buffer[idx].reward
            ts.next_state = np.dstack(
                [
                    self._ring_buffer[i].img
                    for i in range(idx + 1 - self._config.state_size, idx + 1)
                ]
            )
            return ts

        # generate random indices from state_size-1 : buffersize-state_size
        batch_indices = np.random.randint(
            self._config.state_size - 1,
            len(self._ring_buffer) - 1,
            size=self._config.batch_size - 1,  # current state is added seperately
        )
        batch = [make_sample(idx) for idx in batch_indices]
        batch.append(make_sample(len(self._ring_buffer) - 2))
        assert len(batch) == self._config.batch_size

        # save images of batch
        if self._config.dbg_folder_path is not None:
            save_batch(
                batch, batch_idx, self._config.dbg_folder_path, self._config.state_size
            )

        return batch
