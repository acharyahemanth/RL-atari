""" 
unit tests for the replay memory 
To run -> python -m pytest test_replay_memory.py
"""
import pytest
from train import replay_memory
from train.replay_memory import ReplayMemoryConfig, ReplayMemory

import numpy as np


def test_gen_next_state():

    # create replay memory
    replay_memory = ReplayMemory(
        ReplayMemoryConfig(
            buffer_size=100, state_size=4, batch_size=10, dbg_folder_path=""
        )
    )

    # replay memory shouldnt return anything untill 4 images are passed in (state size)
    for i in range(3):
        img = i * np.ones((200, 200, 3))
        state = replay_memory.gen_next_state(img)
        assert (
            state is None
        ), "replay memory shouldnt return anything untill 4 images are passed in"

    # check that the correct state is output after 4th image
    for i in range(3, 1000):
        pix_val = i % 256
        img = pix_val * np.ones((200, 200, 3))
        state = replay_memory.gen_next_state(img)
        assert state is not None
        assert state.shape == (80, 80, 4)
        for j in range(4):
            assert np.all(state[..., j] == (pix_val - 3 + j + 256) % 256)


def test_generate_batch():

    # create replay memory
    replay_memory = ReplayMemory(
        ReplayMemoryConfig(
            buffer_size=100, state_size=4, batch_size=10, dbg_folder_path=None
        )
    )

    def check_batch(batch):
        assert len(batch) == 10
        for sample in batch:
            first_idx = sample.current_state[0, 0, 3] - 3
            for i in range(4):
                assert np.all(sample.current_state[..., i] == first_idx + i)
            if not sample.last_episode_state:
                for i in range(4):
                    assert np.all(sample.next_state[..., i] == first_idx + i + 1)

    # generate first batch with done=True
    for i in range(0, 43):  # first 3 + 4*10
        img = i * np.ones((200, 200, 3))
        state = replay_memory.gen_next_state(img)
    state = replay_memory.gen_next_state(
        43 * np.ones((200, 200, 3))
    )  # end of episode state
    replay_memory.update(1, 1, True)
    state = replay_memory.gen_next_state(
        44 * np.ones((200, 200, 3))
    )  # adding a next one for replay buffer to output end of episode state
    batch = replay_memory.generate_batch(0)
    assert batch is not None
    check_batch(batch)
    assert batch[-1].last_episode_state

    # generate a set of batches and check them
    for i in range(45, 100):
        img = i * np.ones((200, 200, 3))
        state = replay_memory.gen_next_state(img)
        batch = replay_memory.generate_batch(0)
        assert batch is not None
        check_batch(batch)
