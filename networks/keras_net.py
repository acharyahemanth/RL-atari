import numpy as np
import tensorflow.keras as keras
from keras.callbacks.callbacks import Callback
from typing import NamedTuple, List, Tuple
import os
import tensorflow as tf

from .net import TrainingSample
from .net import Network


class KerasTBCallBack(Callback):
    """ called by model.fit() and writes out loss values to the tblogs """

    def __init__(self, batch_idx, writer):
        super().__init__()
        self.batch_idx = batch_idx
        self.writer = writer

    def on_epoch_end(self, epoch, logs):
        with self.writer.as_default():
            tf.summary.scalar("loss", logs["loss"], step=self.batch_idx)


class KerasNetwork(Network):
    def __init__(self):
        self.model = None

    def init(
        self,
        input_shape: Tuple[int],
        num_actions: int,
        discount_factor: float,
        tb_logdir: str,
        env_action_space: List[int],
        tb_writer,
    ):
        """ sets up the keras model """

        self.num_outputs = num_actions
        self.discount_factor = discount_factor
        self.input_shape = input_shape
        self.model = self.create_model()
        self.writer = tb_writer
        self.env_action_space = (
            env_action_space  # ith nn_output maps to env_action_space[i]
        )
        self.openaiaction_to_nn_output = {}
        for idx, action in enumerate(self.env_action_space):
            self.openaiaction_to_nn_output[action] = idx

    def create_model(self):
        """ 
        Builds a keras model of the network
        Network architecture : 
        input -> conv1(16x8x8) -> conv2(32,4,4) -> dense(256) -> dense(num_actions)
        """
        model = keras.Sequential()

        # add conv1
        model.add(
            keras.layers.Conv2D(
                filters=16,
                kernel_size=(8, 8),
                strides=(2, 2),
                data_format="channels_last",
                activation="relu",
                input_shape=self.input_shape,
            )
        )

        # add conv2
        model.add(
            keras.layers.Conv2D(
                filters=32, kernel_size=(4, 4), strides=(2, 2), activation="relu",
            )
        )

        # flatten before adding dense layers
        model.add(keras.layers.Flatten())

        # dense layer
        model.add(keras.layers.Dense(units=256, activation="relu", use_bias=True,))

        # output layer
        model.add(
            keras.layers.Dense(
                units=self.num_outputs, use_bias=True, activation="linear"
            )
        )

        # compile model (i.e define optimizer, cost functions etc)
        model.compile(optimizer="rmsprop", loss="mse")

        print("Keras model building success!!!")
        model.summary()

        return model

    def train(self, batch_idx: int, batch: List[TrainingSample]):
        """ trains the network using the batch of samples """

        # evaluate q function for input samples (i.e expected network output) --------------
        # TODO : this is inefficient, save current q_curr if current state has been run through the nw
        curr_states = np.stack([v.current_state for v in batch], axis=0)
        next_states = np.stack([v.next_state for v in batch], axis=0)
        q_curr = self.predict(
            curr_states, convert_to_openai_action_space=False, predict_all_actions=True
        )  # network q-value predictions for current state
        q_next = self.predict(
            next_states, convert_to_openai_action_space=False, predict_all_actions=True
        )  # network q-value predictions for next state
        for x, qc, qn in zip(batch, q_curr, q_next):
            max_q = np.max(
                qn
            )  # max expected reward if optimal policy is followed from subsequent step
            if x.last_episode_state:
                qc[self.openaiaction_to_nn_output[x.action]] = x.reward
            else:
                qc[self.openaiaction_to_nn_output[x.action]] = (
                    x.reward + self.discount_factor * max_q
                )  # GT for current sample contains the current n/w prediction for all other actions

        # callback to write out TB logs
        tb_callback = KerasTBCallBack(batch_idx, self.writer)

        # update weights
        curr_states = np.stack([v.current_state for v in batch], axis=0)
        self.model.fit(
            x=curr_states,
            y=q_curr,
            batch_size=len(batch),
            epochs=1,
            callbacks=[tb_callback],
            verbose=0,
        )

    def predict(
        self,
        state: List[np.ndarray],
        convert_to_openai_action_space=True,
        predict_all_actions=False,
    ):
        nw_op = self.model.predict(x=state, batch_size=state.shape[0])
        if predict_all_actions:
            assert convert_to_openai_action_space == False
            return nw_op

        if convert_to_openai_action_space:
            return self.env_action_space[np.argmax(nw_op)]
        else:
            return np.argmax(nw_op)

    def save(self, chkpt_folder, epi_cnt):
        self.model.save_weights(os.path.join(chkpt_folder, f"checkpoint_{epi_cnt}"))
