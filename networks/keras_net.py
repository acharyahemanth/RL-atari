import numpy as np
import tensorflow.keras as keras
from keras.callbacks.callbacks import Callback
from typing import NamedTuple, List, Tuple
import os
import tensorflow as tf

from .net import TrainingSample
from .net import Network


class KerasCallBack(Callback):
    """ called by model.fit() and writes out loss values to the tblogs """

    def __init__(self, batch_idx, writer):
        super().__init__()
        self.batch_idx = batch_idx
        self.writer = writer

    def on_epoch_end(self, epoch, logs):
        with self.writer.as_default():
            tf.summary.scalar("loss", logs["loss"], step=self.batch_idx)
            self.writer.flush()


class KerasNetwork(Network):
    def __init__(self):
        self.model = None

    def init(
        self,
        input_shape: Tuple[int],
        num_actions: int,
        discount_factor: float,
        tb_logdir: str,
    ):
        """ sets up the keras model """

        self.num_outputs = num_actions
        self.discount_factor = discount_factor
        self.input_shape = input_shape
        self.model = self.create_model()
        self.writer = tf.summary.create_file_writer(tb_logdir)

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
        q_curr = self.predict(curr_states, predict_all_actions=True)
        q_next = self.predict(next_states, predict_all_actions=True)
        for x, qc, qn in zip(batch, q_curr, q_next):
            max_q = np.max(qn)
            qc[np.where(qn == max_q)] = x.reward + self.discount_factor * max_q

        callback = KerasCallBack(batch_idx, self.writer)

        # update weights
        curr_states = np.stack([v.current_state for v in batch], axis=0)
        self.model.fit(
            x=curr_states,
            y=q_curr,
            batch_size=len(batch),
            epochs=1,
            callbacks=[callback],
        )

    def predict(self, state: List[np.ndarray], predict_all_actions=False):
        nw_op = self.model.predict(x=state, batch_size=state.shape[0])
        if predict_all_actions:
            return nw_op

        return np.argmax(nw_op)

    def save(self, chkpt_folder, epi_cnt):
        self.model.save_weights(os.path.join(chkpt_folder, f"checkpoint_{epi_cnt}"))
