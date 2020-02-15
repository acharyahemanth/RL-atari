import numpy as np
import keras
from typing import NamedTuple, List

from .net import TrainingSample
from .net import Network


class KerasNetwork(Network):
    def __init__(self):
        self.model = None

    def init(self, num_actions: int, discount_factor: float):
        """ sets up the keras model """

        self.num_outputs = num_actions
        self.discount_factor = discount_factor
        self.model = self.create_model()

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

        return model

    def train(self, batch: List[TrainingSample]):
        """ trains the network using the batch of samples """

        # evaluate q function for input samples (i.e expected network output) --------------
        next_states = [v.next_state for v in batch]
        rewards = [v.reward for v in batch]
        predicted_actions_ns = self.model.predict(next_states)
        action_values = [np.max(v) for v in predicted_actions_ns]
        expected_output = np.array(rewards) + self.discount_factor * np.array(
            action_values
        )

        # update weights
        curr_states = [v.current_state for v in batch]
        self.model.fit(
            x=curr_states, y=expected_output, batch_size=len(batch), epochs=1
        )

    def predict(self, state: List[np.ndarray]):
        return self.model.predict(
            x=state,
            batch_size=len(
                state
            ),  # TODO : this doesnt make much sense, why does predict need a batch size?!
        )
