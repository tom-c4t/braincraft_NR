import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras.src.layers import RNN
from keras.src import ops

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from bot import Bot
from environment_1 import Environment

# Specify dimensions of observation and action space
num_states = 67 # 64 depth values from camera + hit sensor + energy level + bias (=1.0)
num_actions = 1 # one angle between -5 and 5 degrees
upper_bound = 5 # upper bound for action
lower_bound = -5 # lower bound for action
batch_size = 64 # batch size

# Define a customized Keras layer for equation 1
class LeakyRNNCell(keras.layers.Layer):
    def __init__(self, units, leak=0.5, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.leak = leak
        self.activation = tf.keras.activations.get(activation)
        self.state_size = self.units
        self.output_size = self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # Input weight matrix Win
        self.Win = self.add_weight(shape=(input_dim, self.units),
                                   initializer="glorot_uniform",
                                   trainable=True,
                                   name="Win")

        # Recurrent weight matrix W
        self.W = self.add_weight(shape=(self.units, self.units),
                                 initializer="orthogonal",
                                 trainable=True,
                                 name="W")

    def call(self, inputs, states):
        prev_state = states[0]
        # f(W * X + Win * I)
        preact = tf.matmul(prev_state, self.W) + tf.matmul(inputs, self.Win)
        candidate = self.activation(preact)

        # X(t+1) = (1 - λ)X(t) + λ * candidate
        new_state = (1.0 - self.leak) * prev_state + self.leak * candidate

        return new_state, [new_state]

# Defining the Actor network
def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

    # input equation (equation 1)
    inputs = keras.Input(shape=(None, num_states))
    print(f"Input: {inputs}")
    cell = LeakyRNNCell(units = 1000, leak=0.3)
    rnn_layer = RNN(cell, return_sequences=True)
    rnn_output = rnn_layer(inputs)

    # Output equation (equation 2)
    outputs = keras.layers.Dense(1, activation="tanh", use_bias=False, kernel_initializer=last_init)(rnn_output)
    # Our upper bound is 5.0 (maximum turning angle)
    outputs = outputs * upper_bound
    model = keras.Model(inputs, outputs)
    return model


# Defining the Critic network
def get_critic():
    # State as input
    state_inputs = keras.Input((batch_size, num_states))
    state_out = keras.layers.Dense(16, activation="relu")(state_inputs)
    state_out = keras.layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = keras.Input((None, num_actions))
    action_out = keras.layers.Dense(32, activation="relu")(action_input)

    # Both are passed through separate layer before concatenating
    concat = keras.layers.Concatenate()([state_out, action_out])

    out = keras.layers.Dense(256, activation="relu")(concat)
    out = keras.layers.Dense(256, activation="relu")(out)
    outputs = keras.layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = keras.Model([state_inputs, action_input], outputs)

    return model


# Define training hyperparameters
actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = keras.optimizers.Adam(critic_lr)
actor_optimizer = keras.optimizers.Adam(actor_lr)

total_episodes = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005


# Return an action
def policy(state):
    sampled_actions = keras.ops.squeeze(actor_model(state), axis=-1)
    sampled_actions = sampled_actions.numpy()

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    return [np.squeeze(legal_action)]


# Introduce an Experience Replay Buffer
class Buffer:
    def __init__(self, buffer_capacity=1000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') observation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            #print(f"Next state batch pre:{next_state_batch.shape}")
            next_state_batch = keras.ops.expand_dims(next_state_batch,0)
            #print(f"Next state batch post:{next_state_batch.shape}")
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic([next_state_batch, target_actions], training=True)
            state_batch = keras.ops.expand_dims(state_batch,0)
            action_batch = keras.ops.expand_dims(action_batch,0)
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -keras.ops.mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = keras.ops.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = keras.ops.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = keras.ops.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = keras.ops.cast(reward_batch, dtype="float32")
        next_state_batch = keras.ops.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# Slow update of target parameters
# Based on rate `tau`, which is much less than one.
def update_target(target, original, tau):
    target_weights = target.get_weights()
    original_weights = original.get_weights()

    for i in range(len(target_weights)):
        target_weights[i] = original_weights[i] * tau + target_weights[i] * (1 - tau)

    target.set_weights(target_weights)

# Return weigths
def training_function():
    # Instantiate every class that is necessary
    num_episodes = 10000
    buffer = Buffer(num_episodes, batch_size)
    bot = Bot()
    environment = Environment()

    while bot.energy > 0:
        # Initialization
        prev_state = np.zeros((num_states))
        state = np.zeros((num_states))
        prev_state[:64] = bot.camera.depths
        prev_state[64:] = bot.hit, bot.energy, 1.0
        #print(f"Prev state: {prev_state}")
        prev_energy = bot.energy
        tf_prev_state = keras.ops.expand_dims(keras.ops.convert_to_tensor(prev_state),0)
        tf_prev_state = keras.ops.expand_dims(tf_prev_state,0)

        # Choose action and interact
        action = policy(tf_prev_state)
        action = action[0]
        energy, hit, depth, values = bot.forward(dtheta=action, environment=environment)
        reward = energy - prev_energy
        state[:64] = depth
        state[64:] = hit, energy, 1.0
        print(f"Reward: {reward}")
        #print(f"State: {state}")

        # Update Experience Replay Buffer
        buffer.record((prev_state, action, reward, state))
        buffer.learn()

        # Update Actor and Critic network
        update_target(target_actor, actor_model, tau)
        update_target(target_critic, critic_model, tau)

        # Return the weights
        Wout = actor_model.get_layer("dense").get_weights()[0]
        Win = actor_model.get_layer("rnn").get_weights()[0]
        W = actor_model.get_layer("rnn").get_weights()[1]

        warmup = 0
        leak = actor_model.get_layer("rnn").cell.leak
        f = actor_model.get_layer("rnn").cell.activation
        g = actor_model.get_layer("dense").activation
        model = Win.T, W.T, Wout.T, warmup, leak, f, g
        yield model

#----------------------------------------------------------------------
if __name__ == "__main__":
    import time
    import numpy as np    
    from challenge import train, evaluate
    
    # Training (100 seconds)
    print(f"Starting training for 100 seconds (user time)")
    model = train(training_function, timeout=100)

    # Evaluation
    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, debug=False, seed=None)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score: {score:.2f} ± {std:.2f}")
