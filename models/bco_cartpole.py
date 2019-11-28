import tensorflow as tf
from tensorflow.keras import layers

from utils import *
from bco import BCO
import gym

class Policy(tf.keras.Model):
  def __init__(self, action_shape):
    super(Policy, self).__init__()
    self.fc = tf.keras.Sequential([
      layers.Dense(8),
      layers.LeakyReLU(0.2),
      layers.Dense(8),
      layers.LeakyReLU(0.2),
      layers.Dense(action_shape, activation='softmax')
    ])

  def call(self, state):
    return self.fc(state)

class InverseDynamicsModel(tf.keras.Model):
  def __init__(self, action_shape):
    super(InverseDynamicsModel, self).__init__()
    self.fc = tf.keras.Sequential([
      layers.Dense(8),
      layers.LeakyReLU(0.2),
      layers.Dense(8),
      layers.LeakyReLU(0.2),
      layers.Dense(action_shape, activation='softmax')
    ])
  def call(self, state, nstate):
    inputs = tf.concat([state, nstate], axis=1)
    return self.fc(inputs)

class BCO_cartpole(BCO):
  def __init__(self, state_shape, action_shape):
    BCO.__init__(self, state_shape, action_shape)

    # set which game to play
    self.env = gym.make('CartPole-v0')

    # loss function and optimizer
    self.sce = tf.keras.losses.SparseCategoricalCrossentropy()
    self.optimizer = tf.keras.optimizers.Adam()
  
  def build_policy_model(self):
    """buliding the policy model as two fully connected layers with leaky relu"""
    self.policy = Policy(self.action_dim)
    self.policy_loss = tf.keras.metrics.Mean(name='policy_loss')

  def build_idm_model(self):
    """building the inverse dynamic model as two fully connnected layers with leaky relu"""
    self.idm = InverseDynamicsModel(self.action_dim)
    self.idm_loss = tf.keras.metrics.Mean(name='idm_loss')

  def pre_demonstration(self):
    """uniform sample action to generate (s_t, s_t+1) and action pairs"""
    terminal = True
    States = []
    Nstates = []
    Actions = []

    for i in range(int(round(self.M / self.alpha))):
      if terminal:
        state = self.env.reset()

      prev_s = state

      a = np.random.randint(self.action_dim)

      state, _, terminal, _ = self.env.step(a)

      States.append(prev_s)
      Nstates.append(state)
      Actions.append(a)

      if i and (i+1) % 10000 == 0:
        print("Collecting idm training data ", i+1)

    return np.array(States), np.array(Nstates), np.array(Actions)

  def post_demonstration(self):
    """using policy to generate (s_t, s_t+1) and action pairs"""
    terminal = True
    States = []
    Nstates = []
    Actions = []

    for i in range(self.M):
      if terminal:
        state = self.env.reset()

      prev_s = state
      s = np.reshape( state, (1, self.state_dim) )

      a = tf.squeeze(self.eval_policy(s)).numpy()
      state, _, terminal, _ = self.env.step(a)

      States.append(prev_s)
      Nstates.append(state)
      Actions.append(a)

    return np.array(States), np.array(Nstates), np.array(Actions)

  def eval_rwd_policy(self):
    """getting the reward by current policy model"""
    terminal = False
    total_reward = 0
    state = self.env.reset()

    while not terminal:
      state = np.reshape( state, (1, self.state_dim) )
      a = tf.squeeze(self.eval_policy(state)).numpy()
      state, reward, terminal, _ = self.env.step(a)
      total_reward += reward

    return total_reward

  @tf.function
  def policy_train_step(self, state, action):
    """tensorflow 2.0 policy train step"""
    with tf.GradientTape() as tape:
      logits = self.policy(state)
      loss = self.sce(action, logits)
    grads = tape.gradient(loss, self.policy.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

    self.policy_loss(loss)

  @tf.function
  def idm_train_step(self, state, nstate, action):
    """tensorflow 2.0 idm train step"""
    with tf.GradientTape() as tape:
      logits = self.idm(state, nstate)
      loss = self.sce(action, logits)
    grads = tape.gradient(loss, self.idm.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.idm.trainable_variables))

    self.idm_loss(loss)
    
if __name__ == "__main__":
  bco = BCO_cartpole(4, 2)
  bco.run()
