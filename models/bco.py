import tensorflow as tf; tf.keras.backend.set_floatx('float64')

from utils import *
import time
import gym

class BCO():
  def __init__(self, state_shape, action_shape):
    # set initial value
    self.state_dim = state_shape            # state dimension
    self.action_dim = action_shape          # action dimension
    self.lr = args.lr                       # model update learning rate
    self.max_episodes = args.max_episodes   # maximum episode
    self.batch_size = args.batch_size       # batch size
    self.alpha = 0.01                       # alpha = | post_demo | / | pre_demo |
    self.M = args.M                         # sample to update inverse dynamic model

    # build policy model and inverse dynamic model
    self.build_policy_model()
    self.build_idm_model()

    self.test_time = 100

  def load_demonstration(self):
    """Load demonstration from the file"""
    if args.input_filename is None or not os.path.isfile(args.input_filename):
      raise Exception("input filename does not exist")

    inputs = []
    targets = []
    
    i = 0
    for trajectory in open(args.input_filename):
      s, s_prime = trajectory.replace("\n", "").replace(",,", ",").split(" ")
      s = eval(s)
      s_prime = eval(s_prime)
      inputs.append(s)
      targets.append(s_prime)
      i += 1
      if i % 10000 == 0:
        print("Loading demonstration ", i)

      if i >= 50000:
        break

    num_samples = len(inputs)

    return num_samples, inputs, targets

  def sample_demo(self):
    """sample demonstration"""
    sample_idx = range(self.demo_examples)
    sample_idx = np.random.choice(sample_idx, self.num_sample)
    S = [ self.inputs[i] for i in sample_idx ]
    nS = [ self.targets[i] for i in sample_idx ]
    return S, nS

  def build_policy_model(self):
    """buliding the policy model as two fully connected layers with leaky relu"""
    raise NotImplementedError

  def build_idm_model(self):
    """building the inverse dynamic model as two fully connnected layers with leaky relu"""
    raise NotImplementedError

  def eval_policy(self, state):
    """get the action by current state"""
    return tf.math.argmax( self.policy(state), axis=1 )

  def eval_idm(self, state, nstate):
    """get the action by inverse dynamic model from current state and next state"""
    return tf.math.argmax( self.idm(state, nstate), axis=1 )

  def pre_demonstration(self):
    """uniform sample action to generate (s_t, s_t+1) and action pairs"""
    raise NotImplementedError

  def post_demonstration(self):
    """using policy to generate (s_t, s_t+1) and action pairs"""
    raise NotImplementedError

  def eval_rwd_policy(self):
    """getting the reward by current policy model"""
    raise NotImplementedError
    
  @tf.function
  def policy_train_step(self, state, action):
    """tensorflow 2.0 policy train step"""
    raise NotImplementedError

  @tf.function
  def idm_train_step(self, state, nstate, action):
    """tensorflow 2.0 idm train step"""
    raise NotImplementedError

  def update_policy(self, state, action):
    """update policy model"""
    num = len(state)
    idxs = get_shuffle_idx(num, self.batch_size)
    for idx in idxs:
      batch_s = tf.constant([ state[i] for i in idx ])
      batch_a = tf.gather( action, idx )
      self.policy_train_step( batch_s, batch_a )
 
  def update_idm(self, state, nstate, action):
    """update inverse dynamic model"""
    num = len(state)
    idxs = get_shuffle_idx(num, self.batch_size)
    for idx in idxs:
      batch_s  = tf.constant([  state[i] for i in idx ])
      batch_ns = tf.constant([ nstate[i] for i in idx ])
      batch_a  = tf.constant([ action[i] for i in idx ])
      self.idm_train_step( batch_s, batch_ns, batch_a )

  def get_policy_loss(self):
    """get policy model loss"""
    loss = self.policy_loss.result()
    self.policy_loss.reset_states()
    return loss

  def get_idm_loss(self):
    """get inverse dynamic model loss"""
    loss = self.idm_loss.result()
    self.idm_loss.reset_states()
    return loss

  def train(self):
    """training the policy model and inverse dynamic model by behavioral cloning"""

    ckpt = tf.train.Checkpoint(model=self.policy)
    manager = tf.train.CheckpointManager(ckpt, args.model_dir, max_to_keep=3)

    print("\n[Training]")
    # pre demonstration to update inverse dynamic model
    S, nS, A = self.pre_demonstration()
    self.update_idm(S, nS, A)
    start = time.time()
    for i_episode in range(self.max_episodes):
      def should(freq):
        return freq > 0 and ((i_episode+1) % freq==0 or i_episode == self.max_episodes-1 )

      # update policy pi
      S, nS = self.sample_demo()
      A = self.eval_idm(S, nS)
      self.update_policy(S, A)
      policy_loss = self.get_policy_loss()

      # update inverse dynamic model
      S, nS, A = self.post_demonstration()
      self.update_idm(S, nS, A)
      idm_loss = self.get_idm_loss()

      if should(args.print_freq):
        now = time.time()
        print('Episode: {:5d}, total reward: {:5.1f}, policy loss: {:8.6f}, idm loss: {:8.6f}, time: {:5.3f} sec/episode'.format((i_episode+1), self.eval_rwd_policy(), policy_loss, idm_loss, (now-start)/args.print_freq))
        start = now

      # saving model
      if should(args.save_freq):
        save_path = manager.save()
        print('saving model: {}'.format(save_path))

  def test(self):
    ckpt = tf.train.Checkpoint(model=self.policy)
    ckpt.restore(tf.train.latest_checkpoint(args.model_dir))
    print('\n[Testing]\nFinal reward: {:5.1f}'.format(self.eval_rwd_policy()))

  def run(self):
    if not os.path.exists(args.model_dir):
      os.makedirs(args.model_dir)

    if args.mode == 'test':
      if args.model_dir is None:
        raise Exception("checkpoint required for test mode")

      self.test()

    if args.mode == 'train':
      # read demonstration data
      self.demo_examples, self.inputs, self.targets = self.load_demonstration()
      self.num_sample = self.M

      self.train()

