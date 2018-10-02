from utils import *
import gym

class BCO():
  def __init__(self, state_shape, action_shape, lr=0.002, maxits=1000, M=1000):
    # set initial value
    self.state_dim = state_shape            # state dimension
    self.action_dim = action_shape          # action dimension
    self.lr = lr                            # model update learning rate
    self.maxits = maxits                    # maximum iteration
    self.batch_size = args.batch_size       # batch size
    self.alpha = 0.01                       # alpha = | post_demo | / | pre_demo |
    self.M = M                              # sample to update inverse dynamic model

    # initial session
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True
    self.sess = tf.Session(config=config)

    # set the input placeholder
    with tf.variable_scope("placeholder") as scpoe:
      self.state = tf.placeholder(tf.float32, [None, self.state_dim], name="state")
      self.nstate = tf.placeholder(tf.float32, [None, self.state_dim], name="next_state")
      self.action = tf.placeholder(tf.float32, [None, self.action_dim], name="action")
    
    # build policy model and inverse dynamic model
    self.build_policy_model()
    self.build_idm_model()

    # tensorboard output
    writer = tf.summary.FileWriter("logdir/", graph=self.sess.graph)

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
    return self.sess.run(self.policy_pred_action, feed_dict={
      self.state: state
    })

  def eval_idm(self, state, nstate):
    """get the action by inverse dynamic model from current state and next state"""
    return self.sess.run(self.idm_pred_action, feed_dict={
      self.state: state,
      self.nstate: nstate
    })

  def pre_demonstration(self):
    """uniform sample action to generate (s_t, s_t+1) and action pairs"""
    raise NotImplementedError

  def post_demonstration(self):
    """using policy to generate (s_t, s_t+1) and action pairs"""
    raise NotImplementedError

  def eval_rwd_policy(self):
    """getting the reward by current policy model"""
    raise NotImplementedError
    
  def update_policy(self, state, action):
    """update policy model"""
    num = len(state)
    idxs = get_shuffle_idx(num, self.batch_size)
    for idx in idxs:
      batch_s = [  state[i] for i in idx ]
      batch_a = [ action[i] for i in idx ]
      self.sess.run(self.policy_train_step, feed_dict={
        self.state : batch_s,
        self.action: batch_a
      })
 
  def update_idm(self, state, nstate, action):
    """update inverse dynamic model"""
    num = len(state)
    idxs = get_shuffle_idx(num, self.batch_size)
    for idx in idxs:
      batch_s  = [  state[i] for i in idx ]
      batch_ns = [ nstate[i] for i in idx ]
      batch_a  = [ action[i] for i in idx ]
      self.sess.run(self.idm_train_step, feed_dict={
        self.state : batch_s,
        self.nstate: batch_ns,
        self.action: batch_a
      })

  def get_policy_loss(self, state, action):
    """get policy model loss"""
    return self.sess.run(self.policy_loss, feed_dict={
      self.state: state,
      self.action: action
    })

  def get_idm_loss(self, state, nstate, action):
    """get inverse dynamic model loss"""
    return self.sess.run(self.idm_loss, feed_dict={
      self.state: state,
      self.nstate: nstate,
      self.action: action
    })

  def train(self):
    """training the policy model and inverse dynamic model by behavioral cloning"""

    saver = tf.train.Saver(max_to_keep=1)

    self.sess.run(tf.global_variables_initializer())

    print("\n[Training]")
    # pre demonstration to update inverse dynamic model
    S, nS, A = self.pre_demonstration()
    self.update_idm(S, nS, A)
    for it in range(self.maxits):
      def should(freq):
        return freq > 0 and ((it+1) % freq==0 or it == self.maxits-1 )

      # update policy pi
      S, nS = self.sample_demo()
      A = self.eval_idm(S, nS)
      self.update_policy(S, A)
      policy_loss = self.get_policy_loss(S, A)

      # update inverse dynamic model
      S, nS, A = self.post_demonstration()
      self.update_idm(S, nS, A)
      idm_loss = self.get_idm_loss(S, nS, A)

      if should(args.print_freq):
        print('iteration: %5d, total reward: %5.1f, policy loss: %8.6f, idm loss: %8.6f' % ((it+1), self.eval_rwd_policy(), policy_loss, idm_loss))

      # saving model
      if should(args.save_freq):
        print('saving model')
        saver.save(self.sess, args.model_dir)

  def test(self):
    saver = tf.train.Saver()
    saver.restore(self.sess, args.model_dir)
    print('\n[Testing]\nFinal reward: %5.1f' % self.eval_rwd_policy())

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

