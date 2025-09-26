import numpy as np

class ActorNet(object):
  """
  A Three-layer fully-connected neural network for actor network. The net has an 
  input dimension of (N, D), with D being the cardinality of the state space.
  There are two hidden layers, with dimension of H1 and H2, respectively.  The output
  provides an action vetcor of dimenson of A. The network uses a ReLU nonlinearity
  for the first and second layer and uses tanh (scaled by a factor of
  ACTION_BOUND) for the final layer. In summary, the network has the following
  architecture:
  
  input - fully connected layer - ReLU - fully connected layer - RelU- fully co-
  nected layer - tanh*ACTION_BOUND
 """

  def __init__(self, input_size, hidden_size, output_size, std=5e-1):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    Win: Input layer weights; has shape (D, H)
    W: Hidden layer weights, has shape (H, H)
    Wout: Output layer weights; has shape (H, A)

    
    We also have the weights for a target network (same architecture but 
    different weights)
    Win_tgt: Input layer weights; has shape (D, H)
    W_tgt: Hidden layer weights, has shape (H, H)
    Wout_tgt: Output layer weights; has shape (H, A)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The continuous variables that constitutes an action vector
      of A dimension.
    """
    
    self.Win = self.he_init(input_size, hidden_size)
    self.W = self.he_init(hidden_size, hidden_size)
    self.Wout = self.he_init(hidden_size, output_size)
    
    self.Win_tgt = self.he_init(input_size, hidden_size)
    self.W_tgt = self.he_init(hidden_size, hidden_size)
    self.Wout_tgt = self.he_init(hidden_size, output_size)
    
    self.optm_cfg_Win = None
    self.optm_cfg_W = None
    self.optm_cfg_Wout = None

    # additional params:
    self.leak = 0.2  # leaking rate for leaky integration
    self.X1 = np.zeros((64, hidden_size)) # store previous hidden state
    self.X2 = np.zeros((64,hidden_size))   # state of hidden neurons
    self.X1_tgt = np.zeros((64,hidden_size))   # state of hidden neurons in target actor
    self.X2_tgt = np.zeros((64,hidden_size))   # state of hidden neurons in target actor
    self.output = np.zeros(output_size) # store the output action

  def evaluate_gradient(self, I, dQ_da, action_bound, index):
    """
    Compute the action and gradients for the network based on the input I
    
    Inputs:
    - I: Input data of shape D. Each X[i] is a training sample.
    - dQ_da: Gradient of the Q-value with respect to the action
    - action_bound: the scaling factor for the action, which is environment
                    dependent.
    - index: index of the current sample in the batch

    Returns:
    - grads: Dictionary mapping parameter names to gradients of those parameters; 
      has the same keys as self.params.
    """

    _ = self.predict_with_batch(I, target=False)
    
    # Backward pass
    grads = {}
    
    delta = dQ_da
    dWout = self.X2.T @ delta
    delta_hidden = np.dot(delta, self.Wout.T) * self.diff_relu(self.X2)
    delta_hidden2 = np.dot(delta_hidden, self.W.T) * self.diff_relu(self.X1)
    dW = delta_hidden.T @ self.X1
    dWin = np.dot(I.T, delta_hidden2)
    grads = (dWin, dW, dWout)

    return grads

  def train(self, I, dQda, action_bound, index):
    """
    Train this neural network using adam optimizer.
    Inputs:
    - I: A numpy array of shape D giving training data.
    - dQda: Gradient of the Q-value with respect to the action
    - action_bound: the scaling factor for the action, which is environment
                    dependent.
    - index: index of the current sample in the batch
    """
 
    # Compute out and gradients using the current minibatch
    grads = self.evaluate_gradient(I, dQda, action_bound, index)

    # Update the weights using adam optimizer
    
    self.Wout = self._adam(self.Wout, grads[2], config=self.optm_cfg_Wout)[0]
    self.W = self._adam(self.W, grads[1], config=self.optm_cfg_W)[0]
    self.Win = self._adam(self.Win, grads[0], config=self.optm_cfg_Win)[0]
    
    # Update the configuration parameters to be used in the next iteration
    self.optm_cfg_Wout = self._adam(self.Wout, grads[2], config=self.optm_cfg_Wout)[1]
    self.optm_cfg_W = self._adam(self.W, grads[1], config=self.optm_cfg_W)[1]
    self.optm_cfg_Win = self._adam(self.Win, grads[0], config=self.optm_cfg_Win)[1]
  
  def train_target(self, tau):
    """
      Update the weights of the target network.
     -tau: coefficent for tracking the learned network.
     """
    self.Wout_tgt = tau*self.Wout+(1-tau)*self.Wout_tgt
    self.W_tgt = tau*self.W+(1-tau)*self.W_tgt
    self.Win_tgt = tau*self.Win+(1-tau)*self.Win_tgt

  def predict_with_batch(self, I, target=False):
    """
    Use the trained weights of this network to predict the action vector for a 
    given state.

    Inputs:
    - X: A numpy array of shape (N, D) 
    - target: if False, use normal weights, otherwise use learned weight.
    - action_bound: the scaling factor for the action, which is environment
                    dependent.

    Returns:
    - y_pred: A numpy array of shape (N,) 
    
    """
    
    if not target:
        Win = self.Win
        W = self.W
        Wout= self.Wout
        X1 = self.X1
        X2 = self.X2
    else:
        Win = self.Win_tgt
        W = self.W_tgt
        Wout = self.Wout_tgt
        X1 = self.X1_tgt
        X2 = self.X2_tgt

    X1_help= I @ Win # Win * I(t)
    relu1 = self.relu(X1_help)
    X1 = relu1 # activation function
    
    X2_help= X1 @ W # W * X1(t)
    relu2 = self.relu(X2_help)
    X2 = relu2 # activation function
    self.output = X2 @ Wout # Wout * X2(t)

    return self.output
  
  def predict_without_batch(self, I, index, target=False):

    X1_help= I @ self.Win # Win * I(t)
    relu1 = self.relu(X1_help)
    self.X1[index] = relu1 # activation function
    X2_help=self.X1[index] @ self.W # W * X1(t)
    relu2 = self.relu(X2_help)
    self.X2[index] = relu2 # activation function
    self.output = self.X2 @ self.Wout # Wout * X2(t)

    return self.output
     
  def _adam(self, x, dx, config=None):
      """
      Uses the Adam update rule, which incorporates moving averages of both the
      gradient and its square and a bias correction term.
    
      config format:
      - learning_rate: Scalar learning rate.
      - beta1: Decay rate for moving average of first moment of gradient.
      - beta2: Decay rate for moving average of second moment of gradient.
      - epsilon: Small scalar used for smoothing to avoid dividing by zero.
      - m: Moving average of gradient.
      - v: Moving average of squared gradient.
      - t: Iteration number (time step)
      """
      if config is None: config = {}
      config.setdefault('learning_rate', 1e-4)
      config.setdefault('beta1', 0.9)
      config.setdefault('beta2', 0.999)
      config.setdefault('epsilon', 1e-8)
      config.setdefault('m', np.zeros_like(x))
      config.setdefault('v', np.zeros_like(x))
      config.setdefault('t', 0)
      
      next_x = None
      
      #Adam update formula,                                                 #
      config['t'] += 1
      config['m'] = config['beta1']*config['m'] + (1-config['beta1'])*dx
      config['v'] = config['beta2']*config['v'] + (1-config['beta2'])*(dx**2)
      mb = config['m'] / (1 - config['beta1']**config['t'])
      vb = config['v'] / (1 - config['beta2']**config['t'])
    
      next_x = x - config['learning_rate'] * mb / (np.sqrt(vb) + config['epsilon'])
      return next_x, config
  def _uniform_init(self, input_size, output_size):
      u = np.sqrt(6./(input_size+output_size))
      return np.random.uniform(-u, u, (input_size, output_size))
  
  def he_init (self, input_size, output_size):
      stddev = np.sqrt(2.0/output_size)
      return np.random.normal(0, stddev, (input_size, output_size))
  
  def relu(self, x):
     return np.where(x > 0, x, 0.002 * x)
  
  def diff_relu(self, x):
    return np.where(x > 0, 1, 0.002)