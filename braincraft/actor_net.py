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
    
    
    self.params = {}
    self.params['Win'] = self._uniform_init(input_size, hidden_size)
    self.params['W'] = self._uniform_init(hidden_size, hidden_size)
    self.params['Wout'] = np.random.uniform(-3e-3, 3e-3, (hidden_size, output_size)) 
    
    self.params['Win_tgt'] = self._uniform_init(input_size, hidden_size)
    self.params['W_tgt'] = self._uniform_init(hidden_size, hidden_size)
    self.params['Wout_tgt'] = np.random.uniform(-3e-3, 3e-3, (hidden_size, output_size))
    
    self.optm_cfg ={}
    self.optm_cfg['Win'] = None
    self.optm_cfg['W'] = None
    self.optm_cfg['Wout'] = None
    
  def evaluate_gradient(self, I, X, leak, action_grads, action_bound, target=False):
    """
    Compute the action and gradients for the network based on the input I
    
    Inputs:
    - I: Input data of shape D. Each X[i] is a training sample.
    - X: Hidden state of shape H
    - target: use default weights if False; otherwise use target weights.
    - action_grads: the gradient output from the critic-network.
    - action_bound: the scaling factor for the action, which is environment
                    dependent.

    Returns:
     A tuple of:
    - actions: a continuous vector
    - grads: Dictionary mapping parameter names to gradients of those parameters; 
      has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    if not target:
        Win = self.params['Win']
        W = self.params['W']
        Wout= self.params['Wout']
    else:
        Win = self.params['Win_tgt']
        W = self.params['W_tgt']
        Wout = self.params['Wout_tgt']
        
    batch_size, _ = I.shape

    # Compute the forward pass
    scores = None
    z1=np.dot(I,Win) # Win * I(t)
    z2=np.dot(X,W) # W * X(t)
    H1 = np.tanh(z1+z2) #f (W*X(t)+Win*I(t))
    X = (1-leak)*X + leak*H1 # leaky integration
    scores=np.tanh(X)
    output=np.dot(scores,Wout)
    
    # Backward pass
    grads = {}
    
    # 1. Gradient of output w.r.t Wout
    dWout = np.dot(scores.T, action_grads)
    
    # 2. Gradient through tanh(X_new)
    dscores = np.dot(action_grads, Wout.T)
    dX_new = dscores * (1 - scores**2)  # tanh derivative: 1 - tanh²(x)
    
    # 3. Gradient through leaky integration
    dH1 = leak * dX_new
    dX_prev = (1-leak) * dX_new
    
    # 4. Gradient through tanh(z1+z2)
    dz = dH1 * (1 - H1**2)  # tanh derivative
    
    # 5. Gradient w.r.t Win and W
    dWin = np.dot(I.T, dz)
    dW = np.dot(X.T, dz)
    
    grads['Win'] = dWin
    grads['W'] = dW
    grads['Wout'] = dWout
    
    return grads

  def train(self, I, action_grads, action_bound):
    """
    Train this neural network using adam optimizer.
    Inputs:
    - I: A numpy array of shape D giving training data.
    """
 
    # Compute out and gradients using the current minibatch
    grads = self.evaluate_gradient(X, action_grads, \
                                      action_bound)
    # Update the weights using adam optimizer
    
    self.params['Wout'] = self._adam(self.params['Wout'], grads['Wout'], config=self.optm_cfg['Wout'])[0]
    self.params['W'] = self._adam(self.params['W'], grads['W'], config=self.optm_cfg['W'])[0]
    self.params['Win'] = self._adam(self.params['Win'], grads['Win'], config=self.optm_cfg['Win'])[0]
    
    # Update the configuration parameters to be used in the next iteration
    self.optm_cfg['Wout'] = self._adam(self.params['Wout'], grads['Wout'], config=self.optm_cfg['Wout'])[1]
    self.optm_cfg['W'] = self._adam(self.params['W'], grads['W'], config=self.optm_cfg['W'])[1]
    self.optm_cfg['Win'] = self._adam(self.params['Win'], grads['Win'], config=self.optm_cfg['Win'])[1]
  
  def train_target(self, tau):
    """
      Update the weights of the target network.
     -tau: coefficent for tracking the learned network.
     """
    self.params['Wout_tgt'] = tau*self.params['Wout']+(1-tau)*self.params['Wout_tgt']
    self.params['W_tgt'] = tau*self.params['W']+(1-tau)*self.params['W_tgt']
    self.params['Win_tgt'] = tau*self.params['Win']+(1-tau)*self.params['Win_tgt']

  def predict(self, I, X, leak, action_bound, target=False):
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
    y_pred = None
    
    if not target:
        Win = self.params['Win']
        W = self.params['W']
        Wout= self.params['Wout']
    else:
        Win = self.params['Win_tgt']
        W = self.params['W_tgt']
        Wout = self.params['Wout_tgt']

    scores = None
    z1=np.dot(I,Win) # Win * I(t)
    z2=np.dot(X,W) # W * X(t)
    H1 = np.tanh(z1+z2) #f (W*X(t)+Win*I(t))
    X = (1-leak)*X + leak*H1 # leaky integration
    scores=np.tanh(X)
    output=np.dot(scores,Wout)
    y_pred=np.clip(output, -action_bound, action_bound)

    return y_pred

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