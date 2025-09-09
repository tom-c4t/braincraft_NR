import numpy as np

class CriticNet(object):
  """
  A Four-layer fully-connected neural network for critic network. 
  -The net has an input dimension of (N, S), with S being the cardinality of 
  the state space.
  
  - The net also has an input dimension of (N, A), where A is the cardinality of
  the action space.
  
  - There are one hidden layer, with dimension of hidden_size.
  
  - The state input connect to the first layer.
  
  - The outputs (from action and state) at the second hidden layer are summed up.
  
  - The network uses tanh for the first layer and uses tanh activation
    for the final layer. 
"""

  def __init__(self, input_size, hidden_size, output_size):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the folloW1g keys:

    W1: Input layer weights; has shape (input_size, hidden_size)
    W: Hidden layer weights; has shape (hidden_size, hidden_size)
    W2: Output layer weights, has shape (hidden_size, output_size)
    
    We also have the weights for a target network (same architecture but 
    different weights)
    W1_tgt: Input layer weights; has shape (input_size, hidden_size)
    W_tgt: Hidden layer weights; has shape (hidden_size, hidden_size)
    W2_tgt: Output layer weights, has shape (hidden_size, output_size)


    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The continuous variables that constitutes an action vector
      of A dimension.
    """
    self.W1 = self._uniform_init(input_size + 1, hidden_size)
    self.W2 = self._uniform_init(hidden_size, output_size)
    self.X = None       # state of hidden neurons in critic 
    self.W1_tgt = self._uniform_init(input_size + 1, hidden_size)
    self.W2_tgt = self._uniform_init(hidden_size, output_size)
    self.X_tgt = None   # state of hidden neurons in target critic
    
    self.optm_cfg_W1 = None
    self.optm_cfg_W2 = None

    # additional params:
    self.gamma = 0.9  # discount factor for Q-learning
    self.lr = 0.003  # learning rate for the critic network


  def evaluate_gradient(self, I, action, rewards, Y_tgt):
    """
    Compute the Q-value and gradients for the network based on the input I, action , Y_tgt
    
    Inputs:
    - I: Input for state
    - action: Input for action
    _ Y_tgt: Target vaule for Q-value, used for update weights (via regression)
    - batch_size
    
   Returns:
    - Q values from critic
    - gradients of the weights and biases in the network
    """
    # predict Q value for state-action pair
    critic_Qs = self.predict(I, action)

    td_error = (Y_tgt - critic_Qs)  # shape: (batch_size, output_size)

    batch_size = I.shape[0]
    # Compute the gradients

    h = np.tanh(np.concatenate([I, action], axis=1) @ self.W1)
    dQ_dW2 = h.T @ td_error / batch_size

    dh = td_error @ self.W2.T * (1 - h**2)  # shape: (batch_size, hidden_size)
    x = np.concatenate([I, action], axis=1)
    dQ_dW1 = x.T @ dh / batch_size

    grads = (dQ_dW1, dQ_dW2)

    return critic_Qs, grads
  

  # gradient of the Q value for a action with the weights from the critic 
  def evaluate_action_gradient(self, I, action, loss, use_target=False):
    """
    Inputs:
    - I: Input for state, shape (N, S), N is the batch size
    - action: Input for action, shape (N, A), N is the batch size
    
    - use_target: use default weights if False; otherwise use target weights.
    
   Returns:
    - grads_action: gradient of the Q value for a action with the weights from the critic
    """
    # Unpack variables from the params dictionary
    if not use_target:
        W1 = self.params['W1']
        W2 = self.params['W2']
    else:
        W1 = self.params['W1_tgt']
        W2 = self.params['W2_tgt']
        
    critic_Qs = self.predict(I, action)

    td_error = (loss - critic_Qs)  # shape: (batch_size, output_size)

    # Compute the gradients

    h = np.tanh(np.concatenate([I, action], axis=1) @ self.W1)

    dQ_da = (td_error @ self.W2.T * (1- h**2)) @ self.W1[-1]
                    
    return dQ_da
    
    
  # train critic network so that estimated Q values are close to true Q values
  def train(self, I, action, reward, I_t1, Y_tgt):
    """
    Train this neural network using adam optimizer.
    Inputs:
    - I: A numpy array of shape (N, D) giving training data.
    """
    # Compute forward pass and gradients using the current minibatch
    critic_Qs, grads_critic = self.evaluate_gradient(I, action, reward, I_t1, Y_tgt)
    # calculate the loss update which is then summed up in ddpg_numpy.py
    critic_loss_update = (Y_tgt - critic_Qs)**2
    # Update the weights using adam optimizer
    #print ('grads W4', grads['W4'])
    self.W2 = self._adam(self.W2, grads_critic[1], config=self.optm_cfg_W2)[0]
    self.W1 = self._adam(self.W1, grads_critic[0], config=self.optm_cfg_W1)[0]
    
    # Update the configuration parameters to be used in the next iteration
    self.optm_cfg_W2 = self._adam(self.W2, grads_critic[1], config=self.optm_cfg_W2)[1]
    self.optm_cfg_W1 = self._adam(self.W1, grads_critic[0], config=self.optm_cfg_W1)[1]

    return critic_loss_update
    
    
 
  def train_target(self, tau):
    """
      Update the weights of the target network.
     -tau: coefficent for tracking the learned network.
     """
    self.params['W2_tgt'] = tau*self.params['W2']+(1-tau)*self.params['W2_tgt']
    self.params['W1_tgt'] = tau*self.params['W1']+(1-tau)*self.params['W1_tgt']
        
    
  def predict(self, I, action, target=False):
    """
    Use the trained weights of this network to predict the Q vector for a 
    given state.

    Inputs:
    - I: states of shape (D) 
    - action: scalar action
    - target: if False, use normal weights, otherwise use learned weight.

    Returns:
    - Q: Q value for the (state, action) pair 
    
    """
    if not target:
        W1 = self.W1
        W2= self.W2
    else:
        W1 = self.W1_tgt
        W2 = self.W2_tgt

    x = np.concatenate([I, action], axis=1)
    h = np.tanh(x @  W1)
    q_value = h @ W2
    return q_value


  def _adam(self, x, dx, config=None):
      """
      Uses the Adam update rule, which incorporates moving averages of both the
      gradient and its square and a bias correction term.
    
      config format:
      - learning_rate: scalar learning rate.
      - beta1: decay rate for moving average of first moment of gradient.
      - beta2: decay rate for moving average of second moment of gradient.
      - epsilon: small scalar used for smoothing to avoid dividing by zero.
      - m: moving average of gradient.
      - v: moving average of squared gradient.
      - t: iteration number (time step)
      """
      if config is None: config = {}
      config.setdefault('learning_rate', 1e-3)
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