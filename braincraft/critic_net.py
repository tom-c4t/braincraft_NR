import numpy as np

class CriticNet(object):
  """
  A Four-layer fully-connected neural network for critic network. 
  -The net has an input dimension of (N, S), with S being the cardinality of 
  the state space.
  
  - The net also has an input dimension of (N, A), where A is the cardinality of
  the action space.
  
  - There are three hidden layers, with dimension of H1, H2, and H3, respectively.  
  
  - The output provides an Q-value vetcor with the dimenson of A. 
  
  - The state input connect to the first hidden layer.
  
  - The action input bypass the first hidden layer and connected directly to 
  the second hidden layer
  
  - The outputs (from action and state) at the second hidden layer are summed up.
  
  - The network uses a ReLU nonlinearity for the first, second and 
    the third layer and uses leaner activation
    for the final layer. 
"""

  def __init__(self, input_size, hidden_size, output_size, weight=None):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    Win: Input layer weights; has shape (D, H)
    W: Hidden layer weights; has shape (H, H)
    Wout: Output layer weights, has shape (H, A)
    
    We also have the weights for a target network (same architecture but 
    different weights)
    Win_tgt: Input layer weights; has shape (D, H)
    W_tgt: Hidden layer weights; has shape (H, H)
    Wout_tgt: Output layer weights, has shape (H, A)


    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The continuous variables that constitutes an action vector
      of A dimension.
    """
    self.params = {}
    if weight == None:
        self.params['Win'] = self._uniform_init(input_size, hidden_size)
        self.params['W'] = self._uniform_init(hidden_size, hidden_size)
        self.params['Wout'] = self._uniform_init(hidden_size, output_size)
    else: 
        self.params['Win'] = weight['Win']
        self.params['W'] = weight['W']
        self.params['Wout'] = weight['Wout']
        
    # Initialization based on "Continuous control with deep reinformcement 
    # learning"
#    self.params['Win_tgt'] = self.params['Win']
#    self.params['b1_tgt'] = self.params['b1']
#    self.params['W_tgt'] = self.params['W']
#    self.params['b2_S_tgt'] = self.params['b2_S']
#    self.params['W2_A_tgt'] = self.params['W2_A']
#    self.params['b2_A_tgt'] = self.params['b2_A']
#    self.params['Wout_tgt'] = self.params['Wout']
#    self.params['b3_tgt'] = self.params['b3']
#    self.params['W4_tgt'] = self.params['W4']
#    self.params['b4_tgt'] = self.params['b4']

    self.params['Win_tgt'] = self._uniform_init(input_size, hidden_size)
    self.params['W_tgt'] = self._uniform_init(hidden_size, hidden_size)
    self.params['Wout_tgt'] = self._uniform_init(hidden_size, hidden_size)

    
    # Initialize the dictionary for optimization configuration for diffrent 
    # layers
    
    self.optm_cfg ={}
    self.optm_cfg['Win'] = None
    self.optm_cfg['W'] = None
    self.optm_cfg['Wout'] = None


  def evaluate_gradient(self, I, actions, leak, Y_tgt, use_target=False):
    """
    Compute the Q-value and gradients for the network based on the input X_S,
    X_A, Y_tgt
    
    Inputs:
    - X_S: Input for state, shape (N, S), N is the batch size
    - X_A: Input for actio, shape (N, A), N is the batch size
    _ Y_tgt: Target vaule for Q-value, used for update weights (via regression)
    - use_target: use default weights if False; otherwise use target weights.
    
   Returns:
     A tuple of:
    - Q_values: a continuous vector, has the same dimension as A
    - loss: 
    - grads: Dictionary mapping parameter names to gradients of those parameters; 
      has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    if not use_target:
        Win= self.params['Win']
        W= self.params['W']
        Wout= self.params['Wout']
    else:
        Win = self.params['Win_tgt']
        W = self.params['W_tgt']
        Wout = self.params['Wout_tgt']


    # Compute the forward pass
    # The first hidden layer
    output= None
    scores = None
    z1=np.dot(I,Win) # Win * I(t)
    z2=np.dot(X,W) # W * X(t)
    H1 = np.tanh(z1+z2) #f (W*X(t)+Win*I(t))
    X = (1-leak)*X + leak*H1 # leaky integration
    scores=np.tanh(X)
    output=np.dot(scores,Wout)

    Q_values=output # q_values is the scores in this case, due to the linear
                    # activation
    batch_size=np.shape(I)[0]       
     # loss
    loss =np.sum((Q_values-Y_tgt)**2, axis=0)/(1.0*batch_size)
    loss = loss[0]
    # error
    error=(Q_values-Y_tgt)   
    
    # Back-propagate to second hidden layer
    # as a sanity check out1 and z3 should have the same shape
    # Backward pass
    grads = {}
    
    grad_output=error*2/batch_size

    # 1. Gradient of output w.r.t Wout
    dWout = np.dot(scores.T, grad_output)
    
    # 2. Gradient through tanh(X_new)
    dscores = np.dot(grad_output, Wout.T)
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
    
                    
    return grads,loss #, Q_values
    
    
  def evaluate_action_gradient(self, I, actions, leak, use_target=False):
    """
    Inputs:
    - X_S: Input for state, shape (N, S), N is the batch size
    - X_A: Input for actio, shape (N, A), N is the batch size
    
    - use_target: use default weights if False; otherwise use target weights.
    
   Returns:
     A tuple of:
    - Q_values: a continuous vector, has the same dimension as A
    - loss: 
    - grads: Dictionary mapping parameter names to gradients of those parameters; 
      has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    if not use_target:
        Win = self.params['Win']
        W= self.params['W']
        Wout = self.params['Wout']
    else:
        Win = self.params['Win_tgt']
        W= self.params['W_tgt']
        Wout = self.params['Wout_tgt']
        
   
    # Compute the forward pass
    # The first hidden layer
    # Compute the forward pass
    # The first hidden layer
    output= None
    scores = None
    z1=np.dot(I,Win) # Win * I(t)
    z2=np.dot(X,W) # W * X(t)
    H1 = np.tanh(z1+z2) #f (W*X(t)+Win*I(t))
    X = (1-leak)*X + leak*H1 # leaky integration
    scores=np.tanh(X)
    output=np.dot(scores,Wout)
        
    grads_action = {}
    
    grad_output=error*2/batch_size

    # 1. Gradient of output w.r.t Wout
    dWout = np.dot(scores.T, grad_output)
    
    # 2. Gradient through tanh(X_new)
    dscores = np.dot(grad_output, Wout.T)
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
                    
    return grads_action  
    
    
    
  def train(self, X_S, X_A, Y_tgt):
    """
    Train this neural network using adam optimizer.
    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    """
    # Compute out and gradients using the current minibatch
    grads, loss = self.evaluate_gradient(X_S, X_A, Y_tgt, use_target=False)
    # Update the weights using adam optimizer
    #print ('grads W4', grads['W4'])
    self.params['W4'] = self._adam(self.params['W4'], grads['W4'], config=self.optm_cfg['W4'])[0]
    self.params['Wout'] = self._adam(self.params['Wout'], grads['Wout'], config=self.optm_cfg['Wout'])[0]
    self.params['W'] = self._adam(self.params['W'], grads['W'], config=self.optm_cfg['W'])[0]
    self.params['W2_A'] = self._adam(self.params['W2_A'], grads['W2_A'], config=self.optm_cfg['W2_A'])[0]    
    self.params['Win'] = self._adam(self.params['Win'], grads['Win'], config=self.optm_cfg['Win'])[0]
    self.params['b4'] = self._adam(self.params['b4'], grads['b4'], config=self.optm_cfg['b4'])[0]
    self.params['b3'] = self._adam(self.params['b3'], grads['b3'], config=self.optm_cfg['b3'])[0]
    self.params['b2_S'] = self._adam(self.params['b2_S'], grads['b2_S'], config=self.optm_cfg['b2_S'])[0]
    self.params['b2_A'] = self._adam(self.params['b2_A'], grads['b2_A'], config=self.optm_cfg['b2_A'])[0]
    self.params['b1'] = self._adam(self.params['b1'], grads['b1'], config=self.optm_cfg['b1'])[0]
    
    # Update the configuration parameters to be used in the next iteration
    self.optm_cfg['W4'] = self._adam(self.params['W4'], grads['W4'], config=self.optm_cfg['W4'])[1]
    self.optm_cfg['Wout'] = self._adam(self.params['Wout'], grads['Wout'], config=self.optm_cfg['Wout'])[1]
    self.optm_cfg['W'] = self._adam(self.params['W'], grads['W'], config=self.optm_cfg['W'])[1]
    self.optm_cfg['W2_A'] = self._adam(self.params['W2_A'], grads['W2_A'], config=self.optm_cfg['W2_A'])[1]
    self.optm_cfg['Win'] = self._adam(self.params['Win'], grads['Win'], config=self.optm_cfg['Win'])[1]
    self.optm_cfg['b4'] = self._adam(self.params['b4'], grads['b4'], config=self.optm_cfg['b4'])[1]
    self.optm_cfg['b3'] = self._adam(self.params['b3'], grads['b3'], config=self.optm_cfg['b3'])[1]
    self.optm_cfg['b2_S'] = self._adam(self.params['b2_S'], grads['b2_S'], config=self.optm_cfg['b2_S'])[1]
    self.optm_cfg['b2_A'] = self._adam(self.params['b2_A'], grads['b2_A'], config=self.optm_cfg['b2_A'])[1]
    self.optm_cfg['b1'] = self._adam(self.params['b1'], grads['b1'], config=self.optm_cfg['b1'])[1]

    return loss
    
    
 
  def train_target(self, tau):
    """
      Update the weights of the target network.
     -tau: coefficent for tracking the learned network.
     """
    self.params['W4_tgt'] = tau*self.params['W4']+(1-tau)*self.params['W4_tgt'] 
    self.params['Wout_tgt'] = tau*self.params['Wout']+(1-tau)*self.params['Wout_tgt']
    self.params['W_tgt'] = tau*self.params['W']+(1-tau)*self.params['W_tgt']
    self.params['W2_A_tgt'] = tau*self.params['W2_A']+(1-tau)*self.params['W2_A_tgt']
    self.params['Win_tgt'] = tau*self.params['Win']+(1-tau)*self.params['Win_tgt']
        
    self.params['b4_tgt'] = tau*self.params['b4']+(1-tau)*self.params['b4_tgt']
    self.params['b3_tgt'] = tau*self.params['b3']+(1-tau)*self.params['b3_tgt']
    self.params['b2_S_tgt'] = tau*self.params['b2_S']+(1-tau)*self.params['b2_S_tgt']
    self.params['b2_A_tgt'] = tau*self.params['b2_A']+(1-tau)*self.params['b2_A_tgt']
    self.params['b1_tgt'] = tau*self.params['b1']+(1-tau)*self.params['b1_tgt']


    
  def predict(self, X_S, X_A, target=False):
    """
    Use the trained weights of this network to predict the Q vector for a 
    given state.

    Inputs:
    - X: numpy array of shape (N, D) 
    - target: if False, use normal weights, otherwise use learned weight.
    - action_bound: the scaling factor for the action, which is environment
                    dependent.

    Returns:
    - y_pred: A numpy array of shape (N,) 
    
    """
    y_pred = None
    
    if not target:
        Win, b1 = self.params['Win'], self.params['b1']
        W, b2_S = self.params['W'], self.params['b2_S']
        W2_A, b2_A = self.params['W2_A'], self.params['b2_A']
        Wout, b3 = self.params['Wout'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

    else:
        Win, b1 = self.params['Win_tgt'], self.params['b1_tgt']
        W, b2_S = self.params['W_tgt'], self.params['b2_S_tgt']
        W2_A, b2_A = self.params['W2_A_tgt'], self.params['b2_A_tgt']
        Wout, b3 = self.params['Wout_tgt'], self.params['b3_tgt']
        W4, b4 = self.params['W4_tgt'], self.params['b4_tgt']

    H1 = np.maximum(0,X_S.dot(Win)+b1)
    
    H2 = np.dot(H1,W)+b2_S + np.dot(X_A, W2_A)+b2_A
    
    H3 = np.maximum(0,H2.dot(Wout)+b3)
    
    score=np.dot(H3, W4)+b4
    
    #print "scores=:", score
    y_pred=score
    

    return y_pred

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