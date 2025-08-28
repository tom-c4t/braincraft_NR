import numpy as np

class ActorNet:

    def __init__(self, input_size, hidden_size, output_size, l):
        self.params = {}
        # weight matrices for value network
        self.params['Win'] = self.uniform_init(input_size, hidden_size)
        self.params['W'] = self.uniform_init(hidden_size, hidden_size)
        self.params['Wout'] = self.uniform_init(hidden_size, output_size)

        #weight matrices for target network
        self.params['Win_tgt'] = self.uniform_init(input_size, hidden_size)
        self.params['W_tgt'] = self.uniform_init(hidden_size, hidden_size)
        self.params['Wout_tgt'] = self.uniform_init(hidden_size, output_size)

        self.optm_cfg = {}
        self.optm_cfg['Win'] = None
        self.optm_cfg['W'] = None
        self.optm_cfg['Wout'] = None

        self.hidden_states = np.random.randn(hidden_size)
        self.l = l
    
    # forward gradient, compute the output from the input
    # always use tanh for g to be able to constrain the output actions between -5 and 5
    def evaluate_gradient(self, I, action_grads, f, g, diff_f, diff_g, target = False):
        if not target:
            Win = self.params['Win']
            W = self.params['W']
            Wout = self.params['Wout']
        else:
            Win = self.params['Win_tgt']
            W = self.params['W_tgt']
            Wout = self.params['Wout_tgt']

        batch_size, _ = I.shape

        hidden_input = np.dot(W, self.hidden_states) + np.dot(Win, I)
        self.hidden_states = (1-self.l) * self.hidden_states + self.l * f(hidden_input)
        # TODO: contrain actions to values between -5 and 5
        actions = np.dot(Wout, g(self.hidden_states))
        
        # Compute gradients
        grads = {}
        grad_output = np.dot(diff_g(self.hidden_states)*(-action_grads), Wout.T)
        grad_hidden = (1-self.l) + self.l * np.dot(diff_f(hidden_input), W.T) 
        grad_input = self.l * np.dot(diff_f(hidden_input), Win.T)

        grads['Wout'] = grad_output/batch_size
        grads['W'] = grad_hidden/batch_size
        grads['Win'] = grad_input/batch_size

        return actions, grads
    
    # Update the weights of the target network
    def train_target(self, tau):
        self.params['Win_tgt'] = tau * self.params['Win'] + (1-tau) * self.params['Win_tgt']
        self.params['W_tgt'] = tau * self.params['W'] + (1-tau) * self.params['W_tgt']
        self.params['Wout_tgt'] = tau * self.params['Wout'] + (1-tau) * self.params['Wout_tgt']

    

    # Helper functions

    # uniformly initializes the weights to optimal initialization values
    def uniform_init(self, size_in, size_out):
        u = np.sqrt(6./(size_in+size_out))
        return np.random.uniform(-u,u,(size_in,size_out))
    
    # Activation functions

    def relu(self, x):
        return max(0, x)
    
    def sigmoid (self, x):
        return 1/(1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    # Derivatives of Activation functions

    def diff_tanh(self, x):
        return 1 - np.tanh(x)**2
    
    def diff_sigmoid(self, x):
        return self.sigmoid(x) * (1-self.sigmoid(x))
    
    def diff_relu(self, x):
        if self.relu(x) == 0:
            return 0
        else:
            return 1