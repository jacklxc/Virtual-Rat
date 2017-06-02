import numpy as npp
import minpy.numpy as np
from minpy.nn.model import ModelBase
from minpy.nn import layers

class SimplePolicyNetwork(ModelBase):
    def __init__(self, N=1, input_dim=3, hidden_dim=5, output_dim=3,
     gamma=0.99, reg=0.1, noise_factor = 0):  
        super(SimplePolicyNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.reg = reg
        self.noise_factor = noise_factor
        # Initialize h0
        self.h0 = np.random.randn(N, hidden_dim)
        self.h0 /= np.sqrt(N)
        self.h = self.h0

        # Initialize parameters for the RNN
        self.add_param(name = 'Wx', shape = (input_dim, hidden_dim))
        self.add_param(name = 'Wh', shape = (hidden_dim, hidden_dim))
        #self.Wh = np.zeros((hidden_dim, hidden_dim))
        self.add_param(name = 'b', shape = (hidden_dim,))
        #self.b = np.zeros((hidden_dim,))

        # Initialize output to vocab weights
        self.add_param(name = 'Wa', shape = (hidden_dim, output_dim))
        self.add_param(name = 'ba', shape = (output_dim,))
        #self.ba = np.zeros((output_dim,))

    def reset_h(self):
        self.h = self.h0

    def step_forward(self, X):
        """
        Each time step calls forward() once. RL's forward.
        Inputs:
        - X: (N,D) shape
        """
        N = X.shape[0]
        output_dim = 3
        self.h = layers.rnn_step(X[:,:], self.h, self.params['Wx'], self.params["Wh"], 
            self.params["b"])
        yy = layers.affine(self.h, self.params['Wa'], self.params["ba"]).reshape([N, 1, output_dim])
        numorator = np.exp(yy - np.max(yy, axis=2, keepdims=True))
        denomenator = np.sum(numorator,axis=2,keepdims=True)
        p = numorator/denomenator
        return p

    def forward(self, X, new_noise_factor = None):
        """
        Forward pass for the whole sequence, like supervised learning.
        Inputs:
        - X: (N, T, D) shape
        """
        N, T, _ = X.shape
        output_dim = 3
        y = np.zeros((N,0,output_dim))
        prev_h = self.h0
        hidden_dim = prev_h.shape[0]
        if new_noise_factor:
            noise_factor = new_noise_factor
        else:
            noise_factor = self.noise_factor
        for t in xrange(T):
            h = layers.rnn_step(X[:,t,:], prev_h, self.params['Wx'], self.params["Wh"], 
                self.params["b"])
            prev_h = h + noise_factor * np.random.normal(0,1,hidden_dim) # White noise with mu=0, sigma=1
            yy = layers.affine(h, self.params['Wa'], self.params["ba"]).reshape([N, 1, output_dim])
            y = np.append(y,yy,axis=1)

        numorator = np.exp(y - np.max(y, axis=2, keepdims=True))
        denomenator = np.sum(numorator,axis=2,keepdims=True)
        ps = numorator/denomenator
        return ps

    def choose_action(self, p):
        """
        Input:
        -p: (1,1,D) shape
        """
        u = npp.random.uniform()
        p = p[0,0,:]
        a = (npp.cumsum(p) > u).argmax()
        y = a
        return a, y

    def loss(self, ps, ys, rs):
        """
        Input
        - ps: (N, T, C)
        - ys: (T, N)
        - rs: (T, N)
        """
        N, T, C = ps.shape
        ps = np.maximum(1.0e-5, np.minimum(1.0 - 1e-5, ps))
        #convert it to one hot encoding
        onehot_label = np.zeros([N, T, C])
        for t in xrange(T):
            np.onehot_encode(ys[t,:], onehot_label[:,t,:])
        loss = -np.sum(np.sum(np.log(ps) * onehot_label, axis=2) * rs)
        regloss = loss#+ 0.5 * self.reg * np.sum(self.params['Wh'] * self.params['Wh'])
        return regloss

    def discount_rewards(self, rs):
        drs = npp.zeros_like(rs, dtype=npp.float)
        s = 0
        for t in reversed(xrange(0, len(rs))):
            # Reset the running sum at a game boundary.
            #if rs[t] != 0:
            #    s = 0
            s = s * self.gamma + rs[t]
            drs[t] = s
        #drs -= np.mean(drs)
        #drs /= np.std(drs)
        return drs



