import numpy as npp
import minpy.numpy as np
from minpy.nn import layers
from minpy.nn.model import ModelBase

class GRURNN(ModelBase):
    """
    This is a virtual rat vanilla RNN that simulates rats' behavior in Duan et. al's paper.

    In each cycle, the RNN receives input vectors of size D and returns an index that indicates
    left, right or cpv (center poke violation).
    """
    def __init__(self, N=1, input_dim=3, output_dim=3, hidden_dim=5, reg = 0, p = 1):
        """
        Construct a new FirstRNN instance.

        Inputs:
        - N: Number of simultaneously trained virtual rat.
        - input_dim: Dimension D of input signal vectors.
        - output_dim: Dimension O of output signal vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - params: None or a dictionary of parameters whose format is the same as self.params.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        super(GRURNN, self).__init__()
        self.N = N
        self.reg = reg
        self.p = p
        # Initialize h0
        self.add_param(name = 'h0', shape = (N, hidden_dim))
        # Initialize parameters for the RNN
        self.add_param(name = 'Wx', shape = (input_dim, 2*hidden_dim))
        self.add_param(name = 'Wh', shape = (hidden_dim, 2*hidden_dim))
        self.add_param(name = 'b', shape = (2*hidden_dim,))
        self.add_param(name = 'Wxh', shape = (input_dim, hidden_dim))
        self.add_param(name = 'Whh', shape = (hidden_dim, hidden_dim))
        self.add_param(name = 'bh', shape = (hidden_dim,))
        # Initialize output to vocab weights
        self.add_param(name = 'Wa', shape = (hidden_dim, output_dim))
        self.add_param(name = 'ba', shape = (output_dim,))


    def loss(self, x, y):
        """
        Compute training-time loss for the RNN.

        Inputs:
        - x: Input data of shape (N, T, D)
        - y: Ground truth output of shape (N, T, O)

        Returns a tuple of:
        - loss: Scalar loss
        """
        N, T, _ = x.shape
        mask = np.ones((N, T),dtype=npp.bool)
        translator = np.array([0,1,2],dtype=npp.int)
        yy = np.sum(y*translator,axis=2)
        softmax = layers.temporal_softmax_loss(x, yy,mask)
        loss = softmax + 0.5 * self.reg * np.sum(self.params['Wh'] * self.params['Wh']) \
        + 0.5 * self.reg * np.sum(self.params['Whh'] * self.params['Whh'])
        return loss

    def forward(self, X, mode):
        """
        Run a test-time forward pass for the model, predicting for input x.

        At each timestep, we embed a new input verctor, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all choices, and choose the one with the highest score to output.
        The initial hidden state is computed by applying an affine
        transform to the input vectors.

        Inputs:
        - x: Array of input signal features of shape (N, D).

        Returns:
        - y: Array of shape (N, T) giving sampled y,
          where each element is an integer in the range [0, V). 
        - probs: Array of shape (N, T, DD) giving the probability of each choice,
          where each element is a float. 
        """
        N, T, _ = X.shape

        hidden_dim = self.params['Wh'].shape[0]
        output_dim =3
        y = np.zeros((N, 0, output_dim))
        prev_h = self.params['h0']
        self.h = np.zeros((T,N,hidden_dim))

        if mode == "train":
            p = self.p
        else:
            p = 1

        for t in xrange(T):
            mask = (np.random.rand(*prev_h.shape) < p) / p
            h = layers.gru_step(X[:, t, :], prev_h, self.params['Wx'], self.params['Wh'], 
                self.params['b'], self.params['Wxh'], self.params['Whh'], self.params['bh'])
            prev_h = h * mask
            self.h[t,:,:] = h
            yy = layers.affine(h, self.params['Wa'], self.params['ba']).reshape([N, 1, output_dim])
            y = np.append(y,yy,axis=1)
        return y



