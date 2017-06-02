import copy
import numpy as np
from VirtualRatFunctions import *

class SimpleRNN(object):
    """
    This is the first virtual rat RNN that simulates rats' behavior in Duan et. al's paper.

    In each cycle, the RNN receives input vectors of size D and returns an index that indicates
    left, right or cpv (center poke violation).
    """
    def __init__(self, N=1, input_dim=3, output_dim=3, hidden_dim=5, 
        params = None, dtype=np.float32):
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
        self.dtype = dtype
        if params is None:
            self.params = {}

            # Initializa h0
            self.params['h0'] = np.zeros((N, hidden_dim))

            # Initialize parameters for the RNN
            self.params['Wx'] = np.random.randn(output_dim, hidden_dim)
            self.params['Wx'] /= np.sqrt(output_dim)
            self.params['Wh'] = np.random.randn(hidden_dim, hidden_dim)
            self.params['Wh'] /= np.sqrt(hidden_dim)
            self.params['b'] = np.zeros(hidden_dim)

            # Initialize output to vocab weights
            self.params['W_vocab'] = np.random.randn(hidden_dim, output_dim)
            self.params['W_vocab'] /= np.sqrt(hidden_dim)
            self.params['b_vocab'] = np.zeros(output_dim)

            # Cast parameters to correct dtype
            for k, v in self.params.iteritems():
                self.params[k] = v.astype(self.dtype)

        else:
            self.params = params
        
        # Save a copy of the initial weights
        self.initparams = copy.deepcopy(self.params)
    def loss(self, x, y):
        """
        Compute training-time loss for the RNN.

        Inputs:
        - x: Input data of shape (N, T, D)
        - y: Ground truth output of shape (N, T, O)

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        y = y.astype(int)

        mask = np.ones(y.shape)

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
        
        loss, grads = 0.0, {}

        # Forward pass
        h0 = self.params['h0']

        h, cache_h = rnn_forward(x, h0, Wx, Wh,b)

        scores, cache_scores = temporal_affine_forward(h, W_vocab, b_vocab)

        loss, dscores = temporal_softmax_loss(scores, y, mask)

        # Back prop
        dh, grads['W_vocab'],grads['b_vocab'] = temporal_affine_backward(dscores,cache_scores)

        dx, grads['h0'], grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(dh ,cache_h)
        

        return loss, grads


    def predict(self, x):
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
        N, T, _= x.shape

        DD = 3 # index 0 is the prob to choose left, 1 is the prob 
               # to choose right, 2 is cpv.

        y = np.zeros((N, T), dtype=np.int32) 
        probs = np.zeros((N, T, DD))

        # Unpack parameters
        h = self.params['h0']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        for t in range(T):
            h, _ = rnn_step_forward(x[:,t,:], h, Wx, Wh, b)
            scores, _ = affine_forward(h,W_vocab,b_vocab)
            p = np.exp(scores)/np.sum(np.exp(scores))
            max_word = np.argmax(p,axis = 1)        
            y[:,t] = max_word
            probs[:,t,:] = p
        return y, probs





