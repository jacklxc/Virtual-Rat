import numpy as np
from VirtualRatFunctions import *

class FirstRNN(object):
    def __init__(self, N=1, input_dim=3, output_dim=3, hidden_dim=10, dtype=np.float32):
        self.data = None
        self.dtype = dtype
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

    def loss(self, x, captions):
        """
        Inputs:
        - x: Input data of shape (N, T, input_dim)
        - captions: Ground truth output of shape (N, T, output_dim)
        """
        captions = captions.astype(int)

        mask = np.ones(captions.shape)

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
        
        loss, grads = 0.0, {}

        # Forward pass
        h0 = self.params['h0']

        h, cache_h = rnn_forward(x, h0, Wx, Wh,b)

        scores, cache_scores = temporal_affine_forward(h, W_vocab, b_vocab)

        loss, dscores = temporal_softmax_loss(scores, captions, mask)

        # Back prop
        dh, grads['W_vocab'],grads['b_vocab'] = temporal_affine_backward(dscores,cache_scores)

        dx, grads['h0'], grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(dh ,cache_h)
        

        return loss, grads


    def sample(self, x, max_length=3):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.

        Inputs:
        - signal: Array of input signal features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = x.shape[0]
        captions = np.ones((N, max_length), dtype=np.int32) ########

        # Unpack parameters
        h = self.params['h0']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        for t in range(max_length):
            h, _ = rnn_step_forward(x, h, Wx, Wh, b)
            scores, _ = affine_forward(h,W_vocab,b_vocab)
            p = np.exp(scores)/np.sum(np.exp(scores))
            max_word = np.argmax(p,axis = 1)        
            captions[:,t] = max_word

        return captions





