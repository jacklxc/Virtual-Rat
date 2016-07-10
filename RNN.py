import numpy as np
from VirtualRatFunctions import *

class FirstRNN():
	"""
	One RNN for each rat. input dimention N * D * T, where N = 2 (rational agent and rat),
	D = 3 (pro_rule, target_on_right, trial_n=1), T is the length of the time sequence.
	Output dimention = 3, (left, right, central poke violation)
	"""
	def __init__(self, N=1, input_dim=3, wordvec_dim=3, hidden_dim=128, dtype=np.float32):
		self.data = None
		self.dtype = dtype
		self.params = {}

		# Initializa h0
		self.params['h0'] = np.random.zeros(N, hidden_dim) #########

		# Initialize word vectors
	    self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
	    self.params['W_embed'] /= 100

	    # Initialize parameters for the RNN
	    self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
	    self.params['W_proj'] /= np.sqrt(input_dim)
	    self.params['Wx'] = np.random.randn(wordvec_dim, hidden_dim)
	    self.params['Wx'] /= np.sqrt(wordvec_dim)
	    self.params['Wh'] = np.random.randn(hidden_dim, hidden_dim)
	    self.params['Wh'] /= np.sqrt(hidden_dim)
	    self.params['b'] = np.zeros(dim_mul * hidden_dim)

	    # Initialize output to vocab weights
		self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
		self.params['W_vocab'] /= np.sqrt(hidden_dim)
		self.params['b_vocab'] = np.zeros(vocab_size)

	    # Cast parameters to correct dtype
    	for k, v in self.params.iteritems():
      		self.params[k] = v.astype(self.dtype)

    def loss(self, inputs, captions):
    	captions_in = captions[:, :-1]
    	captions_out = captions[:, 1:]

	    mask = np.ones(captions_out.shape)

	    # Weight and bias for the affine transform from image features to initial
	    # hidden state
	    W_proj = self.params['W_proj']
	    
	    # Word embedding matrix
	    W_embed = self.params['W_embed']

	    # Input-to-hidden, hidden-to-hidden, and biases for the RNN
	    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

	    # Weight and bias for the hidden-to-vocab transformation.
	    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
	    
	    loss, grads = 0.0, {}

	    # Forward pass
	    h0 = self.params['h0']
	    x, cache_x = word_embedding_forward(captions_in, W_embed)

	    h, cache_h = rnn_forward(x,h0, inputs, Wx, Wh, W_proj,b)

	    scores, cache_scores = temporal_affine_forward(h, W_vocab, b_vocab)

	    loss, dscores = temporal_softmax_loss(scores,captions_out,mask)

	    # Back prop
	    dh, grads['W_vocab'],grads['b_vocab'] = temporal_affine_backward(dscores,cache_scores)


	    dx, grads['h0'], dinput, grads['Wx'], grads['Wh'], grads['W_proj'] grads['b'] = rnn_backward(dh ,cache_h)
	    

	    grads['W_embed'] = word_embedding_backward(dx, cache_x)

	    return loss, grads


	 def sample(self, signal, max_length=3):
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
		N = signal.shape[0]
		captions = np.ones((N, max_length), dtype=np.int32) ########

		# Unpack parameters
		h = self.params['h0']
		W_proj = self.params['W_proj']
		W_embed = self.params['W_embed']
		Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
		W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

		###########################################################################
		# TODO: Implement test-time sampling for the model. You will need to      #
		# initialize the hidden state of the RNN by applying the learned affine   #
		# transform to the input image features. The first word that you feed to  #
		# the RNN should be the <START> token; its value is stored in the         #
		# variable self._start. At each timestep you will need to do to:          #
		# (1) Embed the previous word using the learned word embeddings           #
		# (2) Make an RNN step using the previous hidden state and the embedded   #
		#     current word to get the next hidden state.                          #
		# (3) Apply the learned affine transformation to the next hidden state to #
		#     get scores for all words in the vocabulary                          #
		# (4) Select the word with the highest score as the next word, writing it #
		#     to the appropriate slot in the captions variable                    #
		#                                                                         #
		# For simplicity, you do not need to stop generating after an <END> token #
		# is sampled, but you can if you want to.                                 #
		#                                                                         #
		# HINT: You will not be able to use the rnn_forward or lstm_forward       #
		# functions; you'll need to call rnn_step_forward or lstm_step_forward in #
		# a loop.                                                                 #
		###########################################################################
		D,H = Wx.shape

		x = np.ones((N,D)) #####

		for t in range(max_length):
		    h, _ = rnn_step_forward(x, h, signal, Wx, Wh, W_proj, b)
		    scores, _ = affine_forward(h,W_vocab,b_vocab)
		    p = np.exp(scores)/np.sum(np.exp(scores))
		    #####################################
		    max_word = np.argmax(p,axis = 1) #####       
		    x = W_embed[max_word]
		    captions[:,t] = max_word
		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################
		return captions







