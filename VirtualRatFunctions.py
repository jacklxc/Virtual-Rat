from helpers import DBUtilsClass as db
import numpy as np

def getData():
	CONN = db.Connection()
	CONN.use('pa')
	out = zip(*CONN.query('explain alldata'))
	all_rats = CONN.query('select distinct(ratname) from pa.alldata')

	allRatsData = {}

	for rat in all_rats:
		sqlstr=('select pro_rule, target_on_right, trial_n=1, '
			 	'(cpv=0 AND WR=0) as `left`, (cpv=0 AND WR = 1) as `right`, cpv '
			 	'from pa.alldata where ratname=%s order by sessid, trial_n')

		out = CONN.query(sqlstr, (str(rat[0]),))
		data = np.array(out)
		allRatsData[str(rat[0])] = data
		print rat[0], data.shape
	return allRatsData

def affine_forward(x, w, b):
	"""
	Computes the forward pass for an affine (fully-connected) layer.

	The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
	We multiply this against a weight matrix of shape (D, M) where
	D = \prod_i d_i

	Inputs:
	x - Input data, of shape (N, d_1, ..., d_k)
	w - Weights, of shape (D, M)
	b - Biases, of shape (M,)

	Returns a tuple of:
	- out: output, of shape (N, M)
	- cache: (x, w, b)
	"""
	out = x.reshape(x.shape[0], -1).dot(w) + b
	cache = (x, w, b)
	return out, cache


def affine_backward(dout, cache):
	"""
	Computes the backward pass for an affine layer.

	Inputs:
	- dout: Upstream derivative, of shape (N, M)
	- cache: Tuple of:
	- x: Input data, of shape (N, d_1, ... d_k)
	- w: Weights, of shape (D, M)

	Returns a tuple of:
	- dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
	- dw: Gradient with respect to w, of shape (D, M)
	- db: Gradient with respect to b, of shape (M,)
	"""
	x, w, b = cache
	dx = dout.dot(w.T).reshape(x.shape)
	dw = x.reshape(x.shape[0], -1).T.dot(dout)
	db = np.sum(dout, axis=0)
	return dx, dw, db


def relu_forward(x):
	"""
	Computes the forward pass for a layer of rectified linear units (ReLUs).

	Input:
	- x: Inputs, of any shape

	Returns a tuple of:
	- out: Output, of the same shape as x
	- cache: x
	"""
	out = np.maximum(0, x)
	cache = x
	return out, cache


def relu_backward(dout, cache):
	"""
	Computes the backward pass for a layer of rectified linear units (ReLUs).

	Input:
	- dout: Upstream derivatives, of any shape
	- cache: Input x, of same shape as dout

	Returns:
	- dx: Gradient with respect to x
	"""
	x = cache
	dx = np.where(x > 0, dout, 0)
	return dx



def rnn_step_forward(x, prev_h, s, Wx, Wh, W_proj, b):
	"""
	Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
	activation function.

	The input data has dimension D, the hidden state has dimension H, and we use
	a minibatch size of N.

	Inputs:
	- x: Input data for this timestep, of shape (N, D).
	- prev_h: Hidden state from previous timestep, of shape (N, H)
	- Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
	- Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
	- b: Biases of shape (H,)

	Returns a tuple of:
	- next_h: Next hidden state, of shape (N, H)
	- cache: Tuple of values needed for the backward pass.
	"""
	next_h, cache = None, None
	##############################################################################
	# TODO: Implement a single forward step for the vanilla RNN. Store the next  #
	# hidden state and any values you need for the backward pass in the next_h   #
	# and cache variables respectively.                                          #
	##############################################################################
	xWx = x.dot(Wx)
	hWh = prev_h.dot(Wh)
	sW = s.dot(W_proj)
	H = xWx + hWh + sW + b
	next_h = np.tanh(H)
	cache = (x,prev_h, s, Wx, Wh, W_proj, next_h)
	##############################################################################
	#                               END OF YOUR CODE                             #
	##############################################################################
	return next_h, cache


def rnn_step_backward(dnext_h, cache):
	"""
	Backward pass for a single timestep of a vanilla RNN.

	Inputs:
	- dnext_h: Gradient of loss with respect to next hidden state
	- cache: Cache object from the forward pass

	Returns a tuple of:
	- dx: Gradients of input data, of shape (N, D)
	- dprev_h: Gradients of previous hidden state, of shape (N, H)
	- dWx: Gradients of input-to-hidden weights, of shape (N, H)
	- dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
	- db: Gradients of bias vector, of shape (H,)
	"""
	dx, dprev_h, ds, dWx, dWh, dW_proj, db = None, None, None, None, None, None, None
	##############################################################################
	# TODO: Implement the backward pass for a single step of a vanilla RNN.      #
	#                                                                            #
	# HINT: For the tanh function, you can compute the local derivative in terms #
	# of the output value from tanh.                                             #
	##############################################################################
	x,prev_h, s, Wx, Wh, W_proj, next_h = cache
	dH = dnext_h * (1 - np.square(next_h))
	db = np.sum(dH,axis=0)
	dWx = x.T.dot(dH)
	dx = dH.dot(Wx.T)
	dprev_h = dH.dot(Wh.T)
	dWh = prev_h.T.dot(dH)
	dW_proj = s.T.dot(dH)
	ds = dH.dot(W_proj.T)
	##############################################################################
	#                               END OF YOUR CODE                             #
	##############################################################################
	return dx, dprev_h, ds, dWx, dWh, dW_proj, db


def rnn_forward(x, h0, s, Wx, Wh, W_proj, b):
	"""
	Run a vanilla RNN forward on an entire sequence of data. We assume an input
	sequence composed of T vectors, each of dimension D. The RNN uses a hidden
	size of H, and we work over a minibatch containing N sequences. After running
	the RNN forward, we return the hidden states for all timesteps.

	Inputs:
	- x: Input data for the entire timeseries, of shape (N, T, D).
	- h0: Initial hidden state, of shape (N, H)
	- Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
	- Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
	- b: Biases of shape (H,)

	Returns a tuple of:
	- h: Hidden states for the entire timeseries, of shape (N, T, H).
	- cache: Values needed in the backward pass
	"""
	h, cache = None, None
	##############################################################################
	# TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
	# input data. You should use the rnn_step_forward function that you defined  #
	# above.                                                                     #
	##############################################################################
	N, T, _ = x.shape
	_, H = h0.shape
	h = np.empty((N,T,H))
	prev_h = h0
	cache = []
	for t in xrange(T):
		h[:,t,:], Cache = rnn_step_forward(x[:,t,:],prev_h,s[:,t,:],Wx,Wh, W_proj, b)
		prev_h = h[:,t,:]
		cache.append(Cache)
	##############################################################################
	#                               END OF YOUR CODE                             #
	##############################################################################
	return h, cache


def rnn_backward(dh, cache):
	"""
	Compute the backward pass for a vanilla RNN over an entire sequence of data.

	Inputs:
	- dh: Upstream gradients of all hidden states, of shape (N, T, H)

	Returns a tuple of:
	- dx: Gradient of inputs, of shape (N, T, D)
	- dh0: Gradient of initial hidden state, of shape (N, H)
	- dWx: Gradient of input-to-hidden weights, of shape (D, H)
	- dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
	- db: Gradient of biases, of shape (H,)
	"""
	dx, dprev_h, ds, dWx, dWh, dW_proj, db = None, None, None, None, None, None, None
	##############################################################################
	# TODO: Implement the backward pass for a vanilla RNN running an entire      #
	# sequence of data. You should use the rnn_step_backward function that you   #
	# defined above.                                                             #
	##############################################################################
	N,T,H = dh.shape  
	x, prev_h, s, Wx, Wh, W_proj, next_h = cache[0]
	_,D = x.shape

	dx = np.zeros((N,T,D))
	dh0 = np.zeros((N,H))
	ds = np.zeros((N,T,D))
	dWx = np.zeros((D,H))
	dWh = np.zeros((H,H))
	dW_proj = np.zeros(W_proj.shape)
	db = np.zeros((H,))
	dprev_h = np.zeros((N,H))

	for t in reversed(xrange(T)):
		dx[:,t,:], dprev_h, ds, tdWx, tdWh, tdW_proj, tdb = rnn_step_backward(dh[:,t,:] + dprev_h ,cache[t])
		dWx += tdWx
		dWh += tdWh
		dW_proj += tdW_proj
		db += tdb
	dh0 = dprev_h
	##############################################################################
	#                               END OF YOUR CODE                             #
	##############################################################################
	return dx, dh0, ds, dWx, dWh, dW_proj, db

def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################
  V, D = W.shape
  N, T = x.shape
  out = W[x[np.arange(N)]]
  cache = (x,W)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
	"""
	Backward pass for word embeddings. We cannot back-propagate into the words
	since they are integers, so we only return gradient for the word embedding
	matrix.

	HINT: Look up the function np.add.at

	Inputs:
	- dout: Upstream gradients of shape (N, T, D)
	- cache: Values from the forward pass

	Returns:
	- dW: Gradient of word embedding matrix, of shape (V, D).
	"""
	dW = None
	##############################################################################
	# TODO: Implement the backward pass for word embeddings.                     #
	#                                                                            #
	# HINT: Look up the function np.add.at                                       #
	##############################################################################
	x, W = cache
	V, D = W.shape
	N, T = x.shape
	dW = np.zeros((V,D))
	for t in range(T):
		temp = np.zeros((N,V))
		temp[np.arange(N),x[:,t]] = 1
		dW += temp.T.dot(dout[:,t,:])
	##############################################################################
	#                               END OF YOUR CODE                             #
	##############################################################################
	return dW

def temporal_affine_forward(x, w, b):
	"""
	Forward pass for a temporal affine layer. The input is a set of D-dimensional
	vectors arranged into a minibatch of N timeseries, each of length T. We use
	an affine function to transform each of those vectors into a new vector of
	dimension M.

	Inputs:
	- x: Input data of shape (N, T, D)
	- w: Weights of shape (D, M)
	- b: Biases of shape (M,)

	Returns a tuple of:
	- out: Output data of shape (N, T, M)
	- cache: Values needed for the backward pass
	"""
	N, T, D = x.shape
	M = b.shape[0]
	out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
	cache = x, w, b, out
	return out, cache


def temporal_affine_backward(dout, cache):
	"""
	Backward pass for temporal affine layer.

	Input:
	- dout: Upstream gradients of shape (N, T, M)
	- cache: Values from forward pass

	Returns a tuple of:
	- dx: Gradient of input, of shape (N, T, D)
	- dw: Gradient of weights, of shape (D, M)
	- db: Gradient of biases, of shape (M,)
	"""
	x, w, b, out = cache
	N, T, D = x.shape
	M = b.shape[0]

	dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
	dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
	db = dout.sum(axis=(0, 1))

	return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
	"""
	A temporal version of softmax loss for use in RNNs. We assume that we are
	making predictions over a vocabulary of size V for each timestep of a
	timeseries of length T, over a minibatch of size N. The input x gives scores
	for all vocabulary elements at all timesteps, and y gives the indices of the
	ground-truth element at each timestep. We use a cross-entropy loss at each
	timestep, summing the loss over all timesteps and averaging across the
	minibatch.

	As an additional complication, we may want to ignore the model output at some
	timesteps, since sequences of different length may have been combined into a
	minibatch and padded with NULL tokens. The optional mask argument tells us
	which elements should contribute to the loss.

	Inputs:
	- x: Input scores, of shape (N, T, V)
	- y: Ground-truth indices, of shape (N, T) where each element is in the range
	   0 <= y[i, t] < V
	- mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
	the scores at x[i, t] should contribute to the loss.

	Returns a tuple of:
	- loss: Scalar giving loss
	- dx: Gradient of loss with respect to scores x.
	"""

	N, T, V = x.shape

	x_flat = x.reshape(N * T, V)
	y_flat = y.reshape(N * T)
	mask_flat = mask.reshape(N * T)

	probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
	probs /= np.sum(probs, axis=1, keepdims=True)
	loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
	dx_flat = probs.copy()
	dx_flat[np.arange(N * T), y_flat] -= 1
	dx_flat /= N
	dx_flat *= mask_flat[:, None]

	if verbose: print 'dx_flat: ', dx_flat.shape

	dx = dx_flat.reshape(N, T, V)

	return loss, dx





