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

def rnn_step_forward(x, prev_h, Wx, Wh, b):
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
	H = xWx + hWh + b
	next_h = np.tanh(H)
	cache = (x,prev_h,Wx,Wh,next_h)
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
	dx, dprev_h, dWx, dWh, db = None, None, None, None, None
	##############################################################################
	# TODO: Implement the backward pass for a single step of a vanilla RNN.      #
	#                                                                            #
	# HINT: For the tanh function, you can compute the local derivative in terms #
	# of the output value from tanh.                                             #
	##############################################################################
	x,prev_h,Wx,Wh,next_h = cache
	dH = dnext_h * (1 - np.square(next_h))
	db = np.sum(dH,axis=0)
	dWx = x.T.dot(dH)
	dx = dH.dot(Wx.T)
	dprev_h = dH.dot(Wh.T)
	dWh = prev_h.T.dot(dH)
	##############################################################################
	#                               END OF YOUR CODE                             #
	##############################################################################
	return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
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
	N, T, D = x.shape
	_, H = h0.shape
	h = np.empty((N,T,H))
	prev_h = h0
	cache = []
	for t in xrange(T):
	h[:,t,:], Cache = rnn_step_forward(x[:,t,:],prev_h,Wx,Wh,b)
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
	dx, dh0, dWx, dWh, db = None, None, None, None, None
	##############################################################################
	# TODO: Implement the backward pass for a vanilla RNN running an entire      #
	# sequence of data. You should use the rnn_step_backward function that you   #
	# defined above.                                                             #
	##############################################################################
	N,T,H = dh.shape  
	x, prev_h, Wx, Wh, next_h = cache[0]
	_,D = x.shape

	dx = np.zeros((N,T,D))
	dh0 = np.zeros((N,H))
	dWx = np.zeros((D,H))
	dWh = np.zeros((H,H))
	db = np.zeros((H,))
	dprev_h = np.zeros((N,H))

	for t in reversed(xrange(T)):
	dx[:,t,:], dprev_h, tdWx, tdWh, tdb = rnn_step_backward(dh[:,t,:] + dprev_h ,cache[t])
	dWx += tdWx
	dWh += tdWh
	db += tdb
	dh0 = dprev_h
	##############################################################################
	#                               END OF YOUR CODE                             #
	##############################################################################
	return dx, dh0, dWx, dWh, db