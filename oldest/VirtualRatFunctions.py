from helpers import DBUtilsClass as db
import numpy as np
import cPickle as pkl

def getData(num_rats=15):
    """
    Get data from MySQL database

    Returns:
    -allRatsData: a dictionary with rat names as keys and numpy boolean array of 
    shape N * T * D as elements. In dimention 2, the bits represent pro_rule,
    target_on_right, trial_n=1, left, right, cpv (central poke violation) respectively.
    """
    CONN = db.Connection()
    CONN.use('pa')
    out = zip(*CONN.query('explain alldata'))
    all_rats = CONN.query('select distinct(ratname) from pa.alldata')

    MIN_PERF = 70;

    allRatsData = {}

    if num_rats > len(all_rats):
        num_rats = 15

    for rat in all_rats[:num_rats]:
        sqlstr=('select pro_rule, target_on_right, trial_n=1, '
                '(cpv=0 AND WR=0) as `left`, (cpv=0 AND WR = 1) as `right`, cpv '
                'from pa.alldata a, pa.pasess b where a.sessid=b.sessid '
                'and pro_perf>{} and anti_perf>{} and b.ratname=%s order by a.sessid, trial_n').format(MIN_PERF, MIN_PERF)

        out = CONN.query(sqlstr, (str(rat[0]),))
        data = np.array(out)
        allRatsData[str(rat[0])] = data
        print rat[0], data.shape
    return allRatsData

def uploadRNN(solver, ratname, comments, test_size, lr, hidden_dim, acc):
    dbc = db.Connection()
    D = {}

    model = solver.model

    D['ratname'] = ratname
    D['train_size'] = solver.X.shape[1]
    D['test_size'] = test_size
    D['loss'] = float(solver.loss_history[-1])
    D['hidden_dim'] = hidden_dim
    D['accuracy'] = float(acc)
    D['learning_rate'] = float(lr)
    D['comments'] = comments
    D['b'] = pkl.dumps(model.params['b'])
    D['b_vocab'] = pkl.dumps(model.params['b_vocab'])
    D['W_vocab'] = pkl.dumps(model.params['W_vocab'])
    D['h0'] = pkl.dumps(model.params['h0'])
    D['Wh'] = pkl.dumps(model.params['Wh'])
    D['Wx'] = pkl.dumps(model.params['Wx'])

    print dbc.saveToDB('vrat.rnn',D)

def preProcess(allRatsData, ratnames=[]):
    """
    Divide each rat's data into train, validation and test data for RNN

    Inputs:
    - allRatsData: a dictionary with rat names as keys and numpy boolean array of 
    shape N * T * D as elements. In dimention 2, the bits represent pro_rule,
    target_on_right, trial_n=1, left, right, cpv (central poke violation) respectively.

    Returns:
    - data: a dictionary with rat names as keys and dictionaries as elements.
        Each sub-level dictionary contains the following:
        - 'trainX': Stimulus to the rat for training.
        - 'valX': Stimulus to the rat for validation.
        - 'testX': Stimulus to the rat for test.
        - 'trainY': Reactions of the rat for training.
        - 'valY': Reactions of the rat for validation.
        - 'testY': Reactions of the rat for test.
        - 'trainTrueY': Rational reactions (logically correct reactions) for training.
        - 'valTrueY': Rational reactions (logically correct reactions) for validation.
        - 'testTrueY': Rational reactions (logically correct reactions) for test.
    """
    data = {}

    train_percentile = 0.8
    val_percentile = 0.9

    if len(ratnames) > 0:
        allratnames = ratnames
    else:
        allratnames = allRatsData.keys()

    for ratname in allratnames:
        data[ratname] = {}
        rat_data = data[ratname]
        rat = allRatsData[ratname]
        x = np.zeros((1, rat.shape[0], 3))
        y = np.zeros((1, rat.shape[0]))
        trueY = np.zeros((1, rat.shape[0]))

        x[0,:,:] = rat[:,:3]

        train_num = int(rat.shape[0] * train_percentile)
        val_num = int(rat.shape[0] * val_percentile)

        # Stimulus to rats
        rat_data['X'] = x
        rat_data['trainX'] = x[:,:train_num,:]
        rat_data['valX'] = x[:,train_num:val_num,:]
        rat_data['testX'] = x[:,val_num:,:]

        # Reaction of rats
        y[0,rat[:,3]>0] = 0
        y[0,rat[:,4]>0] = 1
        y[0,rat[:,5]>0] = 2

        rat_data['y'] = y
        rat_data['trainY'] = y[:,:train_num]
        rat_data['valY'] = y[:,train_num:val_num]
        rat_data['testY'] = y[:,val_num:]

        # Rational reaction (logically correct)
        trueY[0,:] = np.logical_not(np.bitwise_xor(rat[:,0],rat[:,1]))
        rat_data['trueY'] = trueY
        rat_data['trainTrueY'] = trueY[:,:train_num]
        rat_data['valTrueY'] = trueY[:,train_num:val_num]
        rat_data['testTrueY'] = trueY[:,val_num:]

    return data

def postProcess(choices, probabilities, allRatsData, trial_window = 3):
    """
    Process data generated from RNN. 

    Inputs:
    -probabilities: Data output from RNN. Array of shape (N, T, DD) giving the probability of each choice,
          where each element is a float. 
    -allRatsData: a dictionary with rat names as keys and numpy boolean array of 
        shape N * T * D as elements. In dimention 2, the bits represent pro_rule,
        target_on_right, trial_n=1, left, right, cpv (central poke violation) respectively.

    Outputs:
    -postRNNdata: a dictionary with rat names as keys and dictionaries as elements.
        Each sub-level dictionary contains the following:
        -'normalized_probs': numpy float array of shape (N * T * 2). Probablities that discarded cpv and normalized only for left and right.
        -'hit_rate': numpy float array of shape (T,). Correct probability for each trial (discard cpv).
        - 'switches': numpy int array that contains the trial number that switches pro_rule.
        - 'p2a_switch': numpy int array that contains the trial number that switches pro_rule form "pro" to "anti".
        - 'a2p_switch': numpy int array that contains the trial number that switches pro_rule form "anti" to "pro". 
        - 'pro_rules': a numpy bool array of shape (T,). Contains the pro_rule for each trial.
        - 'pro_prob': a numpy float array that only contains the probability of correct orientation under "pro" rule.
            At those indices (trials) that is under "anti" rules, the element is None.
        - 'anti_prob': a numpy float array that only contains the probability of correct orientation under "anti" rule.
            At those indices (trials) that is under "pro" rules, the element is None.
        - 'a2p_prob': a numpy float array of shape (2 * trial_window + 1,). Mean probabilities of correct orientation before 
            and after each pro_rule switch form anti to pro.
        - 'p2a_prob': a numpy float array of shape (2 * trial_window + 1,). Mean probabilities of correct orientation before 
            and after each pro_rule switch form pro to anti.
            (index i --> trial from switch = -trial_window + i)
    """
    postRNNdata = {}
    for ratname in probabilities.keys():
        postRNNdata[ratname] = {}
        ratProbs = postRNNdata[ratname]

        probs = probabilities[ratname]
        rat_data = allRatsData[ratname]

        normalized_probs = np.zeros((probs.shape[0],probs.shape[1],probs.shape[2]-1))
        normalized_probs[:,:,0] = probs[:,:,0]/(probs[:,:,0] + probs[:,:,1])
        normalized_probs[:,:,1] = probs[:,:,1]/(probs[:,:,0] + probs[:,:,1])
        ratProbs['normalized_probs'] = normalized_probs


        hit_rate = np.zeros(probs.shape[1])
        right = rat_data['valTrueY'][0,:] == 1
        left = rat_data['valTrueY'][0,:] == 0
        hit_rate[left] = normalized_probs[0,left,0]
        hit_rate[right] = normalized_probs[0,right,1]
        ratProbs['hit_rate'] = hit_rate

        rat_choice = choices[ratname]
        cpv = (rat_choice[0,:] == 2)
        hit_trials = (rat_choice[0,:] == rat_data['valTrueY'][0,:])
        hit = np.zeros(rat_choice.shape[1])
        hit[hit_trials] = 1
        hit[cpv] = np.nan
        ratProbs['hit'] = hit
        accuracy_exclude_cpv = np.nanmean(hit)
        ratProbs['accuracy_exclude_cpv'] = accuracy_exclude_cpv

        T = rat_data['valX'].shape[1]

        switches = []
        p2a_switch = []
        a2p_switch = []
        pro_rule = rat_data['valX'][0,0,0]
        for i in xrange(T):
            if rat_data['valX'][0,i,0] != pro_rule:
                switches.append(i)
                pro_rule = rat_data['valX'][0,i,0]
                if i > trial_window and i < T - trial_window:
                    if pro_rule == 0:
                        p2a_switch.append(i)
                    else:
                        a2p_switch.append(i)
        ratProbs['switches'] = np.asarray(switches)
        ratProbs['p2a_switch'] = np.asarray(p2a_switch)
        ratProbs['a2p_switch'] = np.asarray(a2p_switch)

        pro_rules = rat_data['valX'][0,:,0]
        ratProbs['pro_rules'] = pro_rules

        pro_prob = np.copy(hit_rate)
        anti_prob = np.copy(hit_rate)
        pro_rules = pro_rules.astype(bool)
        pro_prob[np.logical_not(pro_rules)] = None
        anti_prob[pro_rules] = None
        ratProbs['pro_prob'] = pro_prob
        ratProbs['anti_prob'] = anti_prob

        p2a_prob = calculatePerformance(p2a_switch, hit_rate, trial_window)
        a2p_prob = calculatePerformance(a2p_switch, hit_rate, trial_window)
        p2a_prob2 = calculatePerformance(p2a_switch, hit, trial_window)
        a2p_prob2 = calculatePerformance(a2p_switch, hit, trial_window)

        ratProbs['p2a_prob'] = p2a_prob
        ratProbs['a2p_prob'] = a2p_prob
        ratProbs['p2a_prob2'] = p2a_prob2
        ratProbs['a2p_prob2'] = a2p_prob2

    return postRNNdata

def calculatePerformance(switch_index, hit_rate, trial_window):
    # index i --> trial from switch = -trial_window + i
    switch_matrix = np.zeros((len(switch_index), trial_window * 2 + 1))
    for i in xrange(len(switch_index)):
        switch_matrix[i,:] = hit_rate[(switch_index[i] - trial_window): (switch_index[i] + trial_window + 1)]
    switch_prob = np.nanmean(switch_matrix,axis=0)
    # index i --> trial from switch = -trial_window + i

    return switch_prob

def meanPerformance(postRNNdata):
    size = postRNNdata[postRNNdata.keys()[0]]['p2a_prob'].shape[0]
    p2a_mean = np.zeros(size)
    a2p_mean = np.zeros(size)
    p2a_mean2 = np.zeros(size)
    a2p_mean2 = np.zeros(size)
    for ratname in postRNNdata.keys():
        p2a_mean += postRNNdata[ratname]['p2a_prob']
        a2p_mean += postRNNdata[ratname]['a2p_prob']
        p2a_mean2 += postRNNdata[ratname]['p2a_prob2']
        a2p_mean2 += postRNNdata[ratname]['a2p_prob2']
    p2a_mean /= len(postRNNdata)
    a2p_mean /= len(postRNNdata)
    p2a_mean2 /= len(postRNNdata)
    a2p_mean2 /= len(postRNNdata)
    return p2a_mean, a2p_mean, p2a_mean2, a2p_mean2

def realRatMeanPerformance(preData, trial_window = 3):
    normalizedY = {}
    for ratname in preData.keys():
        rat = preData[ratname]
        normalizedY[ratname] = {}
        cpv = (rat['y'][0,:] == 2)
        hit_trials = (rat['y'][0,:] == rat['trueY'][0,:])
        hit = np.zeros(rat['y'].shape[1])
        hit[hit_trials] = 1
        hit[cpv] = np.nan
        normalized_accuracy = np.nanmean(hit)

        normalizedY[ratname]['hit'] = hit
        normalizedY[ratname]['normalized_accuracy'] = normalized_accuracy

        T = rat['X'].shape[1]

        p2a_switch = []
        a2p_switch = []
        pro_rule = rat['X'][0,0,0]
        for i in xrange(T):
            if rat['X'][0,i,0] != pro_rule:
                pro_rule = rat['X'][0,i,0]
                if i > trial_window and i < T - trial_window:
                    if pro_rule == 0:
                        p2a_switch.append(i)
                    else:
                        a2p_switch.append(i)

        p2a_matrix = np.zeros((len(p2a_switch), trial_window * 2 + 1))
        for i in xrange(len(p2a_switch)):
            p2a_matrix[i,:] = hit[(p2a_switch[i] - trial_window): (p2a_switch[i] + trial_window + 1)]
        p2a = np.nanmean(p2a_matrix,axis=0)
        normalizedY[ratname]['p2a'] = p2a

        a2p_matrix = np.zeros((len(a2p_switch), trial_window * 2 + 1))
        for i in xrange(len(a2p_switch)):
            a2p_matrix[i,:] = hit[(a2p_switch[i] - trial_window): (a2p_switch[i] + trial_window + 1)]
        a2p = np.nanmean(a2p_matrix,axis=0)
        normalizedY[ratname]['a2p'] = a2p
    
    # Compute the grand mean
    p2a_mean = np.zeros(trial_window*2+1)
    a2p_mean = np.zeros(trial_window*2+1)
    for ratname in normalizedY.keys():
        p2a_mean += normalizedY[ratname]['p2a']
        a2p_mean += normalizedY[ratname]['a2p']
    p2a_mean /= len(normalizedY)
    a2p_mean /= len(normalizedY)

    return p2a_mean, a2p_mean, normalizedY

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
    cache = (x,prev_h, Wx, Wh, next_h)
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
    x,prev_h, Wx, Wh, next_h = cache
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
    N, T, _ = x.shape
    _, H = h0.shape
    h = np.empty((N,T,H))
    prev_h = h0
    cache = []
    for t in xrange(T):
        h[:,t,:], Cache = rnn_step_forward(x[:,t,:],prev_h,Wx, Wh, b)
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
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
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
    making predictions over a vocabulary (choices) of size V (V = 3) for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss. This variable is not necessary.
    Might be removed later


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