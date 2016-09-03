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

def getSwitch(preData, trial_window = 3):
    switches = {}
    p2a_switch = {}
    a2p_switch = {}
    pro_rules = {}
    for ratname, rat in preData.iteritems():
        SWITCH = []
        P2A = []
        A2P = []

        pro_rule = rat['X'][0,0,0]
        T = rat['X'].shape[1]
        for i in xrange(T):
            if rat['X'][0,i,0] != pro_rule:
                SWITCH.append(i)
                pro_rule = rat['X'][0,i,0]
                if i > trial_window and i < T - trial_window:
                    if pro_rule == 0:
                        P2A.append(i)
                    else:
                        A2P.append(i)
        switches[ratname] = np.asarray(SWITCH)
        p2a_switch[ratname] = np.asarray(P2A)
        a2p_switch[ratname] = np.asarray(A2P)

        pro_rules[ratname] = rat['X'][0,:,0]
    return switches, p2a_switch, a2p_switch, pro_rules
        

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