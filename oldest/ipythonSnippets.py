"""
This file is just a memo to keep some codes that is not used now but might be useful sometime later
"""
######################################################################
import helpers.DBUtilsClass as db
import cPickle as pkl

dbc = db.Connection()
D = {}
D['ratname'] = 'Rtnl'
D['train_size'] = train_num
D['test_size'] = val_num - train_num

D['loss'] = 217.765003
D['learning_rate'] = 8e-5
D['hidden_dim'] = 2
D['accuracy'] = 1.0
D['comments'] = "The well performing RNN with the smallest size ever."
D['b'] = pkl.dumps(RNN.params['b'])
D['b_vocab'] = pkl.dumps(RNN.params['b_vocab'])
D['W_vocab'] = pkl.dumps(RNN.params['W_vocab'])
D['h0'] = pkl.dumps(RNN.params['h0'])
D['Wh'] = pkl.dumps(RNN.params['Wh'])
D['Wx'] = pkl.dumps(RNN.params['Wx'])

dbc.saveToDB('vrat.rnn',D)

######################################################################

lrs = np.random.uniform(1,100,20) * 1e-5
bestAcc = 0
bestRNN = None
bestLR = 0
for lr in lrs:
    RNN = FirstRNN(hidden_dim = 2)
    solver = RNNsolver(RNN, trainX, trainTrueY,optim_config={
                 'learning_rate': lr,
               }, verbose = False)
    solver.train()
    out = RNN.predict(valX)
    acc = np.mean(out == valTrueY)
    print "learning rate: %s, accuracy = %f" % (lr,acc)
    if acc > bestAcc:
        bestAcc = acc
        bestRNN = RNN
        bestLR = lr
        print "Best RNN so far: learning rate: %s, accuracy = %f" % (lr,acc)