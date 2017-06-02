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

#################################################################
simrat = SimRat(RNN)
BLOCKS = 50 # The maximum length of blocks
REPEAT = 10 # Number of repetition times
p2a_drops = np.zeros((BLOCKS))
a2p_drops = np.zeros((BLOCKS))
meanPerformances = np.zeros((BLOCKS))
meanPerformanceExceptSwitchAnti = np.zeros((BLOCKS))
meanPerformanceExceptSwitchPro = np.zeros((BLOCKS))
meanPerformanceSwitchAnti = np.zeros((BLOCKS))
meanPerformanceSwitchPro = np.zeros((BLOCKS))
for block_length in xrange(1,BLOCKS+1):
    #print block_length
    p2a_drop = np.zeros((REPEAT))
    a2p_drop = np.zeros((REPEAT))
    meanPerformance = np.zeros((REPEAT))
    PerformanceExceptSwitchAnti = np.zeros((REPEAT))
    PerformanceExceptSwitchPro = np.zeros((REPEAT))
    PerformanceSwitchAnti = np.zeros((REPEAT))
    PerformanceSwitchPro = np.zeros((REPEAT))
    for i in xrange(REPEAT):
        X, y = generateTrials(block_length, 30)
        choice, probs = simrat.predict(X,y)
        p2a_drop[i] = simrat.p2a_prob[3] - simrat.meanPerformanceExceptSwitchAnti
        a2p_drop[i] = simrat.a2p_prob[3] - simrat.meanPerformanceExceptSwitchPro
        meanPerformance[i] = simrat.hit_rate.mean()
        PerformanceExceptSwitchAnti[i] = simrat.meanPerformanceExceptSwitchAnti
        PerformanceExceptSwitchPro[i] = simrat.meanPerformanceExceptSwitchPro
        PerformanceSwitchAnti = simrat.p2a_prob[3]
        PerformanceSwitchPro = simrat.a2p_prob[3]
        
    p2a_drops[block_length-1] = p2a_drop.mean()
    a2p_drops[block_length-1] = a2p_drop.mean()
    meanPerformances[block_length-1] = meanPerformance.mean()
    meanPerformanceExceptSwitchAnti[block_length-1] = PerformanceExceptSwitchAnti.mean()
    meanPerformanceExceptSwitchPro[block_length-1] = PerformanceExceptSwitchPro.mean()
    meanPerformanceSwitchAnti[block_length-1] = PerformanceSwitchAnti.mean()
    meanPerformanceSwitchPro[block_length-1] = PerformanceSwitchPro.mean()
    
p2aplot, = plt.plot(range(1,BLOCKS+1), p2a_drops, color='r')
a2pplot, = plt.plot(range(1,BLOCKS+1), a2p_drops, color='b')
plt.legend([p2aplot, a2pplot],["pro to anti","anti to pro"])
plt.xlabel('Block length')
plt.ylabel('Switch cost')
plt.title('Switch cost asymmetry as a function of last block length')
plt.show()

meanplot, = plt.plot(range(1,BLOCKS+1), meanPerformances, 'ko-')
meanAnti, = plt.plot(range(1,BLOCKS+1), meanPerformanceExceptSwitchAnti, 'ro-')
meanPro, = plt.plot(range(1,BLOCKS+1), meanPerformanceExceptSwitchPro, 'bo-')
switchAnti, = plt.plot(range(1,BLOCKS+1), meanPerformanceSwitchAnti,'ro--')
switchPro, = plt.plot(range(1,BLOCKS+1), meanPerformanceSwitchPro, 'bo--')
plt.legend([meanplot, meanAnti, meanPro, switchAnti, switchPro],
           ["Mean block performance","Mean anti block performance except swtich","Mean pro block performance except swtich",
           "Mean pro to anti switch", "Mean anti to pro switch"], loc='center left', bbox_to_anchor=(1, 0.7))
plt.xlabel('Block length')
plt.ylabel('Mean Performance')
plt.title('Mean performances as a function of last block length')
plt.show()