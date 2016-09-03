import numpy as np 

class Rat(object):
    def __init__(self,ratname,rat,trial_window=3):

        """
        Create attributes and call methods to get their values.

        Inputs: 
        - ratname: string, the name of rat
        - rat: numpy boolean array of shape N * T * D as elements. In dimention 2, the bits 
            represent pro_rule, target_on_right, trial_n=1, left, right, cpv (central poke 
            violation) respectively.
        - trial_window: int, number of trials computed before and after swtiches.
        """

        self.name = ratname
        self.X, self.y, self.trueY = self._basicData(rat)
        self.session = np.where(self.X[0,:,2]==1)[0] # Stores trial number when new session starts
        self.trial_window = trial_window
        self.pro_rules = self.X[0,:,0]
        self.real_switches, self.real_p2a_switch, self.real_a2p_switch = self._switch(self.X)
        self.hit, self.real_rat_accuracy = self._realRatHit()
        self.real_p2a = self._realSwtichCost(self.real_p2a_switch)
        self.real_a2p = self._realSwtichCost(self.real_a2p_switch)
        self.sessionPerformance = self._realRatSessionPerformance()
        self.train_session, self.val_session, self.test_session = self._divideSessions() # Indices of self.session
        self.trainX, self.trainY, self.trainTrueY = self._divideData(self.train_session)
        self.valX, self.valY, self.valTrueY = self._divideData(self.val_session)
        self.testX, self.testY, self.testTrueY = self._divideData(self.test_session)
        
        self.RNN = None
        self.choice = None
        self.probability = None
        self.final = None # Use val data to predict or test data to predict
        self.normalized_probs = None
        self.hit_rate = None
        self.hit = None
        self.accuracy_exclude_cpv = None
        self.p2a, self.a2p, self.swtich = None, None, None
        self.pro_prob, self.anti_prob = None, None
        self.p2a_prob, self.a2p_prob = None, None
        self.p2a_prob2, self.a2p_prob2 = None, None # real rat like way of processing

    def predict(self, RNN, final = False):
        """
        Make prediction based on the trianed RNN.

        Inputs:
        - RNN: the RNN object.
        - final: bool, True if training + validation data are trained and False if only training
            data is used to train.

        Returns:
        - self.choice: numpy array of shape (N, T) giving sampled y,
          where each element is an integer in the range [0, V). 
        - self.probility: numpy array of shape (N, T, DD) giving the probability of each choice,
          where each element is a float. 
        """
        self.RNN = RNN
        self.final = final
        if final:
            X = self.testX
            y = self.testY
            trueY = self.testTrueY
        else:
            X = self.valX
            y = self.valY
            trueY = self.valTrueY
        self.choice, self.probability = self.RNN.predict(X)
        # Post RNN data process
        self._normalizeProbs()
        self._hitRate(trueY)
        self._hitLikeReal(trueY)
        self.swtich, self.p2a, self.a2p = self._switch(X)
        self._proAnti(X)
        self.p2a_prob = self._calculatePerformance(self.p2a, self.hit_rate)
        self.a2p_prob = self._calculatePerformance(self.a2p, self.hit_rate)
        self.p2a_prob2 = self._calculatePerformance(self.p2a, self.hit)
        self.a2p_prob2 = self._calculatePerformance(self.a2p, self.hit)
        return self.choice, self.probability

    def _basicData(self,rat):
        """
        Format raw data to X, y and trueY
        """
        X = np.zeros((1, rat.shape[0], 3))
        y = np.zeros((1, rat.shape[0]))
        trueY = np.zeros((1, rat.shape[0]))

        X[0,:,:] = rat[:,:3]

        # Reaction of rats
        y[0,rat[:,3]>0] = 0
        y[0,rat[:,4]>0] = 1
        y[0,rat[:,5]>0] = 2

        # Rational reaction (logically correct)
        trueY[0,:] = np.logical_not(np.bitwise_xor(rat[:,0],rat[:,1]))
        return X, y, trueY

    def _switch(self, X):
        """
        Find out trial numbers when switch occured.
        """
        SWITCH = []
        P2A = []
        A2P = []

        pro_rule = X[0,0,0]
        T = X.shape[1]
        for i in xrange(T):
            if X[0,i,0] != pro_rule:
                SWITCH.append(i)
                pro_rule = X[0,i,0]
                if i > self.trial_window and i < T - self.trial_window:
                    if pro_rule == 0:
                        P2A.append(i)
                    else:
                        A2P.append(i)
        switches = np.asarray(SWITCH)
        p2a_switch = np.asarray(P2A)
        a2p_switch = np.asarray(A2P)
        return switches, p2a_switch, a2p_switch

    def _realRatHit(self):
        """
        Returns:
        hit: a numpy boolean array of rat's real performance (for each trial, hit or not) 
        real_rat_accuracy: a float of the correct rate of the real rat's performance  
        """
        cpv = (self.y[0,:] == 2)
        hit_trials = (self.y[0,:] == self.trueY[0,:])
        hit = np.zeros(self.y.shape[1])
        hit[hit_trials] = 1
        hit[cpv] = np.nan
        real_rat_accuracy = np.nanmean(hit)
        return hit, real_rat_accuracy

    def _realSwtichCost(self,switches):
        """
        Calculates the switch cost of the real rat's performance.
        Returns:
        - switch_cost: a numpy float array of shape (2 * self.trial_window + 1,). Mean percentage
         of correct orientation before and after each pro_rule switch form pro to anti.
            (index i --> trial from switch = -trial_window + i)
        """
        switch_matrix = np.zeros((len(switches), self.trial_window * 2 + 1))
        for i in xrange(len(switches)):
            switch_matrix[i,:] = self.hit[(switches[i] - self.trial_window):
             (switches[i] + self.trial_window + 1)]
        swtich_cost = np.nanmean(switch_matrix,axis=0)
        return swtich_cost

    def _realRatSessionPerformance(self):
        """
        Calculates the real rat's performance of each session.

        Returns:
        -sessionPerformance: numpy float array
        """
        sessionPerformance = []
        prev_i = self.session[0]
        for i in self.session[1:]:
            perf = np.nanmean(self.hit[prev_i:i])
            sessionPerformance.append(perf)
            prev_i = i
        perf = np.nanmean(self.hit[prev_i:])
        sessionPerformance.append(perf)
        sessionPerformance = np.asarray(sessionPerformance)
        return sessionPerformance

    def _divideData(self, session_indices):
        """
        Divide self.X, self.y, self.trueY into train, val and test data portiions.
        The smallest unit is session (day).

        Inputs:
        -session_indices

        Outputs:
        -x, y, trueY: the corresponding train, val or test set of x, y and trueY
        """
        x = np.zeros((1,0,3))
        y = np.zeros((1,0))
        trueY = np.zeros((1,0))
        for i in session_indices:
            session = self.session[i]
            if i<len(self.session)-1:
                next_session = self.session[i+1]
                x = np.concatenate((x,self.X[:,session:next_session,:]),axis=1)
                y = np.concatenate((y,self.y[:,session:next_session]),axis=1)
                trueY = np.concatenate((trueY,self.trueY[:,session:next_session]),axis=1)
            else:
                x = np.concatenate((x,self.X[:,session:,:]),axis=1)
                y = np.concatenate((y,self.y[:,session:]),axis=1)
                trueY = np.concatenate((trueY,self.trueY[:,session:]),axis=1)
        return x, y, trueY

    def _divideSessions(self):
        """
        Set the ratio of train, val and test data, then randomly permute indices of sessions
        to determine which session goes to which data set.
        """
        session_num = self.session.shape[0]
        indices = np.random.permutation(session_num)
        train_num = int(session_num * 0.8)
        val_num = int(session_num * 0.9)
        train_session = indices[:train_num]
        val_session = indices[train_num:val_num]
        test_session = indices[val_num:]
        return train_session, val_session, test_session

    def _normalizeProbs(self):
        """
        Exclude the probability of cpv and normalize the probabilities of left and right.
        """
        self.normalized_probs = np.zeros((self.probability.shape[0],self.probability.shape[1],
            self.probability.shape[2]-1))
        self.normalized_probs[:,:,0] = self.probability[:,:,0]/(self.probability[:,:,0]
         + self.probability[:,:,1])
        self.normalized_probs[:,:,1] = self.probability[:,:,1]/(self.probability[:,:,0] 
            + self.probability[:,:,1])

    def _hitRate(self,trueY):
        """
        Combines the probabilities to get the probabilities of choosing correctly.
        """
        self.hit_rate = np.zeros(self.probability.shape[1])
        right = trueY[0,:] == 1
        left = trueY[0,:] == 0
        self.hit_rate[left] = self.normalized_probs[0,left,0]
        self.hit_rate[right] = self.normalized_probs[0,right,1]
    
    def _hitLikeReal(self, trueY):
        """
        Record correct or not for each trial. If cpv occurs, leave it as NaN.
        """
        cpv = (self.choice[0,:] == 2)
        hit_trials = (self.choice[0,:] == trueY[0,:])
        self.hit = np.zeros(self.choice.shape[1])
        self.hit[hit_trials] = 1
        self.hit[cpv] = np.nan
        self.accuracy_exclude_cpv = np.nanmean(self.hit)

    def _proAnti(self,X):
        """
        Seperate self.hit_rate to probabilites of correct choice during pro and anti.
        """
        pro_rules = X[0,:,0]
        self.pro_prob = np.copy(self.hit_rate)
        self.anti_prob = np.copy(self.hit_rate)
        pro_rules = pro_rules.astype(bool)
        self.pro_prob[np.logical_not(pro_rules)] = None
        self.anti_prob[pro_rules] = None

    def _calculatePerformance(self, switch_index, hit_rate):
        """
        Calculate the switch cost for this particular rat.

        Inputs:
        - switch_index: indices of trial numbers that swtich occurs.
        - hit_rate: either p2a or a2p probabilities

        Returns:
        - switch_prob: numpy float array that contains mean correct rate around switches
        """     
        # index i --> trial from switch = -trial_window + i
        switch_matrix = np.zeros((len(switch_index), self.trial_window * 2 + 1))
        for i in xrange(len(switch_index)):
            switch_matrix[i,:] = hit_rate[(switch_index[i] - self.trial_window): 
            (switch_index[i] + self.trial_window + 1)]
        switch_prob = np.nanmean(switch_matrix,axis=0)
        # index i --> trial from switch = -trial_window + i

        return switch_prob

