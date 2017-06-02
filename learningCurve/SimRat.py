import numpy as np 
import minpy.numpy as mnp

class SimRat(object):
    def __init__(self, RNN, ratname = "SimRat", trial_window=3):

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
        self.trial_window = trial_window
        self.RNN = RNN
        self.history = np.zeros((0,5))
    def predict(self, X, y):
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
        self.X = X
        self.y = y
        self.session = np.where(self.X[0,:,2]==1)[0] # Stores trial number when new session starts
        self.pro_rules = self.X[0,:,0]
        self.probability = self.RNN.forward(X).asnumpy()
        # Post RNN data process
        # Process probability itself.
        self.normalized_probs = self._normalizeProbs(self.probability)
        self.hit_rate = self._hitRate(y, self.normalized_probs)

        # Extract rule and orientation from input.
        self.swtich, self.p2a, self.a2p, self.p2a_left, self.p2a_right, \
        self.a2p_left, self.a2p_right = self._switch(X)

        # Separate probability based on rule and orientation.
        self.pro_prob, self.anti_prob = self._proAnti(X, self.hit_rate)
        self.pro_left_prob, self.pro_right_prob, self.anti_left_prob, self.anti_right_prob \
            = self._proAntiLeftRight(X, self.hit_rate)
        self.p2a_prob = self._calculatePerformance(self.p2a, self.hit_rate)
        self.a2p_prob = self._calculatePerformance(self.a2p, self.hit_rate)
        self.p2a_left_prob = self._calculatePerformance(self.p2a_left, self.hit_rate)
        self.a2p_left_prob = self._calculatePerformance(self.a2p_left, self.hit_rate)
        self.p2a_right_prob = self._calculatePerformance(self.p2a_right, self.hit_rate)
        self.a2p_right_prob = self._calculatePerformance(self.a2p_right, self.hit_rate)
        return self.probability

    def addHistory(self, record):
        self.history = np.append(self.history,record,axis=0)

    def computeLearningCurve(self, session_length = 150):
        T, D = self.history.shape
        total_session = T/session_length
        self.pro_rate = np.zeros((total_session,))
        self.anti_rate = np.zeros((total_session))

        for i in range(total_session):
            history_piece = self.history[i*session_length:(i+1)*session_length,:]
            self.pro_rate[i], self.anti_rate[i] = self._sessionAccuracy(history_piece)

    def _sessionAccuracy(self,history):
        """
        history[0,:] =  (pro_rule, left/right, trial=1, action, reward), i.e. (X, action, reward)
        """
        switch = history[1:,0]!=history[:-1,0]
        switch = np.append([False],switch)

        block = np.logical_not(switch)
        pro = history[:,0]==1
        anti = history[:,0]==0
        pro_block = np.logical_and(pro, block)
        anti_block = np.logical_and(anti, block)

        correct = history[:,-1]>0
        pro_rate = np.mean(correct[pro_block])
        anti_rate = np.mean(correct[anti_block])

        return pro_rate, anti_rate

    def _switch(self, X):
        """
        Find out trial numbers when switch occured.
        """
        SWITCH = []
        P2A = []
        A2P = []
        P2A_left = []
        P2A_right = []
        A2P_left = []
        A2P_right = []

        pro_rule = X[0,0,0]
        T = X.shape[1]
        for i in xrange(T):
            if X[0,i,0] != pro_rule:
                SWITCH.append(i)
                pro_rule = X[0,i,0]
                if i > self.trial_window and i < T - self.trial_window:
                    if pro_rule == 0:
                        P2A.append(i)
                        if X[0,i,1] == 0:
                            P2A_left.append(i)
                        elif X[0,i,1] == 1:
                            P2A_right.append(i)
                    else:
                        A2P.append(i)
                        if X[0,i,1] == 0:
                            A2P_left.append(i)
                        elif X[0,i,1] == 1:
                            A2P_right.append(i)
        switches = np.asarray(SWITCH)
        p2a_switch = np.asarray(P2A)
        a2p_switch = np.asarray(A2P)
        p2a_left_switch = np.asarray(P2A_left)
        a2p_left_switch = np.asarray(A2P_left)
        p2a_right_switch = np.asarray(P2A_right)
        a2p_right_switch = np.asarray(A2P_right)

        return switches, p2a_switch, a2p_switch, p2a_left_switch, p2a_right_switch, \
            a2p_left_switch, a2p_right_switch

    def _normalizeProbs(self, probability):
        """
        Exclude the probability of cpv and normalize the probabilities of left and right.
        """
        normalized_probs = np.zeros((probability.shape[0],probability.shape[1],
            probability.shape[2]-1))
        normalized_probs[:,:,0] = probability[:,:,0]/(probability[:,:,0]
         + probability[:,:,1])
        normalized_probs[:,:,1] = probability[:,:,1]/(probability[:,:,0] 
            + probability[:,:,1])
        return normalized_probs

    def _hitRate(self,trueY, normalized_probs):
        """
        Combines the probabilities to get the probabilities of choosing correctly.
        """
        hit_rate = np.zeros(normalized_probs.shape[1])
        right = trueY[0,:] == 1
        left = trueY[0,:] == 0
        hit_rate[left] = normalized_probs[0,left,0]
        hit_rate[right] = normalized_probs[0,right,1]
        return hit_rate

    def _proAnti(self,X, hit_rate):
        """
        Seperate self.hit_rate to probabilites of correct choice during pro and anti.
        """
        pro_rules = X[0,:,0]
        pro_prob = np.copy(hit_rate)
        anti_prob = np.copy(hit_rate)
        pro_rules = pro_rules.astype(bool)
        pro_prob[np.logical_not(pro_rules)] = None
        anti_prob[pro_rules] = None
        return pro_prob, anti_prob

    def _proAntiLeftRight(self,X, hit_rate):
        """
        Seperate self.hit_rate to probabilites of correct choice during pro and anti.
        """
        pro_rules = X[0,:,0]
        left = X[0,:,1]==0
        right = X[0,:,1]==1

        pro_prob = np.copy(hit_rate)
        anti_prob = np.copy(hit_rate)
        pro_left_prob = np.copy(hit_rate)
        anti_left_prob = np.copy(hit_rate)
        pro_right_prob = np.copy(hit_rate)
        anti_right_prob = np.copy(hit_rate)

        pro = pro_rules.astype(bool)
        anti = np.logical_not(pro)
        pro_left_prob[np.logical_not(np.logical_and(pro,left))] = None
        pro_right_prob[np.logical_not(np.logical_and(pro,right))] = None
        anti_left_prob[np.logical_not(np.logical_and(anti,left))] = None
        anti_right_prob[np.logical_not(np.logical_and(anti,right))] = None
        return pro_left_prob, pro_right_prob, anti_left_prob, anti_right_prob

    def _calculatePerformance(self, switch_index, hit_rate):
        """
        Calculate the switch cost for this particular rat.

        Inputs:
        - switch_index: indices of trial numbers that switch occurs.
        - hit_rate: correct probability of each trial.

        Returns:
        - switch_prob: numpy float array that contains mean correct rate around switches
        """     
        # index i --> trial from switch = -trial_window + i
        switch_matrix = np.zeros((len(switch_index), self.trial_window * 2 + 1))
        for i in xrange(len(switch_index)):
            switch_matrix[i,:] = hit_rate[(switch_index[i] - self.trial_window): 
            (switch_index[i] + self.trial_window + 1)]
        switch_prob = np.nanmean(switch_matrix,axis=0)

        return switch_prob