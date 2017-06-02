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
        prob, baseline = self.RNN.forward(X)
        self.probability = prob.asnumpy()
        # Post RNN data process
        self._normalizeProbs()
        self._hitRate(y)
        self.swtich, self.p2a, self.a2p = self._switch(X)
        self._proAnti(X)
        self.p2a_prob = self._calculatePerformance(self.p2a, self.hit_rate)
        self.a2p_prob = self._calculatePerformance(self.a2p, self.hit_rate)
        return self.probability

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

