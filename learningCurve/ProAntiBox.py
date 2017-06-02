import numpy as np
class ProAntiBox(object):
    """
    This class is an environment for reinforcement learning virtual rat, like a bpod.
    It provides inputs to virtual rat and record the rat's response.
    If the accuracy is lower by a threshold, the ProAntiBox will repeat the same block until the accuracy goes up.
    Thus each block is newly generated intead of all generated once in advance.
    """
    def __init__(self, mode = "alternative", block_size = 20, random_range = 5, trial_per_episode = 3, reward_ratio = 1):
        """
        Input:
        - X: (N, T, D)
        - y: (N, T)
        - length: length of the sequence of trial generated each time.
        - block_size: block length
        """
        self.block_size = block_size
        self.random_range = random_range
        self.reward_ratio = reward_ratio
        self.t = 0 # Normal step count, reset when reaches self.length
        self.trial_in_episode = 0 # Special step count, reset when done, i.e. when paprameter update occurs

        self.episode = trial_per_episode # How many trials per parameter update occurs
        self.pro_rule = True # Start from Pro rule.
        self.history = np.zeros((0,5))

        self.set_mode(mode)

    def set_mode(self,mode):
        """
        Trial allocator.
        """
        self.mode = mode
        length = self.block_size + np.random.randint(-self.random_range,high=self.random_range+1)
        if mode == "pro_only":
            self.X , self.y = self._make_block(length, True)
        elif mode == "anti_only":
            self.X , self.y = self._make_block(length, False)            
        else:
            self.X, self.y = self._alternate(length, self.pro_rule)

    def reset(self):
        """
        Start of a new epoch. Required by the solver.
        """
        return self.X[:,self.t,:]

    def output_history(self):
        history = np.copy(self.history)
        self.history = np.zeros((0,5))
        return history

    def _make_block(self, length, pro_rule):
        X = np.zeros((1,length,3),dtype = np.int)
        y = np.zeros((1,length),dtype = np.int)
        #X[0,0,2] = 1
        if pro_rule:
            X[0,:,0] = 1
        X[0,:,1] = np.random.randint(2, size=self.length)
        y[0,:] = np.logical_not(np.bitwise_xor(X[0,:,0],X[0,:,1]))
        return X,y

    def render(self):
        return

    def step(self, action):
        record = np.zeros((1,5)) # (pro_rule, left/right, trial=1, action, reward)
        y = self.y[0,self.t]
    	if self.t>0 and self.X[0,self.t,0]!=self.X[0,self.t-1,0]:
    		# Give switch trials a stronger reward or punishment.
    		reward = self.reward_ratio if action == y else -1
    	else:
        	reward = 1 if action == y else -1

        record[0,:3] = self.X[0,self.t,:]
        record[0,-2] = action
        record[0,-1] = reward

        self.t+=1
        self.trial_in_episode+=1
        done = True if self.tt>=self.episode-1 else False
        if done:
            N, _, D = self.X.shape
            x = np.zeros((N,D))
            self.trial_in_episode=0
        else:
            x = self.X[:,self.t,:]
        if self.t>=self.length-1:
            self.set_mode(self.mode)
            self.t=0
        self.history = np.append(self.history,record,axis=0)
        return x, reward, done, y # To fill the spot of info

    def _low_accuracy(self):
        return np.mean(self.history[-(self.t-self.last_block):,3]) < 0.4
