import numpy as np
class simpleBox(object):
    """
    This class is an environment for reinforcement learning virtual rat, like a bpod.
    It provides inputs to virtual rat and record the rat's response.
    """
    def __init__(self, mode = "alternative", X=None, y=None, length = 100,
     block_size = 20, random_range = 5, trial_per_episode = 3, repeat = False, reward_ratio = 1):
        """
        Input:
        - X: (N, T, D)
        - y: (N, T)
        - length: length of the sequence of trial generated each time.
        - block_size: block length
        - repeat: if true, then always use the same X sequence without changing.
        """
        self.realX = X
        self.realy = y
        self.length = length
        self.block_size = block_size
        self.random_range = random_range
        self.repeat = repeat
        self.reward_ratio = reward_ratio
        self.t = 0 # Normal step count, reset when reaches self.length
        self.tt = 0 # Special step count, reset when done, i.e. when paprameter update occurs
        self.episode = trial_per_episode # How many trials per parameter update occurs
        self.change_mode(mode)

    def change_mode(self,mode):
        self.mode = mode
        if mode == "real":
            self.X = self.realX[:,:self.length,:]
            self.y = self.realy[:,:self.length,:]
        elif mode == "pro_only":
            self.X , self.y = self._pro_only()
        elif mode == "anti_only":
            self.X , self.y = self._anti_only()            
        else:
            self.X, self.y = self._alternate()

    def stop_repeat(self):
        self.repeat = False

    def reset(self):
        return self.X[:,self.t,:]

    def _pro_only(self):
        X = np.zeros((1,self.length,3),dtype = np.int)
        y = np.zeros((1,self.length),dtype = np.int)
        X[0,0,2] = 1
        X[0,:,0] = 1
        X[0,:,1] = np.random.randint(2, size=self.length)
        y[0,:] = np.logical_not(np.bitwise_xor(X[0,:,0],X[0,:,1]))
        return X,y

    def _anti_only(self):
        X = np.zeros((1,self.length,3),dtype = np.int)
        y = np.zeros((1,self.length),dtype = np.int)
        X[0,0,2] = 1
        X[0,:,1] = np.random.randint(2, size=self.length)
        y[0,:] = np.logical_not(np.bitwise_xor(X[0,:,0],X[0,:,1]))
        return X,y

    def _alternate(self):
        X = np.zeros((1,self.length,3),dtype = np.int)
        y = np.zeros((1,self.length),dtype = np.int)
        X[0,0,2] = 1
        pro_rule = 1
        prev_t = 0
        t = self.block_size + np.random.randint(-self.random_range,high=self.random_range+1)
        while t<self.length:
            X[0,prev_t:t,0] = pro_rule
            pro_rule = np.logical_not(pro_rule)
            prev_t = t
            t += self.block_size + np.random.randint(-self.random_range,high=self.random_range+1)
        X[0,prev_t:,0] = pro_rule

        X[0,:,1] = np.random.randint(2, size=self.length)
        y[0,:] = np.logical_not(np.bitwise_xor(X[0,:,0],X[0,:,1]))
        return X,y

    def render(self):
        return

    def step(self, action):
    	if self.t>0 and self.X[0,self.t,0]!=self.X[0,self.t-1,0]:
    		# Give switch trials a stronger reward or punishment.
    		reward = self.reward_ratio if action == self.y[0,self.t] else -1
    	else:
        	reward = 1 if action == self.y[0,self.t] else -1

        y = self.y[0,self.t]
        self.t+=1
        self.tt+=1
        done = True if self.tt>=self.episode-1 else False
        if done:
            N, _, D = self.X.shape
            x = np.zeros((N,D))
            self.tt=0
        else:
            x = self.X[:,self.t,:]
        if self.t>=self.length-1:
            if not self.repeat:
                self.change_mode(self.mode)
            self.t=0
        return x, reward, done, y # To fill the spot of info


