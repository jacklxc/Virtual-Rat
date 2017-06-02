import numpy as np
class simpleBox(object):
    """
    This class is an environment for reinforcement learning virtual rat, like a bpod.
    It provides inputs to virtual rat and record the rat's response.
    """
    def __init__(self, mode = "alternative", X=None, y=None, length = 10000,
     block_size = 20, random_range = 5, trial_per_episode = 3,
    repeat = False, reward_ratio = 1, p2a=0.5, correction = True):
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
        self.correction = correction
        self.t = 0 # Normal step count, reset when reaches self.length
        self.tt = 0 # Special step count, reset when done, i.e. when paprameter update occurs
        self.last_block = 0
        self.episode = trial_per_episode # How many trials per parameter update occurs

        self.history = np.zeros((0,5))

        self.change_mode(mode, p2a=p2a)

    def change_mode(self,mode,p2a=0.5):
        self.mode = mode
        self.t = 0
        if mode == "real":
            self.X = self.realX[:,:self.length,:]
            self.y = self.realy[:,:self.length,:]
        elif mode == "pro_only":
            self.X , self.y = self._pro_only()
        elif mode == "anti_only":
            self.X , self.y = self._anti_only()
        elif mode == "switch_ratio":
            self.X, self.y = self._switch_ratio(p2a)            
        else:
            self.X, self.y = self._alternate()

    def stop_repeat(self):
        self.repeat = False

    def reset(self):
        return self.X[:,self.t,:]

    def output_history(self):
        history = np.copy(self.history)
        self.history = np.zeros((0,5))
        return history

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

    def _switch_ratio(self, p2a):
        """
        Concatenate Pro+Anti and Anti+Pro block based on binomial trial with probability "p2a"
        """
        X = np.zeros((1,self.length,3),dtype = np.int)
        y = np.zeros((1,self.length),dtype = np.int)
        pro_rule = 0 if np.random.binomial(1,p2a,1)[0] else 1
        prev_t = 0
        t = self.block_size + np.random.randint(-self.random_range,high=self.random_range+1)
        while t<self.length:
            mode = np.random.binomial(1,p2a,1)[0] #a2p or p2a
            X[0,prev_t,2] = 1
            pro_rule = mode
            X[0,prev_t:t,0] = pro_rule
            pro_rule = np.logical_not(pro_rule)
            prev_t = t
            t += self.block_size + np.random.randint(-self.random_range,high=self.random_range+1)
            if t<self.length:
                X[0,prev_t:t,0] = pro_rule
                prev_t = t
                t += self.block_size + np.random.randint(-self.random_range,high=self.random_range+1)
            else:
                break # Actually this is not necessary but for bug free, I add break here.
        X[0,prev_t:,0] = pro_rule
        X[0,:,1] = np.random.randint(2, size=self.length)
        y[0,:] = np.logical_not(np.bitwise_xor(X[0,:,0],X[0,:,1]))
        return X, y


    def render(self):
        return

    def step(self, action):
        record = np.zeros((1,5)) # (pro_rule, left/right, trial=1, action, reward)
        X = self.X[0,self.t,:]
        y = self.y[0,self.t]
        if self.t>0 and self.X[0,self.t,0]!=self.X[0,self.t-1,0]:
            # Switch trial.
            # Give switch trials a stronger reward or punishment.
            reward = self.reward_ratio if action == y else -1
            if self.correction:
                if self._low_accuracy():
                    self.t = self.last_block
                else:
                    self.last_block = self.t
        else:
            reward = 1 if action == y else -1

        record[0,:3] = X
        record[0,-2] = action
        record[0,-1] = reward

        self.t+=1
        self.tt+=1
        done = True if self.tt>=self.episode else False
        if done:
            self.tt=0
        x = self.X[:,self.t,:]
        if self.t>=self.length-1:
            if not self.repeat:
                self.change_mode(self.mode)
            self.t=0
        self.history = np.append(self.history,record,axis=0)
        return x, reward, done, y # To fill the spot of info

    def _low_accuracy(self):
        return np.mean(self.history[-10:,-1]) < 0.4
