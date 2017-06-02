import numpy as np
class trialGenerator(object):
    def __init__(self, N):
        self.N = N

    def getSequence(self, length = 10000, mode = "test", p2a = 0.5, block_size = 30, std = 0):
        if mode == "alternative":
            X = self._alternate(length,block_size,p2a,std)
        elif mode == "a2p":
            X = self._alternate(length ,block_size,0,std)
        elif mode == "p2a":
            X = self._alternate(length ,block_size,1,std)
        elif mode == "test":
        	X = self._test(length ,block_size,std)
        elif mode == "pro_only":
            X = self._pro_only(length)
        elif mode == "anti_only":
            X = self._anti_only(length)

        y = self._getY(X)
        X_batch = self._makeBatches(X)
        y_batch = self._makeBatches(y)
        return X_batch,y_batch

    def _pro_only(self, length):
        X = np.zeros((1, length,3),dtype = np.int)
        y = np.zeros((1, length,3),dtype = np.int)
        X[0,0,2] = 1
        X[0,:,0] = 1
        X[0,:,1] = np.random.randint(2, size=length)
        true = np.logical_not(np.bitwise_xor(X[0,:,0],X[0,:,1]))
        y[0,true==0,0] = 1
        y[0,true==1,1] = 1
        y[0,true==2,2] = 1
        return X

    def _anti_only(self, length):
        X = np.zeros((1, length,3),dtype = np.int)
        X[0,0,2] = 1
        X[0,:,1] = np.random.randint(2, size=length)
        return X

    def _alternate(self, length, block_size, p2a, std):
        X = np.zeros((1,length,3),dtype = np.int)
        pro_rule = 0 if np.random.binomial(1,p2a,1)[0] else 1
        prev_t = 0
        t = block_size + np.rint(np.random.randn() * std).astype(int)
        while t<length:
            mode = np.random.binomial(1,p2a,1)[0] #a2p or p2a
            X[0,prev_t,2] = 1
            pro_rule = mode
            X[0,prev_t:t,0] = pro_rule
            pro_rule = np.logical_not(pro_rule)
            prev_t = t
            t += block_size + np.rint(np.random.randn() * std).astype(int)
            if t<length:
                X[0,prev_t:t,0] = pro_rule
                prev_t = t
                t += block_size + np.rint(np.random.randn() * std).astype(int)
            else:
                break # Actually this is not necessary but for bug free, I add break here.
        X[0,prev_t:,0] = pro_rule
        X[0,:,1] = np.random.randint(2, size=length)
        return X

    def _test(self, length, block_size, std):
        X = np.zeros((1,length,3),dtype = np.int)
        pro_rule = 1
        prev_t = 0
        t = block_size + np.round(np.random.randn() * std).astype(int)
        while t<length:
            X[0,prev_t:t,0] = pro_rule
            pro_rule = np.logical_not(pro_rule)
            prev_t = t
            t += block_size + np.round(np.random.randn() * std).astype(int)
        X[0,prev_t:,0] = pro_rule
        X[0,0,2] = 1
        X[0,:,1] = np.random.randint(2, size=length)
        return X

    def _getY(self,X):
        """
        Return a one hot encoded ground truth y.
        """
        y = np.zeros((1, X.shape[1],3),dtype = np.int)
        true = np.logical_not(np.bitwise_xor(X[0,:,0],X[0,:,1]))
        y[0,true==0,0] = 1
        y[0,true==1,1] = 1
        y[0,true==2,2] = 1
        return y

    def _compress_label(self, y):
        """
        Compress one hot encoded label to normal label. (0,1,2) style
        """
        translator = np.array([0,1,2],dtype=np.int)
        yy = np.sum(y*translator,axis=2)
        return yy


    def _makeBatches(self, data):
        """
        Divide data with N=1 to batches with N=self.N
        """
        _, T, D = data.shape
        batch_size = int(T/self.N)
        batches = np.zeros((self.N, batch_size, D),dtype=np.int)
        for i in xrange(self.N):
            batches[i,:,:] = data[0,i*batch_size:(i+1)*batch_size,:]
        return batches



