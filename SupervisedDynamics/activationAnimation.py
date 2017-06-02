import numpy as np
import cPickle as pkl
import matplotlib.pyplot as plt

def load_weights(filename):
    with open(filename,"rb") as f:
        weights = pkl.load(f)
    return weights

def phasePlaneAnimation(h, dim1=0, dim2=1, start = 0, end = 1000):
    """
    Input:
    - h: numpy array in shape (T,N,H)
    """
    _, N, H = h.shape
    history = np.zeros((H,0))
    for n in range(N):
        history = np.append(history,h[:,n,:].T,axis=1)
    length = end-start
    dim1 = 0
    dim2 = 1

    fig, ax = plt.subplots()

    pro = history[-2,:]==1
    anti = history[-2,:]==0
    left = history[-1,:]==0
    right = history[-1,:]==1

    pro_left = np.logical_and(pro,left)
    pro_right = np.logical_and(pro,right)
    anti_left = np.logical_and(anti,left)
    anti_right = np.logical_and(anti,right)

    for t in range(length):
        if t==0:
            points, = ax.plot(history[dim1,start:end][0],
                history[dim1,start:end][0],marker='o',ls="-")
            ax.set_xlim(-1.2,1.2)
            ax.set_ylim(-1.2,1.2)
        elif t<3 or t>length-3:
            points.set_data(history[dim1,start:end][t],history[dim2,start:end][t])
        else:
            points.set_data(history[dim1,start:end][t-2:t+2],history[dim2,start:end][t-2:t+2])
        plt.pause(0.25)

h = load_weights("activation.pkl")
phasePlaneAnimation(h)