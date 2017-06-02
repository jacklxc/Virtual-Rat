import numpy as np
import optim
class RNNsolver(object):
    """
    An RNNsolver encapsulates all the logic necessary for training
    image captioning models. The CaptioningSolver performs stochastic gradient
    descent using different update rules defined in optim.py.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    To train a model, you will first construct an RNNsolver instance,
    passing the model, dataset, and various options (learning rate, etc) 
    to the constructor. You will then call the train() method to run the 
    optimization procedure and train the model.

    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_acc_history and solver.val_acc_history will be lists containing
    the accuracies of the model on the training and validation set at each epoch.

    Example usage might look something like this:

    model = MyAwesomeModel(hidden_dim=100)
    solver = CaptioningSolver(model, data,
                  update_rule='sgd',
                  optim_config={
                    'learning_rate': 1e-3,
                  },
                  lr_decay=0.95,
                  num_epochs=10, batch_size=100,
                  print_every=100)
    solver.train()


    An RNNsolver works on a model object that must conform to the following
    API:

    - model.params must be a dictionary mapping string parameter names to numpy
    arrays containing parameter values.

    - model.loss(features, captions) must be a function that computes
    training-time loss and gradients, with the following inputs and outputs:

    Inputs:
    - x: Input data of shape (N, T, D)
    - y: Ground truth output of shape (N, T, O)

    Returns a tuple of:
    - loss: Scalar loss
    - grads: Dictionary of gradients parallel to self.params
    """
    def __init__(self, model,X, y, **kwargs):
        self.model = model
        
        # Unpack keyword arguments
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        batch_length = kwargs.pop('batch_length', 1) # Here batch_size means the length trials

        if batch_length > X.shape[1]:
            batch_length = X.shape[1]
        self.batch_length = batch_length

        self.X = self._packBatches(X)
        self.y = self._packBatches(y)

        self.T = self.X.shape[1]
        self.num_batches = self.X.shape[0]

        self.num_epochs = kwargs.pop('num_epochs', 100)

        self.verbose = kwargs.pop('verbose', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)
        self._reset()

    def _packBatches(self, data):
        N, T, D = data.shape
        batch_num = np.max([int (T / self.batch_length),1])
        iterator = np.zeros((batch_num, N, self.batch_length, D), dtype=np.int)
        for i in xrange(int(batch_num)):
            iterator[i,:,:,:] = data[:,i*self.batch_length: (i+1)*self.batch_length,:]
        return iterator

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.average_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.iteritems()}
            self.optim_configs[p] = d

    def _step(self, X, y):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """

        # Compute loss and gradient
        loss, grads = self.model.loss(X, y)
        average_loss = loss / y.shape[1]
        self.loss_history.append(loss)
        self.average_loss_history.append(average_loss)
        # Perform a parameter update
        for p, w in self.model.params.iteritems():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.T
        for i in xrange(self.num_epochs):
            for j in xrange(self.num_batches):
                t = i * self.num_epochs + j
                self._step(self.X[j,:,:,:],self.y[j,:,:,:])

                # Maybe print training loss
                if self.verbose and j==0:
                    print '(Epoch %d / %d) loss: %f, average loss: %f' % (
                        i+1, self.num_epochs, self.loss_history[-1], 
                        self.average_loss_history[-1])

                # At the end of every epoch, increment the epoch counter and decay the
                # learning rate.
                if j==self.num_batches-1:
                    self.epoch += 1
                    for k in self.optim_configs:
                        self.optim_configs[k]['learning_rate'] *= self.lr_decay