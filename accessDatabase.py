import helpers.DBUtilsClass as db
import cPickle as pkl

dbc = db.Connection()

# Get mothods
dir(dbc)

help(dbc.saveToDB)

D = {}
D['ratname'] = 'Z009'
D['train_size'] = 10000
D['test_size'] = 1000

D['W_init'] = pkl.dumps(range(100))

dbc.saveToDB('vrat.rnn',D)