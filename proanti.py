from helpers import DBUtilsClass as db
import numpy as np

CONN = db.Connection()

CONN.use('pa')
out = CONN.query('show tables')

if False:
	for row in out:
		exp = CONN.query("explain {}".format(row[0]))
		exp = zip(*exp)
		print exp[0]

out = zip(*CONN.query('explain alldata'))
print "{},{}\n".format(out[0], out[1])

all_rats = CONN.query('select distinct(ratname) from pa.alldata')

for rat in all_rats:
	sqlstr=('select pro_rule, target_on_right, trial_n=1, '
		 	'(cpv=0 AND WR=0) as `left`, (cpv=0 AND WR = 1) as `right`, cpv '
		 	'from pa.alldata where ratname=%s order by sessid, trial_n')

	out = CONN.query(sqlstr, (str(rat[0]),))
	data = np.array(out)
	train_size = 0.8 * data.shape[0]
#	W = trainRNN(data[:train_size,:])
#	loss = testRNN(W, data[train_size:])
#	simdata = gendata(W, data[:,:3])
	print rat[0], data.shape