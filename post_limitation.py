import numpy as np

for eval_id in [0, 5, 6, 8]:
    res = np.loadtxt('res/study_error_%d.csv' % eval_id, delimiter=',')
    print(sum(res != 0))
