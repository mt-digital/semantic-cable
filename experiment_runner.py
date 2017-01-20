import numpy as np
import pickle

from experiment import Experiment

# n_trials = 2
n_trials = 20

min_pct = 0
max_pct = 100
d_pct = 6.25
# d_pct = 50.0

fox_pcts = list(np.arange(min_pct, max_pct + .001, d_pct))

print(
    """
Running experiment with the following percents: {}
and {} trials per mixture
    """.format(fox_pcts, n_trials)
)

ex = Experiment(n_trials, fox_pcts)
# ex = Experiment(n_trials, [25, 50])

# ex.load_texts('Sample Week for Zipf experiment')
ex.load_texts('Three Months for Semantic Network Experiments')

ex.run()


# hack to deal with OS X pickle implementation bug; fails on files over 2G
# see https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
MAX_BYTES = 2**31 - 1

with open('n_trials=20;d_pct=6.25', 'wb') as f:

    pickle_bytearray = bytearray(pickle.dumps(ex))
    ba_len = len(pickle_bytearray)

    for idx in range(0, ba_len, MAX_BYTES):
        f.write(pickle_bytearray[idx:idx+MAX_BYTES])
