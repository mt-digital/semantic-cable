import numpy as np
import pickle

from experiment import Experiment

# n_trials = 2
n_trials = 20

min_pct = 0
max_pct = 100
d_pct = 6.25

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

with open('n_trials=20;d_pct=6.25', 'wb') as f:
    pickle.dump(ex, f)
