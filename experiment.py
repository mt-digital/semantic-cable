import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from collections import Counter

from build_network import build_network
from util import get_corpus_text


class Experiment:

    def __init__(self, n_trials, pct_fox_vals=[0, 25, 50, 75, 100], tau=0.6):

        self.pct_fox_vals = pct_fox_vals
        self.tau = tau
        self.run_dict = None
        self.n_trials = n_trials

    def load_texts(self, corpus_name):

        self.fox_texts, _ = get_corpus_text(corpus_name, 'FOXNEWSW')
        self.msnbc_texts, _ = get_corpus_text(corpus_name, 'MSNBCW')

        pass

    def run(self, nproc=1):
        self.run_dict = {}
        for pct in self.pct_fox_vals:
            self.run_dict[pct] = experiment_run(
                self.msnbc_texts, self.fox_texts, pct, self.n_trials, self.tau
            )

    def plot_betas(self):
        '''
        sort based on pct_fox then plot beta
        '''
        pass

    def save(self, path):

        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)


class ExperimentRunResult:

    def __init__(self, pct_fox, fit_coefficients, embedding_matrices,
                 tau, log_freq, test_fit_coefficients=None):
        '''
        pct_fox (float): 0 <= pct_fox <= 100
        fit_coefficients (FitCoefficients): coeffs
        embedding_matrices (np.ndarray): n_words x 300 embedding matrices,
            one for each test_fit; if pct_fox = 0 or 100 then a list of length
            1
        tau (float): tolerance which similarity must exceed to count as edge
        log_freq (np.ndarray): log of degree frequencies
        fit_log_freq (np.ndarray):
        test_fit_coefficients (list(FitCoefficients)): list of
            coeffs used in building average beta, for example
        '''

        self.pct_fox = pct_fox
        self.fit_coefficients = fit_coefficients
        self.embedding_matrices = embedding_matrices
        self.tau = tau
        self.test_fit_coefficients = test_fit_coefficients
        self.log_freq = log_freq

    def plot_powerfit(self, title='Power law fit plot'):

        n_counts = len(self.fit_v)

        plt.plot(self.logx, self.logv, marker='o', mew=3,
                 color='white', lw=0, ms=10)
        plt.plot(np.arange(n_counts)[:4],
                 self.fit_v[:4], color='red', lw=3)
        plt.xlim([-0.35, 3])
        plt.ylim([self.logv.min() - 0.25, 0])
        plt.xlabel('log$_{10}$(bin index)')
        plt.ylabel('log$_{10}$(frequency)')
        plt.title('nbins = {}'.format(n_counts))
        plt.text(1.85, -1.25, '$\\beta = %.2f$' % self.beta, fontsize=15,
                 bbox={'facecolor': '#1E90FF', 'alpha': 0.5, 'pad': 10})
        plt.show()


class FitCoefficients:

    def __init__(self, polyfit_coeffs):

        self.coeffs = polyfit_coeffs
        self.slope = polyfit_coeffs[0]
        self.intercept = polyfit_coeffs[1]


def experiment_run(msnbc_texts, fox_texts, pct_fox, n_trials, tau):

    if pct_fox < 0 or pct_fox > 100:
        raise RuntimeError('pct_fox must be a value between 0 and 100')

    if pct_fox == 0:

        print('Running 100% MSNBC')
        return _get_run_result(msnbc_texts, pct_fox, tau)

    elif pct_fox == 100:

        print('Running 100% Fox News')
        return _get_run_result(fox_texts, pct_fox, tau)

    print('Running mix of {}% Fox News'.format(pct_fox))

    trial_results = []
    embedding_matrices = []
    for ii in range(n_trials):
        print('{}% mix trial # {} of {}'.format(pct_fox, ii+1, n_trials))
        texts = _mix_texts(msnbc_texts, fox_texts, pct_fox)
        result = _get_run_result(texts, pct_fox, tau)
        trial_results.append(result)
        embedding_matrices.append(result.embedding_matrices[0])

    trial_fit_coefficients = [
        res.fit_coefficients for res in trial_results
    ]

    beta = np.mean([fc.slope for fc in trial_fit_coefficients])
    ave_intercept = np.mean([fc.intercept for fc in trial_fit_coefficients])

    coeffs = FitCoefficients([beta, ave_intercept])

    # don't write log_freq or fit_log_freq since it's not clear how to average
    return ExperimentRunResult(
        pct_fox, coeffs, embedding_matrices, tau, None, trial_fit_coefficients
    )


def _get_run_result(texts, pct_fox, tau):

    e = build_network(texts)

    # build adjacency matrix used to construct graph
    A = np.copy(e.edgeweight_mat)
    A[A <= tau] = 0.0
    A[A > tau] = 1.0

    g = nx.from_numpy_matrix(A)

    degs = np.flipud(np.sort(np.array(list(g.degree().values()))))

    c = list(Counter(degs).items())

    # for each degree, k, calculate the frequency
    freq = np.array([a[1] for a in c], dtype=float)
    freq = freq / freq.sum()

    k = np.arange(len(freq)) + 1

    log_freq = np.log10(freq)
    log_k = np.log10(k)

    coefficients = np.polyfit(log_k, log_freq, 1)

    return ExperimentRunResult(
        pct_fox, FitCoefficients(coefficients), [e.matrix], tau, log_freq
    )


def _mix_texts(msnbc_texts, fox_texts, pct_fox):

    # copy so as not to change self.msnbc_texts from Experiment class
    texts = list(msnbc_texts)
    # texts = msnbc_texts

    # find number of fox texts to include
    n_msnbc_texts = len(texts)
    n_fox_texts = len(fox_texts)
    n_replace = int(np.floor(pct_fox * .01 * n_msnbc_texts))

    # get n_fox_texts random indices from 0 - len(msnbc_texts)
    max_len = min(n_msnbc_texts, n_fox_texts)
    replace_idxs = np.random.choice(max_len, n_replace)

    print(
        'replacing {} of {} MSNBC shows with Fox News shows'.format(
            n_replace, n_msnbc_texts
        )
    )

    for idx in replace_idxs:
        texts[idx] = fox_texts[idx]

    return texts
