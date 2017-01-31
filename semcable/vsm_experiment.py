import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx
import nltk
import numpy as np
import os
import pickle

from collections import Counter
from numpy import array, zeros, log, power
from numpy.linalg import norm
from pandas import DataFrame
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import normalize

from build_network import build_network
from .util import get_corpus_text, _calculate_adjacency, vis_graph


# VSM_EXPERIMENT_IATV_CORPUS = 'Three Months for Semantic Network Experiments'
VSM_EXPERIMENT_IATV_CORPUS = 'Sample Week for Zipf experiment'

plt.style.use('ggplot')


class LookupCount:
    def __init__(self, index):
        self.index = index
        self.count = 1


class VSMResult:

    def __init__(self, ppmi, embedding):

        self.ppmi = ppmi
        self.embedding = embedding

    def make_graph(self, k_ave=20.16, tol=1e-3):

        self.graph = nx.relabel_nodes(
            nx.from_numpy_matrix(
                _calculate_adjacency(self.embedding.edgeweight_mat, k_ave, tol)
            ),
            self.embedding.index_lookup
        )

    def get_top_n(self, cue_word, exclude_list=[], exclude_adverbs=False,
                  n=None):
        emb = self.embedding
        wl = emb.word_lookup

        associates = [el for el in list(self.graph[cue_word].keys())
                      if el not in exclude_list]
        associates.remove(cue_word)

        if exclude_adverbs:
            associates = [
                el[0] for el in nltk.pos_tag(associates) if el[1] != 'RB'
            ]

        associate_idxs = [wl[a] for a in associates]
        cue_idx = wl[cue_word]

        ew = emb.edgeweight_mat
        associate_weights = ew[associate_idxs, cue_idx]

        rel_weights = associate_weights / associate_weights.sum()

        assoc_weights = [
            (associates[i], rel_weights[i]) for i in range(len(associates))
        ]

        assoc_weights.sort(key=lambda x: -x[1])

        if n is not None and n > 0:
            assoc_weights = assoc_weights[:n]

        return assoc_weights

    def vis_subgraph(self, word, exclude_adverbs=True,
                     exclude_list=[], figsize=(20, 12),
                     node_color='dodgerblue', labels_x_offset=10,
                     labels_y_offset=5, **kwargs):
        '''
        Build and visualize a subgraph centered around input word

        **kwargs: see util.vis_graph

        Returns (networkx.Graph)
        '''
        neighbors = [el for el in list(self.graph[word].keys())
                     if el not in exclude_list]

        sg_nodes = [word] + neighbors
        # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        if exclude_adverbs:
            sg_nodes = [
                el[0] for el in nltk.pos_tag(sg_nodes) if el[1] != 'RB'
            ]

        subg = self.graph.subgraph(sg_nodes)

        vis_graph(subg, figsize=figsize, node_color=node_color,
                  labels_x_offset=labels_x_offset,
                  labels_y_offset=labels_y_offset)


def generate_barh_plots(save_dir, cue_words, msnbc_result, fox_result,
                        exclude_lookup=None, n=15, msnbc_color='dodgerblue',
                        fox_color='red'):
    '''
    Arguments:
        save_dir (str): directory to place plots. must exist
        cue_words (list(str)):  list of words to make associate plots of
        vsm_result_dict (dict): {'FOXNEWSW': fox_vsm_result,
                                 'MSNBCW': msnbc_vsm_result}
        exclude_lookup (dict): lookup for any of the cue_words that have
            words that should be excluded

    Returns:
        None
    '''

    for cue in cue_words:
        fox_topn = fox_result.get_top_n(cue, n=n)
        fox_topn.reverse()

        msnbc_topn = msnbc_result.get_top_n(cue, n=n)
        msnbc_topn.reverse()

        fox_index = [a[0] for a in fox_topn]
        fox_data = [a[1] for a in fox_topn]
        fox_df = DataFrame(index=fox_index, data=fox_data)

        msnbc_index = [a[0] for a in msnbc_topn]
        msnbc_data = [a[1] for a in msnbc_topn]
        msnbc_df = DataFrame(index=msnbc_index, data=msnbc_data)

        fig, axes = plt.subplots(nrows=1, ncols=2)

        msnbc_df.plot(kind='barh', legend=False, color=msnbc_color, ax=axes[0])
        fox_df.plot(kind='barh', legend=False, color=fox_color, ax=axes[1])

        axes[0].set_title('MSNBC', fontsize=12)
        axes[1].set_title('Fox News', fontsize=12)

        for ax in axes:
            ax.set_xlabel('relative association strength')
            xticks = ax.get_xicks()
            ax.set_xticks(xticks[::2])

        st = fig.suptitle("cue: '{}'".format(cue))
        st.set_x(0.55)

        plt.tight_layout()
        fig.subplots_adjust(top=0.85)

        PdfPages(os.path.join(save_dir, cue + '.pdf')).savefig(fig)


class VSMExperiment:

    def __init__(self, base_network, n_weeks_mix=0,
                 window_width=4, alpha=0.75):

        self.base_network = base_network
        self.n_weeks_mix = n_weeks_mix
        self.window_width = window_width
        self.alpha = alpha

    def load_texts(self):

        self.texts = get_corpus_text(
            VSM_EXPERIMENT_IATV_CORPUS, self.base_network,
            n_weeks=self.n_weeks_mix
        )

    def calculate_ppmi(self):

        self.ppmi = PPMI.from_texts(self.texts, self.window_width, self.alpha)

    def calculate_embeddings(self, embedding_dim=300):

        self.embedding = Embedding.from_ppmi(self.ppmi)

    def calculate_edgeweights(self):

        self.edgeweights = _calculate_edgeweights(self.embedding.matrix)

    def calculate_adjacency(self, k_ave_target=20.16):
        '''
        Caculate adjacency matrix with average k matching k_ave_target
        '''
        self.A = _calculate_adjacency(self.edgeweights, k_ave_target, 1e-3)

    def write_outputs(self, save_dir):
        '''
        Write edgeweights, vocab, and adjacency matrix
        '''
        def p(fname): return os.path.join(save_dir, fname)

        np.save(self.edgeweights, p('edgeweights'))
        np.save(self.adjancency, p('adjancency'))

        pickle.dump(self.embedding.word_lookup, p('word_lookup'))
        pickle.dump(self.embedding.index_lookup, p('index_lookup'))
        pickle.dump(self.ppmi, p('ppmi'))

    # def fit_powerlaw(self):

    #     freq = np.flipud(np.sort(self.A.sum(axis=1)))
    #     total_edges = freq.sum()
    #     # # for each degree, k, calculate the frequency
    #     # freq = np.array([a[1] for a in c], dtype=float)
    #     self.freq = freq / total_edges

    #     self.k = np.arange(len(freq)) + 1

    #     self.log_freq = np.log10(self.freq)
    #     self.log_k = np.log10(self.k)

    #     self.powerlaw_coefficients = \
    #         FitCoefficients(np.polyfit(self.log_k, self.log_freq, 1))

    def generate_graph(self):

        pass


class Embedding:

    matrix = None
    edgeweight_mat = None

    # we'll need to look up words by their indices and vice versa
    word_lookup_counts = None
    word_lookup = None
    index_lookup = None

    def __init__(self, matrix, word_lookup, index_lookup, U_full=None):
        '''
        Arguments:
            matrix (numpy.array): reduced U from SVD
            graph (networkx.Graph): k-nn graph representation of embeddings
            word_lookup (dict): (word, index) pairs for reverse lookups
            index_lookup (dict): (index, word) pairs for word lookups in matrix
        '''
        self.matrix = matrix
        self.word_lookup = word_lookup
        self.index_lookup = index_lookup

    @classmethod
    def from_ppmi(cls, ppmi, embedding_dim=300):

        word_lookup_counts = ppmi.word_lookup_counts

        word_lookup = {k: v.index for k, v in word_lookup_counts.items()}
        index_lookup = _index_lookup_table(word_lookup_counts)

        embeddings, _, _ = randomized_svd(ppmi.matrix, embedding_dim)

        new_embedding = cls(embeddings, word_lookup, index_lookup)

        new_embedding.word_lookup_counts = word_lookup_counts

        return new_embedding

    def make_edgeweight_mat(self):

        if self.matrix is None:
            raise RuntimeError('Must first create embedding matrix')

        self.edgeweight_mat = _calculate_edgeweights(self.matrix)

    # def generate_graph(self, k, words=None):

    #     if self.edgeweight_mat is None:
    #         self.edgeweight_mat = _calculate_edgeweights(self.matrix)

    #     return make_graph(self.edgeweight_mat, k, self.word_lookup_counts,
    #                       self.index_lookup, words=words)


def _calculate_edgeweights(embedding_mat):

    normed_embeddings = normalize(embedding_mat, norm='l2', axis=1)

    # return arccos(-normed_embeddings.dot(normed_embeddings.T))
    return normed_embeddings.dot(normed_embeddings.T)


class PPMI:

    matrix = None
    word_lookup_counts = None

    def __init__(self, matrix, word_lookup_counts):
        self.matrix = matrix
        self.word_lookup_counts = word_lookup_counts

    @classmethod
    def from_texts(cls, texts, window_length=4, alpha=1.0, verbose=False):

        word_lookup_counts, _, matrix = \
            _calculate_ppmi(texts, window_length, alpha, verbose=verbose)

        return cls(matrix, word_lookup_counts)


# we first need to iterate through the texts to find $f_q(x, y)$
# for all $x$ and $y$
def _calculate_ppmi(texts, window_distance, alpha, verbose=False):
    '''

    '''
    word_lookup_counts = {}
    context_lookup_counts = {}
    word_idx = 0

    # step 1) individual word counts and joint word/context counts
    for idx, text in enumerate(texts):
        if verbose:
            print('on text {} out of {} calculating PPMI matrix'.format(
                    idx, len(texts)
                )
            )
        for ii, word in enumerate(text):

            # handle single-word frequency
            if word not in word_lookup_counts:
                word_lookup_counts.update({word: LookupCount(word_idx)})
                word_idx += 1
            else:
                word_lookup_counts[word].count += 1

            # determine f_q(x, y)
            # first, window indices
            w1 = ii + 1
            w2 = w1 + window_distance
            for context in text[w1:w2]:
                if (word, context) not in context_lookup_counts:
                    context_lookup_counts.update({(word, context): 1})
                else:
                    context_lookup_counts[(word, context)] += 1

            # now look previous
            if ii > 0:
                w = max(ii - window_distance, 0)
                for context in text[w:ii]:
                    if (word, context) not in context_lookup_counts:
                        context_lookup_counts.update({(word, context): 1})
                    else:
                        context_lookup_counts[(word, context)] += 1

    # step 2) process word and joint counts to calculate PPMI matrix

    N = len(word_lookup_counts.items())
    norm_coeff = 1.0 / N

    # context smoothing "alleviates bias towards rare words"
    # Levy, O., Goldberg, Y., & Dagan, I. (2015).
    # Transactions of the Association for Computational Linguistics, 3, 211–225
    scaled_context_vec = power(
        array([v.count for v in word_lookup_counts.values()]),
        alpha
    )
    scaled_context_norm_coeff = 1.0 / norm(scaled_context_vec)

    ppmi_matrix = zeros((N, N))

    word_context_pairs = set(context_lookup_counts.keys())

    pairs_lookup = {}
    for wc_pair in word_context_pairs:
        word = wc_pair[0]
        if word in pairs_lookup:
            pairs_lookup[word].append(wc_pair)
        else:
            pairs_lookup.update({word: [wc_pair]})

    for word, lookup_count in word_lookup_counts.items():

        i = lookup_count.index
        P_word = norm_coeff * lookup_count.count

        relevant_pairs = pairs_lookup[word]

        for pair in relevant_pairs:

            context = pair[1]
            context_wlc = word_lookup_counts[context]

            j = context_wlc.index

            P_context = \
                scaled_context_norm_coeff * pow(context_wlc.count, alpha)

            P_joint = norm_coeff * context_lookup_counts[pair]

            pmi_val = log(P_joint / (P_context * P_word))

            ppmi_matrix[i, j] = pmi_val if pmi_val > 0 else 0

    return word_lookup_counts, context_lookup_counts, ppmi_matrix


class Experiment:

    def __init__(self, n_trials, pct_fox_vals=[0, 25, 50, 75, 100], tau=0.6):

        self.pct_fox_vals = pct_fox_vals
        self.tau = tau
        self.run_dict = None
        self.n_trials = n_trials

    def load_texts(self, corpus_name):

        self.fox_texts = get_corpus_text(corpus_name, 'FOXNEWSW')
        self.msnbc_texts = get_corpus_text(corpus_name, 'MSNBCW')

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


def _index_lookup_table(word_index_counts):

    return dict((wic.index, word) for word, wic in word_index_counts.items())
