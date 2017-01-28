'''
Experimental harness for semantic networks of mixed cable programming
'''
import lda
import io
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle

from collections import Counter
from mongoengine import connect
from nltk.corpus import stopwords
from scipy.optimize import minimize_scalar
from tarfile import TarFile

from .util import vis_graph, get_word_idx_lookup

STOPWORDS = stopwords.words('english')


class Experiment:

    def __init__(self, doc_word_mat, vocab):

        self.doc_word_mat = doc_word_mat
        self.vocab = vocab
        self.model = None
        self.E = None
        self.A = None
        self.power_law_coefficients = None

    def fit_lda(self, n_topics=20, n_iter=1500):

        self.model = lda.LDA(n_topics=n_topics,
                             n_iter=n_iter, random_state=None)

        self.model.fit(self.doc_word_mat)

    def calculate_adjacency(self, k_ave_target=11.67, tol=1e-3):

        if self.model is None:
            raise RuntimeError('LDA model not initialized')

        X = self.model.topic_word_
        Cd = X.sum(axis=0)

        C = 1.0 / Cd
        self.E = C * ((X.T).dot(X))

        self.A = _calculate_adjacency(self.E, k_ave_target, tol)

    def make_graph(self):

        if self.A is None:
            raise RuntimeError('Adjacency matrix not yet built')

        self.graph = nx.from_numpy_matrix(self.A)  # , create_using=nx.DiGraph)

        mapping = dict(enumerate(self.vocab))
        self.graph = nx.relabel_nodes(self.graph, mapping)

        self.degs = np.flipud(
                np.sort(
                    np.array(
                        list(self.graph.degree().values())
                    )
                )
            )

    def fit_powerlaw(self):

        try:
            c = list(Counter(self.degs).items())
        except:
            raise RuntimeError('Experiment.make_graph must be run first')

        # for each degree, k, calculate the frequency
        self.freq = np.array([a[1] for a in c], dtype=float)
        self.freq = self.freq / self.freq.sum()
        self.freq = np.flipud(np.sort(self.freq))

        self.k = np.arange(len(self.freq)) + 1

        self.log_freq = np.log10(self.freq)
        self.log_k = np.log10(self.k)

        self.power_law_coefficients = np.polyfit(self.log_k, self.log_freq, 1)

    def visualize_powerlaw_fit(self, title=None, xlim=[-0.35, 3]):

        try:
            plt.plot(self.log_k, self.log_freq, marker='o', mew=3,
                     color='white', lw=0, ms=10)
        except:
            raise RuntimeError('Experiment.fit_powerlaw must be run first')

        plt.xlim(xlim)
        plt.xlabel('log$_{10}$(bin index)')
        plt.ylabel('log$_{10}$(frequency)')
        plt.show()

    def visualize_graph(self, cue_word, node_color='#CABAC1', alpha=0.5,
                        figsize=(10, 10), layout='graphviz'):
        '''
        visualize the local part of the graph around a list of words and
        maximum degree of nodes away to include
        '''
        if self.A is None:
            raise RuntimeError('Adjacency matrix not yet built')

        vocab_idx_lookup = get_word_idx_lookup(self.vocab)

        cue_idx = vocab_idx_lookup[cue_word]

        a_cue = self.A[:, cue_idx]

        subgraph_idxs = [cue_idx]

        assoc_idxs = np.where(a_cue == 1.0)[0]
        assoc_idxs = assoc_idxs[assoc_idxs != cue_idx]
        # edges = [(cue_idx, ai) for ai in assoc_idxs]

        for cue_idx in assoc_idxs:
            a_cue = self.A[:, cue_idx]
            assoc_idxs = np.where(a_cue == 1.0)[0]
            assoc_idxs = assoc_idxs[assoc_idxs != cue_idx]
            subgraph_idxs.extend(assoc_idxs)
            # edges.extend((cue_idx, ai) for ai in assoc_idxs)

        subgraph = self.graph.subgraph(subgraph_idxs)

        mapping = dict(enumerate(self.vocab))
        subgraph = nx.relabel_nodes(subgraph, mapping)

        vis_graph(subgraph, node_color=node_color, figsize=figsize,
                  alpha=alpha, layout=layout)


def _get_tf_name(var_str, tf_names):
    return next(n for n in tf_names if var_str in n)


class Results:

    def __init__(self, doc_word_matrix, vocab, words_of_interest,
                 woi_edgeweights, graph, degs):

        self.degs = degs
        self.graph = graph

        self.words_of_interest = words_of_interest
        # words of interest edgweights
        self.woi_edgeweights = woi_edgeweights

        self.vocab = vocab
        self.doc_word_matrix = doc_word_matrix

    @classmethod
    def from_tar(cls, tar_path):

        tf = TarFile.open(tar_path)
        tf_names = list(tf.getnames())

        dwm_name = _get_tf_name('doc_word_mat', tf_names)
        vocab_name = _get_tf_name('vocab', tf_names)
        woi_name = _get_tf_name('words_of_interest', tf_names)
        woi_weights_name = _get_tf_name('woi_edgeweights', tf_names)
        graph_name = _get_tf_name('graph', tf_names)
        degs_name = _get_tf_name('degrees', tf_names)

        v = tf.extractfile(vocab_name)
        vocab = [el.decode('unicode_escape').strip() for el in v.readlines()]

        woi = tf.extractfile(woi_name)
        words_of_interest = [el.decode('unicode_escape').strip()
                             for el in woi.readlines()]

        woi_edgeweights = np.load(
            io.BytesIO(tf.extractfile(woi_weights_name).read())
        )
        doc_word_mat = np.load(
            io.BytesIO(tf.extractfile(dwm_name).read())
        )

        degs = pickle.load(tf.extractfile(degs_name))
        graph = pickle.load(tf.extractfile(graph_name))

        return cls(
            doc_word_mat, vocab, words_of_interest,
            woi_edgeweights, graph, degs
        )

    def fit_powerlaw(self):

        try:
            c = list(Counter(self.degs).items())
        except:
            raise RuntimeError('Experiment.make_graph must be run first')

        # for each degree, k, calculate the frequency
        self.freq = np.array([a[1] for a in c], dtype=float)
        self.freq = self.freq / self.freq.sum()
        self.freq = np.flipud(np.sort(self.freq))

        self.k = np.arange(len(self.freq)) + 1

        self.log_freq = np.log10(self.freq)
        self.log_k = np.log10(self.k)

        self.power_law_coefficients = np.polyfit(self.log_k, self.log_freq, 1)

    def visualize_powerlaw_fit(self, title=None, xlim=[-0.35, 3]):

        try:
            plt.plot(self.log_k, self.log_freq, marker='o', mew=3,
                     color='white', lw=0, ms=10)
        except:
            raise RuntimeError('Experiment.fit_powerlaw must be run first')

        plt.xlim(xlim)
        plt.xlabel('log$_{10}$(bin index)')
        plt.ylabel('log$_{10}$(frequency)')
        plt.show()

    def visualize_graph(self, cue_word, node_color='#CABAC1', alpha=0.5,
                        figsize=(10, 10), layout='graphviz'):
        '''
        visualize the local part of the graph around a list of words and
        maximum degree of nodes away to include
        '''
        # if self.A is None:
        #     raise RuntimeError('Adjacency matrix not yet built')

        woi_idx_lookup = get_word_idx_lookup(self.words_of_interest)

        cue_idx = woi_idx_lookup[cue_word]

        a_cue = self.woi_edgeweights[:, cue_idx]
        print(a_cue[:10])
        subgraph_idxs = [cue_idx]

        assoc_idxs = np.where(a_cue == 1.0)[0]
        assoc_idxs = assoc_idxs[assoc_idxs != cue_idx]
        # edges = [(cue_idx, ai) for ai in assoc_idxs]

        # for cue_idx in assoc_idxs:
        #     a_cue = self.A[:, cue_idx]
        #     assoc_idxs = np.where(a_cue == 1.0)[0]
        #     assoc_idxs = assoc_idxs[assoc_idxs != cue_idx]
        #     subgraph_idxs.extend(assoc_idxs)
        #     # edges.extend((cue_idx, ai) for ai in assoc_idxs)

        subgraph_idxs.extend(assoc_idxs)

        subgraph = self.graph.subgraph(subgraph_idxs)

        mapping = dict(enumerate(self.vocab))
        subgraph = nx.relabel_nodes(subgraph, mapping)

        vis_graph(subgraph, node_color=node_color, figsize=figsize,
                  alpha=alpha, layout=layout)


class NetworkStats:

    def __init__(self, power_law_exp):
        self.power_law_exp = power_law_exp


def get_iatv_corpus_doc_data(iatv_corpus_name, network, start_date=None,
                             end_date=None, db_name='metacorps'):

    db = connect().get_database(db_name)

    doc_ids = db.get_collection('iatv_corpus').find_one(
        {'name': iatv_corpus_name}
    )['documents']

    iatv_docs = db.get_collection('iatv_document')

    docs = [iatv_docs.find_one({'_id': doc_id})
            for doc_id in doc_ids]

    docs = [doc['document_data'] for doc in docs if doc['network'] == network]

    return docs


def _calculate_adjacency(edgeweight_matrix, k_ave_target, tol):
    '''

    Arguments:
        edgeweight_matrix (np.array): probabilities P(w_1|w_2)
        k_ave_target (float): empirical average node degree that output
            adjacency graph should match
        tol (float): tolerance to match k_ave_target

    Returns:
        (np.array): Adjacency matrix where k_ave ≈ k_ave_target
    '''
    tau_max = edgeweight_matrix.max()
    tau_min = edgeweight_matrix.min()

    tau = tau_max - tau_min

    n_words = len(edgeweight_matrix)
    denom = 1.0 / n_words

    def f(tau):
        A = edgeweight_matrix.copy()

        A[A > tau] = 1.0
        A[A < 1.0] = 0.0

        k_ave_calc = denom * A.sum()

        k_diff = abs(k_ave_calc - k_ave_target)

        return k_diff

    res = minimize_scalar(f, bounds=(tau_min, tau_max), method='bounded')
    tau = res.x

    A = edgeweight_matrix.copy()
    A[A > tau] = 1.0
    A[A < 1.0] = 0.0

    return A
