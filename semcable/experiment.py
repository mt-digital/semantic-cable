'''
Experimental harness for semantic networks of mixed cable programming
'''
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import lda

from collections import Counter
from mongoengine import connect
from nltk.corpus import stopwords
from scipy.optimize import minimize_scalar

from util import vis_graph

STOPWORDS = stopwords.words('english')


class Experiment:

    def __init__(self, doc_word_mat, vocab):

        self.doc_word_mat = doc_word_mat
        self.vocab = vocab
        self.model = None
        self.E = None
        self.A = None
        self.power_law_coefficients = None

    def fit_lda(self, n_topics=20):

        self.model = lda.LDA(n_topics=n_topics, random_state=None)
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
                        figsize=(10, 10), layout='spectral'):
        '''
        visualize the local part of the graph around a list of words and
        maximum degree of nodes away to include
        '''
        if self.A is None:
            raise RuntimeError('Adjacency matrix not yet built')

        vocab_idx_lookup = get_word_idx_fun(self.vocab)

        cue_idx = vocab_idx_lookup(cue_word)

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


class NetworkStats:

    def __init__(self, power_law_exp):
        self.power_law_exp = power_law_exp


def make_doc_word_matrix(docs, remove_commercials=True):
    '''
    Make word count for list of documents
    '''
    if remove_commercials:
        texts = [[word.lower() for word in doc.split()
                  if word.isalpha()
                  and not any(c.islower() for c in word)
                  ]
                 for doc in docs]
    else:
        texts = [[word.lower() for word in doc.split()
                  if word.isalpha()
                  ]
                 for doc in docs]

    c = Counter([])
    for t in texts:
        c.update(t)
    # remove 1st word always 'transcript'; also lower now
    texts = [[word for word in text[1:]
              if c[word] >= 1 and word not in STOPWORDS]
             for text in texts]

    c = Counter([])
    for t in texts:
        c.update(t)

    vocab = list(c.keys())
    n_words = len(vocab)
    n_docs = len(texts)
    doc_word_mat = np.zeros((n_docs, n_words))

    vocab_lookup = get_word_idx_fun(vocab)

    for t_idx, t in enumerate(texts):
        c = Counter(t)
        for k, v in c.items():
            doc_word_mat[t_idx, vocab_lookup(k)] = v

    return (doc_word_mat, vocab)


def get_word_idx_fun(vocab):

    def fun(word):
        return [i for i, el in enumerate(vocab) if el == word][0]

    return fun


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