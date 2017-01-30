import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import warnings

from collections import Counter, OrderedDict
from datetime import datetime, timedelta
from mongoengine import connect
from nltk.corpus import stopwords
from numpy import array
from scipy.optimize import minimize_scalar


STOPWORDS = set(stopwords.words('english'))


def get_iatv_corpus_names(db_name='metacorps'):

    return [c['name'] for c in
            connect().get_database(
                    db_name
                ).get_collection(
                    'iatv_corpus'
                ).find()
            ]


def get_corpus_text(iatv_corpus_name, network, n_weeks=0, db_name='metacorps',
                    remove_commercials=True):

    return make_texts_list(

        get_iatv_corpus_doc_data(
            iatv_corpus_name, network, n_weeks=n_weeks, db_name=db_name
        ),
        remove_commercials=remove_commercials
    )


def get_iatv_corpus_doc_data(iatv_corpus_name, network,
                             start_date=datetime(2016, 9, 1), n_weeks=None,
                             db_name='metacorps'):

    db = connect().get_database(db_name)

    doc_ids = db.get_collection('iatv_corpus').find_one(
        {'name': iatv_corpus_name}
    )['documents']

    iatv_docs = db.get_collection('iatv_document')

    docs = [iatv_docs.find_one({'_id': doc_id})
            for doc_id in doc_ids]

    if network == 'MSNBCW':
        other_network = 'FOXNEWSW'
    else:
        other_network = 'MSNBCW'

    texts = [
        doc['document_data'] for doc in docs if doc['network'] == network
    ]
    # else:
    #     docs = [
    #         doc['document_data'] for doc in docs
    #         if doc['network'] == network and doc['start_localtime'] <= end_date
    #     ]

    if n_weeks is not None and n_weeks != 0:

        end_date = start_date + timedelta(days=7*n_weeks)

        other_texts = [
            doc['document_data'] for doc in docs
            if doc['network'] == other_network and
            doc['start_localtime'] <= end_date
        ]

        texts = texts + other_texts

    return texts


def text_counts(docs, remove_commercials=True):
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
              if c[word] >= 10 and word not in STOPWORDS]
             for text in texts]

    c = Counter([])
    for t in texts:
        c.update(t)

    return (texts, c)


def make_texts_list(docs, remove_commercials=True):
    '''
    Make texts lists for list of documents
    '''
    if remove_commercials:
        texts = [[word.lower() for word in doc.split()
                  if word.isalpha()
                  and not any(c.islower() for c in word)
                  ]
                 for doc in docs if len(doc) != 0]
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

    return texts


def make_doc_word_matrix(texts):

    c = OrderedCounter([])
    for t in texts:
        c.update(t)

    vocab = list(c.keys())
    n_words = len(vocab)
    n_docs = len(texts)
    doc_word_mat = np.zeros((n_docs, n_words), dtype=int)

    vocab_lookup = get_word_idx_lookup(vocab)

    for t_idx, t in enumerate(texts):
        c = Counter(t)
        for k, v in c.items():
            doc_word_mat[t_idx, vocab_lookup[k]] = v

    return (doc_word_mat, vocab)


def vis_graph(g, node_color='r', figsize=(10, 10),
              layout='graphviz', alpha=0.5,
              labels_x_offset=0.1, labels_y_offset=0.1):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if layout == 'graphviz':
        node_pos = nx.drawing.nx_pydot.graphviz_layout(g)
    elif layout == 'circular':
        node_pos = nx.drawing.layout.circular_layout(g)
    elif layout == 'spectral':
        node_pos = nx.drawing.layout.spectral_layout(g)
    elif layout == 'spring':
        node_pos = nx.drawing.layout.spring_layout(g)
    elif layout == 'shell':
        node_pos = nx.drawing.layout.shell_layout(g)
    else:
        warnings.warn(
            'layout {} not found, defaulting to graphviz'.format(layout)
        )
        node_pos = nx.drawing.nx_pydot.graphviz_layout(g)

    if node_pos not in ['spring']:
        label_pos = {

            k: array(
                [v[0] + labels_x_offset, v[1] + labels_y_offset]
            )

            for k, v in node_pos.items()
        }

    nx.draw_networkx_labels(g, font_weight='bold', font_size=18, pos=label_pos)
    nx.draw_networkx_nodes(g, node_size=500,
                           node_color=node_color, pos=node_pos, alpha=alpha)
    nx.draw_networkx_edges(g, pos=node_pos, edge_color='grey', width=1.0)

    return fig, ax


# see https://docs.python.org/3.4/library/collections.html?highlight=ordereddict#ordereddict-examples-and-recipes
class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def get_word_idx_lookup(vocab):

    return dict((el, idx) for idx, el in enumerate(vocab))


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

        del A

        return k_diff

    res = minimize_scalar(f, bounds=(tau_min, tau_max), method='bounded')
    tau = res.x

    A = edgeweight_matrix.copy()
    A[A > tau] = 1.0
    A[A < 1.0] = 0.0

    return A
