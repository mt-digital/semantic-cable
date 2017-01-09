'''
Tools for building and analyzing a semantic network starting from
a list of lists of text

Author: Matthew A Turner
Date: 20 December 2016
'''
import networkx as nx

from numpy.linalg import norm  # , svd
from numpy import arccos, zeros, log, flipud
from nltk.corpus import stopwords
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import normalize

STOPWORDS = stopwords.words('english')


class LookupCount:
    def __init__(self, index):
        self.index = index
        self.count = 1


# we first need to iterate through the texts to find $f_q(x, y)$
# for all $x$ and $y$
def _calculate_ppmi(texts, window_distance, alpha,
                    verbose=False, saveroot='ppmi_calculation'):
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
    norm = 1.0 / N
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
        P_word = norm * lookup_count.count

        relevant_pairs = pairs_lookup[word]
        # relevant_pairs = [
        #     wc_pair for wc_pair in word_context_pairs if word == wc_pair[0]
        # ]

        for pair in relevant_pairs:

            context = pair[1]
            context_wlc = word_lookup_counts[context]

            j = context_wlc.index

            P_context = norm * context_wlc.count
            P_joint = norm * context_lookup_counts[pair]

            pmi_val = log(P_joint / (P_context * P_word))

            ppmi_matrix[i, j] = pmi_val if pmi_val > 0 else 0

    #np.savetxt(saveroot + '.ppmi_matrix', ppmi_matrix)

    return word_lookup_counts, context_lookup_counts, ppmi_matrix


def cos_sim(x, y):
    if norm(x)*norm(y) > 0:
        return x.dot(y) / (norm(x)*norm(y))
    else:
        return 0


def ham(x, y):
    return arccos(- cos_sim(x, y))


# def calculate_edgeweights(U_reduced):
def calculate_edgeweights(embedding_mat):

    normed_embeddings = normalize(embedding_mat, norm='l2', axis=1)

    return arccos(-normed_embeddings.dot(normed_embeddings.T))

    # nrows = U_reduced.shape[0]
    # combos = [(i, j) for i in range(nrows) for j in range(nrows) if i != j]

    # weights = {
    #     (i, j): ham(U_reduced[i], U_reduced[j])
    #     for i, j in combos
    # }

    # return weights


# TODO these variable names are bad: word index and word lookup...
def get_knn(word, edgeweight_mat, k, word_index_counts, word_lookup_table):
    word_idx = word_index_counts[word].index

    # get similarity vector from the edgeweight matrix
    word_sim_vec = edgeweight_mat[word_idx]
    k_best_indices = flipud(word_sim_vec.argsort())[:k]

    return [
        ((word, word_lookup_table[nn_idx]), edgeweight_mat[word_idx, nn_idx])
        for nn_idx in k_best_indices
    ]
    # edge_list = edges.items()
    # word_edges = list(filter(lambda x: x[0][0] == word_idx, edge_list))

    # word_edges.sort(key=lambda x: -x[1])

    # knn = word_edges[:k]

#     ret = [
#         ((word, word_lookup_table[edge[0][1]]), edge[1])
#         for edge in knn
#     ]

    # return ret


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

        # U, _, _ = svd(ppmi.matrix)
        # embeddings = U[:, :embedding_dim]

        embeddings, _, _ = randomized_svd(ppmi.matrix, embedding_dim)

        # return cls(embeddings, word_lookup, index_lookup, U_full=U)
        new_embedding = cls(embeddings, word_lookup, index_lookup)

        new_embedding.word_lookup_counts = word_lookup_counts

        return new_embedding

    def make_edgweight_mat(self):

        if self.matrix is None:
            raise RuntimeError('Must first create embedding matrix')

        self.edgeweight_mat = calculate_edgeweights(self.matrix)

    # def generate_graph(self, tau, words=None):
    def generate_graph(self, k, words=None):

        if self.edgeweight_mat is None:
            self.edgeweight_mat = calculate_edgeweights(self.matrix)

        # network = SemanticNetwork(
        #     matrix, tau, self.word_lookup_counts, self.index_lookup,
        #     words=words
        # )
        return make_graph(self.edgeweight_mat, k, self.word_lookup_counts,
                          self.index_lookup, words=words)

    # def from_texts(cls, texts, words=None, k=5, embedding_dim=300,
    #                window_distance=5, alpha=0.75, ):
    #     '''
    #     Generate k-nearest neighbor embedding from a set of texts.

    #     Arguments:
    #         texts list(list): list of lists of ordered word
    #             (i.e. list of documents)
    #         k (int): number of nearest neighbors to include in graph
    #         embedding_dim (int): number of columns from U matrix in SVD to
    #             include in embedded word vectors
    #         window_distance (int): window to use for PPMI calculation
    #         alpha (float): value between 0 and 1 for smoothing context counts
    #             in PPMI calculation
    #     '''

    #     print('Calculating PPMI')
    #     word_lookup_counts, _, ppmi_mat = \
    #         _calculate_ppmi(texts, window_distance, alpha)

    #     U, s, V = svd(ppmi_mat)

    #     U_reduced = U[:, :embedding_dim]

    #     word_lookup = {k: v.index for k, v in word_lookup_counts.items()}
    #     index_lookup = _index_lookup_table(word_lookup_counts)
    #     print('Calculating Edge Weights')
    #     matrix = calculate_edgeweights(U_reduced)


    #     print('Making graph')
    #     graph = make_graph(
    #         matrix, k, word_lookup_counts, index_lookup, words=words
    #     )

    #     return cls(matrix, graph, word_lookup, index_lookup)


class SemanticNetwork:

    def __init__(self, *args):
        return None


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


def make_graph(edges, k, word_index_counts, word_lookup_table, words=None):

    if words is None:
        words = word_index_counts.keys()

    G = nx.Graph()
    G.add_nodes_from(words)

    for i, word in enumerate(words):
        print('on word {} out of {}'.format(i, len(words)))
        try:
            knn = get_knn(word, edges, k, word_index_counts, word_lookup_table)
            # ignore weights for now
            G.add_edges_from([e[0] for e in knn])
        except KeyError:
            print('word {} not found'.format(word))

    return G


def _index_lookup_table(word_index_counts):

    return dict((wic.index, word) for word, wic in word_index_counts.items())
