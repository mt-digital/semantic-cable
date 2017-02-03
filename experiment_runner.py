import numpy as np
import pickle
import os

from sys import argv

from semcable.experiment import Experiment
from semcable.util import make_doc_word_matrix, get_corpus_text


def print_help():

    print('''
Usage:\n python experiment_runner.py <base_corpus> <n_weeks> <save_dir>
''')

try:
    base_corpus_network = argv[1]

    if base_corpus_network == '-h':
        print_help()

    n_weeks = int(argv[2])
    save_dir = argv[3]

    if len(argv) >= 5:
        n_iter = int(argv[4])
    else:
        n_iter = 1500

    if len(argv) == 6:
        n_topics = int(argv[5])
    else:
        n_topics = 110

except:
    print_help()

# texts = get_corpus_text('Sample Week for Zipf experiment',
if n_weeks == 0:
    texts = get_corpus_text('Three Months for Semantic Network Experiments',
                            base_corpus_network)
else:
    texts = get_corpus_text(
        'Three Months for Semantic Network Experiments',
        base_corpus_network,
        n_weeks=n_weeks
    )

doc_word_mat, vocab = make_doc_word_matrix(texts)

ex = Experiment(doc_word_mat, vocab)
ex.fit_lda(n_topics=n_topics, n_iter=n_iter)
print('\nFinished fitting LDA. Fitting adjacency\n')
ex.calculate_adjacency()
print('\nMaking graph\n')
ex.make_graph()
print('\nFitting powerlaw\n')
ex.fit_powerlaw()

opj = os.path.join

# write graph and powerlaw fit data
pickle.dump(ex.graph, open(opj(save_dir, 'graph'), 'wb'))
pickle.dump(ex.degs, open(opj(save_dir, 'degrees'), 'wb'))

# write log likelihoods used for parameterization
pickle.dump(ex.model.loglikelihoods_,
            open(opj(save_dir, 'loglikelihoods'), 'wb'))

# select and write words of interest vectors
words_of_interest = [
    'swamp', 'trump', 'environment', 'regulations',
    'regulators', 'economy', 'jobs', 'manufacturing', 'immigrant',
    'immigration', 'reform', 'drain', 'attack', 'hit', 'punch', 'clinton',
    'hillary', 'bill', 'donald', 'pelosi', 'ryan', 'paul', 'congress',
    'senate', 'obama', 'washington', 'dc', 'environment', 'epa',
    'lobbyists', 'lobbyist', 'lobbying', 'attacking', 'hitting', 'attacked',
    'lashed', 'punched', 'punching', 'isis', 'iraq', 'peace', 'war',
    'battle', 'fight', 'fought', 'unemployment', 'millenial', 'millenials',
    'vote', 'voting', 'voter', 'fraud', 'tax', 'taxes', 'returns'
]


with open(opj(save_dir, 'words_of_interest'), 'w') as f:
    for w in words_of_interest:
        f.write(w + '\n')

vocab_lookup = dict((w, i) for i, w in enumerate(vocab))
woi_indexes = [vocab_lookup[w] for w in words_of_interest]

e_woi = ex.E[:, woi_indexes]
np.save(opj(save_dir, 'woi_edgeweights'), e_woi)

# save doc word
np.save(os.path.join(opj(save_dir, 'doc_word_mat')), doc_word_mat)
# save vocab
with open(opj(save_dir, 'vocab'), 'w') as f:
    for v in vocab:
        f.write(v + '\n')

open(opj(save_dir, 'lda_params'), 'w').write(
    '{{"n_topics": {}, "n_iter": {}}}'.format(n_topics, n_iter)
)

# np.save(opj(save_dir, 'edgeweights'), ex.E)
# np.save(opj(save_dir, 'adjacency'), ex.A)
