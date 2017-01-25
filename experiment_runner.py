import logging
import numpy as np
import pickle

from sys import argv

from semcable.experiment import Experiment
from util import make_doc_word_matrix, get_corpus_text

def print_help():

    print('''
Usage:\n python experiment_runner.py <base_corpus> <n_weeks> <pickle_filename>
''')

try:
    base_corpus_network = argv[1]

    if base_corpus_network== '-h':
        print_help()

    n_weeks = argv[2]
    pickle_filename = argv[3]

    if len(argv) == 5:
        n_iter = int(argv[4])
    else:
        n_iter = 1500

except:
    print_help()

# logging.basicConfig(datefmt='%m-%d %H:%M:%S',
#                     filename='/home/mturner8/logs/semcable/' + pickle_filename,
#                     filemode='w')

texts = get_corpus_text('Three Months for Semantic Network Experiments',
                         base_corpus_network)
# texts = get_corpus_text('Sample Week for Zipf experiment',

doc_word_mat, vocab = make_doc_word_matrix(texts)

ex = Experiment(doc_word_mat, vocab)
ex.fit_lda(n_topics=80, n_iter=n_iter)
print('\nFinished fitting LDA. Fitting adjacency\n')
ex.calculate_adjacency()
print('\nMaking graph\n')
ex.make_graph()
print('\nFitting powerlaw\n')
ex.fit_powerlaw()


pickle.dump(ex, open(pickle_filename, 'wb'), protocol=4)


# # hack to deal with OS X pickle implementation bug; fails on files over 2G
# # see https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
# MAX_BYTES = 2**31 - 1

# with open('n_trials=20;d_pct=6.25', 'wb') as f:

#     pickle_bytearray = bytearray(pickle.dumps(ex))
#     ba_len = len(pickle_bytearray)

#     for idx in range(0, ba_len, MAX_BYTES):
#         f.write(pickle_bytearray[idx:idx+MAX_BYTES])

