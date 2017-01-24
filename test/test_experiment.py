import numpy as np

from unittest import TestCase

from semcable.experiment import make_doc_word_matrix


class TestDocWordMatrix(TestCase):

    def test_make_doc_word_matrix(self):
        '''
        Ensure list of lists with words as entries properly converted to count matrix
        '''
        texts = [['this', 'is', 'a', 'test', 'good', 'test'],
                 ['what', 'a', 'good', 'test', 'this', 'is'],
                 ['wow', 'what', 'a', 'test', 'wow']
                 ]

        generated_dwmat, generated_vocab = make_doc_word_matrix(texts)

        expected_vocab = np.array(
            ['this', 'is', 'a', 'test', 'good', 'what', 'wow']
        )

        expected_dwmat = np.array(
            [[1, 1, 1, 2, 1, 0, 0],
             [1, 1, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 0, 1, 2]
             ],
            dtype=float
        )

        assert (generated_vocab == expected_vocab).all(), generated_vocab
        assert (expected_dwmat == generated_dwmat).all(), generated_dwmat
