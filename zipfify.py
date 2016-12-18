from collections import Counter
from mongoengine import connect
from gensim import corpora
# from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords

import matplotlib.pyplot as plt

STOPWORDS = set(stopwords.words('english'))


c = connect()
db = c.get_database('metacorps')

iatv_corpus = db.get_collection('iatv_corpus')
iatv_docs = db.get_collection('iatv_document')

zipf = iatv_corpus.find_one({'name': 'Sample Week for Zipf experiment'})

docs = [iatv_docs.find_one({'_id': doc_id}) for doc_id in zipf['documents']]

msnbc_docs = [doc for doc in docs if doc['network'] == 'MSNBCW']
fox_docs = [doc for doc in docs if doc['network'] == 'FOXNEWSW']


def calculate_counts(docs):
    '''
    Make word count for list of documents
    '''
    texts = [[word for word in doc['document_data'].lower().split()
              if word.isalpha() and word not in STOPWORDS]
             for doc in docs]

    c = Counter([])
    for t in texts:
        c.update(t)

    texts = [[word for word in doc['document_data'].lower().split()
              if c['word'] >= 10]
             for doc in docs]

    return (texts, c)


msnbc_texts, msnbc_counts = calculate_counts(msnbc_docs)
msnbc_counts = list(msnbc_counts.items())

fox_texts, fox_counts = calculate_counts(fox_docs)
fox_counts = list(fox_counts.items())

msnbc_counts.sort(key=lambda x: -x[1])
fox_counts.sort(key=lambda x: -x[1])

y_m = [a[1] for a in msnbc_counts]
y_f = [a[1] for a in fox_counts]

plt.loglog(y_m, lw=3, label='MSNBC')
plt.loglog(y_f, lw=3, label='FOX')

plt.legend()


def build_corpora(fox_texts, msnbc_texts):
    fox_dict = corpora.Dictionary(fox_texts)
    msnbc_dict = corpora.Dictionary(msnbc_texts)

    fox_dict.save('fox_dict')
    msnbc_dict.save('msnbc_dict')

    fox_corpus = [fox_dict.doc2bow(text) for text in fox_texts]
    msnbc_corpus = [msnbc_dict.doc2bow(text) for text in msnbc_texts]

    corpora.MmCorpus.serialize('fox_corpus.mm', fox_corpus)
    corpora.MmCorpus.serialize('msnbc_corpus.mm', msnbc_corpus)

    return (fox_dict, msnbc_dict, fox_corpus, msnbc_corpus)


fox_dict, msnbc_dict, fox_corpus, msnbc_corpus = \
    build_corpora(fox_texts, msnbc_texts)
