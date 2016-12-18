from collections import Counter
from mongoengine import connect
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))


c = connect()
db = c.get_database('metacorps')

doc_ids = db.get_collection('iatv_corpus').find_one(
        {'name': 'Three Months for Semantic Network Experiments'}
    )['documents']

iatv_docs = db.get_collection('iatv_document')

docs = [iatv_docs.find_one({'_id': doc_id})['document_data']
        for doc_id in doc_ids]


def calculate_counts(docs):
    '''
    Make word count for list of documents
    '''
    texts = [[word for word in doc.lower().split()
              if word.isalpha() and word not in STOPWORDS]
             for doc in docs]

    c = Counter([])
    for t in texts:
        c.update(t)

    texts = [[word for word in text
              if c[word] >= 10]
             for text in texts]

    c = Counter([])
    for t in texts:
        c.update(t)

    return (texts, c)

texts, counts = calculate_counts(docs)
