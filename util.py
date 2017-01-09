from collections import Counter
from mongoengine import connect
from nltk.corpus import stopwords


STOPWORDS = set(stopwords.words('english'))


def get_iatv_corpus_names(db_name='metacorps'):

    return [c['name'] for c in
            connect().get_database(
                    db_name
                ).get_collection(
                    'iatv_corpus'
                ).find()
            ]


def get_iatv_corpus_doc_data(iatv_corpus_name, network, db_name='metacorps'):

    db = connect().get_database(db_name)

    doc_ids = db.get_collection('iatv_corpus').find_one(
        {'name': iatv_corpus_name}
    )['documents']

    iatv_docs = db.get_collection('iatv_document')

    docs = [iatv_docs.find_one({'_id': doc_id})
            for doc_id in doc_ids]

    docs = [doc['document_data'] for doc in docs if doc['network'] == network]

    return docs


def text_counts(docs):
    '''
    Make word count for list of documents
    '''
    texts = [[word for word in doc.lower().split()
              if word.isalpha() and word not in STOPWORDS]
             for doc in docs]

    c = Counter([])
    for t in texts:
        c.update(t)

    texts = [[word for word in text[1:]  # remove 1st word always 'transcript'
              if c[word] >= 10]
             for text in texts]

    c = Counter([])
    for t in texts:
        c.update(t)

    return (texts, c)

# texts, counts = text_counts(docs)


def get_corpus_text(iatv_corpus_name, network, db_name='metacorps'):

    return text_counts(
        get_iatv_corpus_doc_data(iatv_corpus_name, network, db_name=db_name)
    )
