import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from Log import *

train = pd.read_csv('/home/ys/PycharmProjects/book1/Datasets/IMDB/labeledTrainData.tsv', delimiter='\t')
test = pd.read_csv('/home/ys/PycharmProjects/book1/Datasets/IMDB/testData.tsv', delimiter='\t')
# print(train.info)


def review_to_text(review, remove_stopwords):
    raw_text = BeautifulSoup(review, 'html').get_text()

    letters = re.sub('[^a-zA-Z]', ' ', raw_text)

    words = letters.lower().split()

    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]

    return words


X_train = []

for review in train['review']:
    X_train.append(' '.join(review_to_text(review, True)))

y_train = train['sentiment']

X_test = []

for review in test['review']:
    X_test.append(' '.join(review_to_text(review, True)))

from sklearn.feature_extraction.text import CountVectorizer,  TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# pip_tfidf = Pipeline([('tfidf_vec', TfidfVectorizer(analyzer='word')), ('mnb', MultinomialNB())])
# params_tfidf = {'tfidf_vec__binary':[True, False], 'tfidf_vec__ngram_range':[(1, 1), (1, 2)], 'mnb__alpha':[0.1, 1.0, 10.0]}

pip_tfidf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('mnb', MultinomialNB()),
])

params_tfidf = {'vect__max_df': (0.5, 0.75, 1.0),
                'vect__max_features': (None, 5000, 10000, 50000),
                'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
                'tfidf__use_idf': (True, False),
                'tfidf__norm': ('l1', 'l2'),
                'mnb__alpha': (0.1, 1.0, 10.0)}

logger.info("begin grid......")
gs_tfidf = GridSearchCV(pip_tfidf, params_tfidf, cv=4, n_jobs=-1, verbose=1)
gs_tfidf.fit(X_train, y_train)

logger.info(gs_tfidf.best_score_)
logger.info(gs_tfidf.best_params_)
tfidf_y_predict = gs_tfidf.predict(X_test)
logger.info(gs_tfidf.best_score_)
logger.info("end grid......")

# submission_tfidf = pd.DataFrame({'id': test['id'], 'sentiment': tfidf_y_predict})
# submission_tfidf.to_csv('./Datasets/IMDB/submission_tfidf.csv', index=False)
