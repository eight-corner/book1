from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset='all')
print(len(news.data))

from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

from sklearn.feature_extraction.text import  CountVectorizer
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_predict = mnb.predict(X_test)

from sklearn.metrics import classification_report
classification_report(y_test, y_predict, target_names=news.target_names)

# 广泛应用于海量互联网文本分类任务
# 特征独立的假设，使得参数规模从幂指数量级降到线性级别
# 在特征关联较强的分类任务上性能表现不佳

