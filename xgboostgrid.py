import pandas as pd

train = pd.read_csv('/home/ys/PycharmProjects/book1/Datasets/Titanic/train.csv')
test = pd.read_csv('/home/ys/PycharmProjects/book1/Datasets/Titanic/test.csv')
print(train.info)

selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch','Fare']
X_train = train[selected_features]
y_train = train['Survived']
X_test = test[selected_features]
X_train['Embarked'].fillna('S', inplace=True)
X_test['Embarked'].fillna('S', inplace=True)
X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)
print(X_train.head())

from sklearn.feature_extraction import DictVectorizer
dict_vec = DictVectorizer(sparse=False)
X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
print(X_train[0])
print(dict_vec.feature_names_)
X_test = dict_vec.transform(X_test.to_dict(orient='record'))

from sklearn.model_selection import GridSearchCV

params = {'max_depth':range(2, 7), 'n_estimators':range(100, 1100, 200), 'learning_rate':[0.05, 0.1, 0.25, 0.5, 1.0]}

from xgboost import XGBClassifier
xgbc_best = XGBClassifier()

gs = GridSearchCV(xgbc_best, params, n_jobs=-1, cv=5, verbose=1)

gs.fit(X_train, y_train)

print(gs.best_score_)
print(gs.best_params_)

xgbc_best_y_predict = gs.predict(X_test)
xgbc_best_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_best_y_predict})
xgbc_best_submission.to_csv('./Datasets/Titanic/xgbc_best_submission.csv', index=False)
