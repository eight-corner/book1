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


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
from sklearn.model_selection import cross_val_score
print(cross_val_score(rfc, X_train, y_train, cv=5).mean())

from xgboost import XGBClassifier
xgbc = XGBClassifier()
print(cross_val_score(xgbc, X_train, y_train, cv=5).mean())


rfc.fit(X_train,y_train)
rfc_y_predict = rfc.predict(X_test)

rfc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': rfc_y_predict})
rfc_submission.to_csv('./Datasets/Titanic/rfc_submission.csv', index=False)


xgbc.fit(X_train, y_train)
xgbc_y_predict = xgbc.predict(X_test)
xgbc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_y_predict})
xgbc_submission.to_csv('./Datasets/Titanic/xgbc_submission.csv', index=False)

