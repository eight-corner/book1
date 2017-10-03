from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)

from sklearn.model_selection import train_test_split
X_train , X_test , y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25,random_state=33)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

from sklearn.svm import LinearSVC
lsvc = LinearSVC()
lsvc.fit(X_train, y_train)
y_predict = lsvc.predict(X_test)

score_lsvc = lsvc.score(X_test, y_test)
print(score_lsvc)

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_predict, target_names=digits.target_names.astype(str))
print(cr)


