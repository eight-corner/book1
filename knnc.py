from sklearn.datasets import load_iris
iris = load_iris()
print(iris.data.shape)
print(iris.DESCR)

from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(iris.data, iris.target,test_size=0.25, random_state=33)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test  = ss.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_predict = knc.predict(X_test)

print(knc.score(X_test, y_test))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict, target_names=iris.target_names))

# 样本遍历，逐一比较，内存占用大



