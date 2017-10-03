# 导入pandas与numpy工具包。
import pandas as pd
import numpy as np

# 创建特征列表。
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

# 使用pandas.read_csv函数从互联网读取指定数据。
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names = column_names )
# 将?替换为标准缺失值表示。
data = data.replace(to_replace='?', value=np.nan)
# 丢弃带有缺失值的数据（只要有一个维度有缺失）。
data = data.dropna(how='any')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size= 0.25, random_state=33)
# 样本分布
print(y_train.value_counts())
print(y_test.value_counts())

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

from sklearn.linear_model import LogisticRegression
lr  = LogisticRegression()
lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)

from sklearn.linear_model import SGDClassifier
sgdc = SGDClassifier()
sgdc.fit(X_train, y_train)
sgdc_y_predict = sgdc.predict(X_test)


from sklearn.metrics import classification_report
print(classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))
print(classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))

# 线性分类器：最基本的学习模型。
# lr对参数的计算采用精确的解析解，计算时间长，模型略高
# 10W以上数据规模，考虑时间的耗用，更加推荐使用随机梯度算法对模型参数进行估计