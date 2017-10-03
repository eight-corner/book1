
# 从sklearn.datasets导入波士顿房价数据读取器。
from sklearn.datasets import load_boston
# 从读取房价数据存储在变量boston中。
boston = load_boston()
# 输出数据描述。
# 从sklearn.cross_validation导入数据分割器。
from sklearn.model_selection import train_test_split

# 导入numpy并重命名为np。
import numpy as np

X = boston.data
y = boston.target

# 随机采样25%的数据构建测试样本，其余作为训练样本。
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.25)


# 从sklearn.tree中导入DecisionTreeRegressor。
from sklearn.tree import DecisionTreeRegressor
# 使用默认配置初始化DecisionTreeRegressor。
dtr = DecisionTreeRegressor()
# 用波士顿房价的训练数据构建回归树。
dtr.fit(X_train, y_train)
# 使用默认配置的单一回归树对测试数据进行预测，并将预测值存储在变量dtr_y_predict中。
dtr_y_predict = dtr.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error
# 使用R-squared、MSE以及MAE指标对默认配置的回归树在测试集上进行性能评估。
print('R-squared value of DecisionTreeRegressor:', dtr.score(X_test, y_test))
print('The mean squared error of DecisionTreeRegressor:', mean_squared_error(y_test, dtr_y_predict))
print('The mean absoluate error of DecisionTreeRegressor:', mean_absolute_error(y_test, dtr_y_predict))

