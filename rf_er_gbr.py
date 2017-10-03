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
# 从sklearn.ensemble中导入RandomForestRegressor、ExtraTreesGressor以及GradientBoostingRegressor。
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

# 使用RandomForestRegressor训练模型，并对测试数据做出预测，结果存储在变量rfr_y_predict中。
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_y_predict = rfr.predict(X_test)

# 使用ExtraTreesRegressor训练模型，并对测试数据做出预测，结果存储在变量etr_y_predict中。
'''
极端随机森林 于普通随机森林不同：
在每当构建一棵树的分裂节点的时候，不会任意地选取特征，
而是先随机收集一部分特征，然后利用信息熵和基尼不纯度等指标挑选最佳的节点特征

'''
etr = ExtraTreesRegressor()
etr.fit(X_train, y_train)
etr_y_predict = etr.predict(X_test)

# 使用GradientBoostingRegressor训练模型，并对测试数据做出预测，结果存储在变量gbr_y_predict中。
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
gbr_y_predict = gbr.predict(X_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error
# 使用R-squared、MSE以及MAE指标对默认配置的随机回归森林在测试集上进行性能评估。
print('R-squared value of RandomForestRegressor:', rfr.score(X_test, y_test))
print( 'The mean squared error of RandomForestRegressor:', mean_squared_error(y_test, rfr_y_predict))
print( 'The mean absoluate error of RandomForestRegressor:', mean_absolute_error(y_test, rfr_y_predict))


# 使用R-squared、MSE以及MAE指标对默认配置的极端回归森林在测试集上进行性能评估。
print('R-squared value of ExtraTreesRegessor:', etr.score(X_test, y_test))
print('The mean squared error of  ExtraTreesRegessor:', mean_squared_error(y_test,etr_y_predict))
print('The mean absoluate error of ExtraTreesRegessor:', mean_absolute_error(y_test, etr_y_predict))

# 利用训练好的极端回归森林模型，输出每种特征对预测目标的贡献度。
print(zip(etr.feature_importances_, boston.feature_names))
featrue_importance = zip(etr.feature_importances_, boston.feature_names)
print(np.sort(list(featrue_importance), axis= 0))
# 使用R-squared、MSE以及MAE指标对默认配置的梯度提升回归树在测试集上进行性能评估。
print('R-squared value of GradientBoostingRegressor:', gbr.score(X_test, y_test))
print('The mean squared error of GradientBoostingRegressor:', mean_squared_error(y_test, gbr_y_predict))
print('The mean absoluate error of GradientBoostingRegressor:', mean_absolute_error(y_test, gbr_y_predict))


# 许多业界从事商业分析系统开发和搭建的工作者更加青睐于集成模型，
#并经常以这些模型的性能表现为基准，与新设计的其他模型性能进行比对。