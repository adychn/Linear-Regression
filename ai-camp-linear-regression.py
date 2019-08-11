#!/usr/bin/env python
# coding: utf-8

# # Linear Regression预测房价
# In[ ]:
import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from sklearn.linear_model import LinearRegression # sk-learn库Linear Regression模型
from sklearn.model_selection import train_test_split # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
import math #数学库

# 从../input/kc_house_data.csv文件中读入数据
# In[ ]:
data = pd.read_csv("kc_house_data.csv")

# In[ ]:
data.dtypes

# 获得自变量X和因变量Y
# In[ ]:
X = data[['bedrooms','bathrooms','sqft_living','floors', 'condition', 'sqft_lot', 'yr_built']]
Y = data['price']
Y.shape
# 获得2:1的训练：测试数据比例
# In[ ]:
xtrain, xtest, ytrain, ytest= train_test_split(X, Y, test_size=1/3, random_state=0)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

# In[ ]:
xtrain = np.asmatrix(xtrain)
xtest = np.asmatrix(xtest) # as matrix will put it as a row if one dimension
ytrain = np.ravel(ytrain) # ravel will put it as a column.
ytest = np.ravel(ytest)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

# In[ ]:
# 观察房价和生活面积的关系
plt.scatter(X['sqft_living'], Y)
plt.xlabel('sqft_living')
plt.ylabel('price')


# In[ ]:
# 观察生活面积分布
X['sqft_living'].hist()
plt.xlabel('sqft_living')
plt.ylabel('count')



# In[ ]:
# 用xtrain和ytrain训练模型
model = LinearRegression()
model.fit(xtrain, ytrain)

# In[ ]:
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))

# In[ ]:
model.intercept_

# In[ ]:
#一个房子，3个卧室，2个卫生间，2500sqft，2层楼，预测房价
model.predict([[3, 2, 2500, 2]])

# In[ ]:
pred = model.predict(xtrain)

# 训练集上的均方差MSE
mse = ((pred - ytrain) * (pred - ytrain)).sum() / len(ytrain)
print(mse)
print(metrics.mean_squared_error(pred, ytrain))

# 平均相对误差
me = abs(pred - ytrain).sum() / len(ytrain)
print(me)
