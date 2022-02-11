# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 09:16:24 2022

@author: Feng
"""

import numpy as np
import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.metrics as sm
import sklearn.ensemble as se
import matplotlib.pylab as mp
import timeit
start=timeit.default_timer()

boston=sd.load_boston()

x,y=su.shuffle(boston.data,boston.target, random_state=7)

train_size=int(len(x)*0.8) #train_size是一个分割点
train_x,test_x,train_y,test_y = \
    x[:train_size],x[train_size:],\
    y[:train_size],y[train_size:]

#构建随机森林横型,最大树深度。树数量,子表最小样本数
model=se.RandomForestRegressor(max_depth=10, n_estimators=20000, min_samples_split=2)

model.fit(train_x,train_y)
pred_test_y=model.predict(test_x)

#评估预测结果
rr=sm.r2_score(test_y, pred_test_y)
print(rr)
end=timeit.default_timer()
print(end-start)