# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 19:57:50 2020

@author: GbolahanOlumade
"""

from sklearn import datasets
iris = datasets.load_iris()

iris.data
iris.target

iris.target_names
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import datasets

iris = datasets.load_iris()
x = iris.data[:,0]  #X-Axis - sepal length
y = iris.data[:,1]  #Y-Axis - sepal length
species = iris.target #Speciss

x_min, x_max = x.min() - .5,x.max() + .5
y_min, y_max = y.min() - .5,y.max() + .5


#SCATTERPLOT
plt.figure()
plt.title('Iris Dataset - Cklassification by Sepal Sizes')
plt.scatter(x,y, c=species)
plt.xlabel('Sepal length')
plt.ylabel('Special width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

df = pd.DataFrame({'A': ['high','medium','los'], 'B':[10,20,30]},
                  index=[0,1,2])

df_with_dummies = pd.get_dummies(df, prefix='A', columns=['A'])

from sklearn import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np


iris = datasets.load_iris()
iris.data
X = iris.data[:, [2,3]]
y = iris.target

scale = StandardScaler()

X_scale=scale.fit_transform(X)

scale2 = MinMaxScaler()
Min_scale = scale2.fit_transform(X)


iris = datasets.load_iris()

iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
 columns= iris['feature_names'] + ['species'])

iris.species = np.where(iris.species == 0.0, 'setosa', np.where(iris.
species==1.0,'versicolor', 'virginica'))

iris.columns = iris.columns.str.replace(' ','')
iris.describe()


iris.hist()

plt.subplots
plt.show()

iris.boxplot()
plt.show()


iris.groupby(by = "species").mean()
# plot for mean of each feature for each label class
iris.groupby(by = "species").mean().plot(kind="bar")
plt.title('Class vs Measurements')
plt.ylabel('mean measurement(cm)')
plt.xticks(rotation=0) # manage the xticks rotation
plt.grid(True)
# Use bbox_to_anchor option to place the legend outside plot area to be tidy
plt.legend(loc="upper left", bbox_to_anchor=(1,1))


corr = iris.corr()

import statsmodels.api as sm
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()
print (corr)