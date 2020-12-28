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