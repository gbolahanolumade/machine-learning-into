# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 23:10:45 2021

@author: GbolahanOlumade
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_df=pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df.info()
test_df.info()

test_df['Survived']= -55

df = pd.concat((train_df,test_df))