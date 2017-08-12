'''
Merging predictions
'''

import sys
import datetime
import numpy as np
import pandas as pd

pred1 = pd.read_csv("Predictions\\cb_sub1.csv")
pred2 = pd.read_csv("Predictions\\prediction_26_depth7_rate2_iter100.csv")
pred3 = pd.read_csv("Predictions\\prediction_14_depth7.csv")
pred4 = pd.read_csv("Predictions\\pred_cat_2.csv")

pred_merge = pd.DataFrame()
pred_merge['ID']  = pred1['ID']
pred_merge['click'] = (pred2['click'] + pred4['click'])/2

pred_merge.to_csv("Predictions\\prediction_merge7.csv", index=False)