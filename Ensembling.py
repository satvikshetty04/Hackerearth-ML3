'''
Merging predictions
'''

import sys
import datetime
import numpy as np
import pandas as pd

pred1 = pd.read_csv("Predictions\\cb_sub1.csv")
pred2 = pd.read_csv("Predictions\\prediction_13.csv")
pred3 = pd.read_csv("Predictions\\prediction_11.csv")
pred4 = pd.read_csv("Predictions\\prediction_12.csv")

pred_merge = pd.DataFrame()
pred_merge['ID']  = pred1['ID']
pred_merge['click'] = (pred1['click'] + pred2['click'] + pred3['click'] + pred4['click'])/4

pred_merge.to_csv("Predictions\\prediction_merge3.csv", index=False)