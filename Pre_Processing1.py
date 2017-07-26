'''
Pre-Processing file.
'''

import sys
import datetime
import numpy as np
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.isnull().sum(axis=0)/train.shape[0]

#for letter in 'Python':     # First Example
#   print 'Current Letter :', letter

#if answer in ['y', 'Y', 'yes', 'Yes', 'YES']:
#    print("this will do the calculation")


train['devid'] = np.where(train['devid'].isnull() 
                        &(~train['browserid'].isnull())
                        &(train['browserid'].isin(['IE', 'Google Chrome', 'Firefox', 'Opera'])), 
                        'Mobile', train['devid'] )

train['devid'] = np.where(train['devid'].isnull() 
                        &(~train['browserid'].isnull())
                        &(train['browserid'].isin(['InternetExplorer', 'Mozilla Firefox', 'Mozilla', 'Chrome'])), 
                        'Desktop', train['devid'] )

train['devid'] = np.where(train['devid'].isnull() 
                        &(~train['browserid'].isnull())
                        &(train['browserid'].isin(['Internet Explorer', 'Safari', 'Edge'])), 
                        'Tablet', train['devid'] )


unique_id = list(set(train[train['devid'].isnull() & train['browserid'].isnull()]['offerid']))
unique_id = sorted(unique_id)

for i in range(0, train.shape[0]):
    if train.iloc[i]['offerid'].isin(unique_id):
        
print([train[train['offerid']==i]['devid'].mode() for i in unique_id])

        