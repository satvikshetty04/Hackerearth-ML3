'''
Pre-Processing file.
'''

import sys
import datetime
import numpy as np
import pandas as pd
import os


# --------------------- Initializing variables

start_time = datetime.datetime.now()
print("Started at: " + str(start_time))


# --------------------- Loading dataset

train = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 3\\train.csv",
                    header=0)

test = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 3\\test.csv",
                    header=0)

# --------------------- Creating day, hour features

train['day'] = pd.to_datetime(train['datetime']).dt.day
train['hour'] = pd.to_datetime(train['datetime']).dt.hour
test['day'] = pd.to_datetime(test['datetime']).dt.day
test['hour'] = pd.to_datetime(test['datetime']).dt.hour


print(train)
print(test)


# train = pd.read_csv("train.csv")
# test = pd.read_csv("test.csv")

print("Time taken: "+ str(datetime.datetime.now() - start_time))