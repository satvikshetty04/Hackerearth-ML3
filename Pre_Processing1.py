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
print("Started at: " + start_time)


# --------------------- Converters


def date_conv(x):
    return x

# --------------------- Loading dataset

train = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 3\\train.csv",
                    header=0,
                    converters={'datetime':date_conv})


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
