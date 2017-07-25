'''
Pre-Processing file.
'''

import sys
import datetime
import numpy as np
import pandas as pd
import os

start_time = datetime.datetime.now()
print("Started at: " + start_time)

# --------------------- Converters


# --------------------- Loading dataset

train = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 3\\train.csv",
                    header=0,
                    converters={'datetime':date_conv})


