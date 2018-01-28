import pandas as pd
#import sklearn
import numpy
#import sys

cars = pd.read_fwf('car.data.txt', sep=" ", header=None, names=['a'])

cSort = cars.sort(columns=1, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last',)
print(cSort)