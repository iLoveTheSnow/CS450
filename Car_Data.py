import pandas as pd
#import sklearn
import numpy
#import sys



data = pd.read_fwf('auto-mpg.data.txt', sep=" ", header=None, names=["MPG", "b", "c", "d", "e", "f", "g", "h", "i"])
data.drop(['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], axis=1, inplace=True)
mean = data['MPG'].mean()

print(data)
print("The mean value if MPG is =", mean)

target = pd.read_fwf('auto-mpg.data-original.txt', sep=" ", header=None, names=["MPG", "b", "c", "d", "e", "f", "g", "h", "i"])
target.drop(['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], axis=1, inplace=True)

print("this is the target\n", target)
tMean = target['MPG'].mean()
print("The mean value of Target is =", tMean)

