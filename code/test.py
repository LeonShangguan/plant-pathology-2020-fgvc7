import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

X = np.array([[1, 2, 3, 4]
             , [11, 12, 13, 14]
             , [21, 22, 23, 24]
             , [31, 32, 33, 34]
             , [41, 42, 43, 44]
             , [51, 52, 53, 54]
             , [61, 62, 63, 64]
             , [21, 22, 23, 24]
             , [31, 32, 33, 34]
             , [41, 42, 43, 44]
             , [51, 52, 53, 54]
             , [61, 62, 63, 64]
             , [71, 72, 73, 74]])
y = np.array([1, 2, 0, 2, 1, 0, 0, 0, 3, 3, 2, 3, 3])

folder = KFold(n_splits=4, random_state=0, shuffle=True)
sfolder = StratifiedKFold(n_splits=4, random_state=0, shuffle=True)
for train, test in sfolder.split(y, y):
    print((train, test))
print('----------')
for train, test in folder.split(y, y):
    print(train, test)