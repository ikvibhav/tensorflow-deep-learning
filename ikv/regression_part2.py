import matplotlib
matplotlib.use('Qt5Agg')

#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

X = np.arange(-100, 100, 4)
y = np.arange(-90, 110, 4)

X_train = X[:40]
y_train = y[:40]

X_test = X[40:]
y_test = y[40:]

plt.figure(figsize=(10, 7))
plt.scatter(X_train, y_train, c='b', label='Training data')
plt.scatter(X_test, y_test, c='g', label='Testing data')
plt.legend()
plt.show()