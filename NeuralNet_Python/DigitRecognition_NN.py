#! /usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
#import snips

X = pd.read_csv('HandWrittenDigit.csv')
y = pd.read_csv('Labels.csv')

classifier = MLPClassifier(solver = 'sgd')
classifier.hidden_layer_size = (40,)
classifier.activation = 'logistic'
classifier.fit(X,y)
plt.plot(classifier.loss_curve_)
plt.show()