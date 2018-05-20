import numpy as np
#import tensorflow as tf
#import matplotlib.pyplot as plt
import data_processing
#import random
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
data =data_processing.load_data(download = False)
new_data =data_processing.convert2onehot(data)

clf = RandomForestClassifier(random_state=14)
x0 = new_data[:,:21]
y0 = new_data[:,21:]
scores=cross_validation.cross_val_score(clf, x0 , y0 , scoring='accuarcy')
print('the accuracy is {0:.lf}%'.format(np.mean(scores)*100))








