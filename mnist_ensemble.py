''' MNIST data set (ensemble models)'''
# load data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
import time
import os
os.system('clear')

# get the data
mnist = fetch_mldata('MNIST original')
X_mnist = mnist['data']
y_mnist = mnist['target']
# split ti test and train
X_train_val, x_test = X_mnist[:60000], X_mnist[-10000:]
y_train_val,  y_test = y_mnist[:60000], y_mnist[-10000:]
np.random.seed(42)
# shuffle train data
shuffled_indices = np.random.permutation(len(X_train_val))
# split the total train data to validation and train
x_train = X_train_val[shuffled_indices[:50000]]
y_train = y_train_val[shuffled_indices[:50000]]
x_valid = X_train_val[shuffled_indices[-10000:]]
y_valid = y_train_val[shuffled_indices[-10000:]]

def plot_digit(data):
    image = data.reshape((28,28))
    plt.imshow(image, cmap = 'binary')
    plt.colorbar()
    plt.axis("off")

#
class extenssion(BaseEstimator, TransformerMixin):
    ''' a class to shift image in any direction by one pixel \
    and add to training set'''

    def __init__(self, add_right = 1,  add_left = 1, add_down = 1, add_up =1):
        self.add_right = add_right
        self.add_left = add_left
        self.add_down = add_down
        self.add_up = add_up

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        X = X.reshape(len(X), 28,28)
        X_extend = X[:]
        if self.add_right:
            right = np.c_[np.zeros((len(X),28,1)), X[:,:,:-1]]
            X_extend = np.concatenate((X_extend, right), axis =2)
        if self.add_left:
            left = np.c_[X[:,:,1:], np.zeros((len(X),28,1))]
            X_extend = np.concatenate((X, left), axis =2)
        if self.add_down:
            down = np.append( np.zeros((len(X), 1,28)), X[:,:-1,], axis = 1)
            X_extend = np.concatenate((X_extend, down), axis =2)
        if self.add_up:
            up = np.append(X[:,1:,], np.zeros((len(X), 1,28)), axis = 1)
            X_extend = np.concatenate((X_extend, up), axis =2)
        return X_extend.reshape(len(X),-1)

pre_pipline = Pipeline([
('extend', extenssion()),
# ('scaler', StandardScaler())
])
X_train = pre_pipline.fit_transform(x_train)
print X_train.shape
X_valid = pre_pipline.transform(x_valid)
X_test = pre_pipline.transform(x_test)
# np.savez('pre_data', X_train, X_valid, X_test)

# a = np.load('pre_data.npz')
# X_train = a['arr_0']
# X_valid = a['arr_1']
# print X_train.shape
# print X_valid.shape


rnd_clf = RandomForestClassifier(n_estimators = 500,random_state = 42, max_features ='auto')

# ## getting the import features from random forest
# # z =zip(np.argsort(rnd_clf.feature_importances_)[::-1][:200]/28 ,\
# # np.argsort(rnd_clf.feature_importances_)[::-1][:200]%28)
# # for i in z:
# #     print i
#

## fine tune random forest
# results: (15, 100) or (5,300)
# param_grid = [\
# {'n_estimators': [10, 100,300], 'max_features' : [5, 10, 28]}
# # ]
# rnd_clf = RandomForestClassifier(random_state = 42)
# grid_search = GridSearchCV(rnd_clf, param_grid, cv =5, scoring ='accuracy', verbose=3)
# grid_search.fit(X_train, y_train)
# z= zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'])
# for tup in z:
#     print tup
# print grid_search.best_params_

extra_clf = ExtraTreesClassifier(random_state = 42, n_estimators = 500)


pca_knn = Pipeline([\
('PCA', PCA(random_state = 42, n_components =.95)),
('knn_clf', KNeighborsClassifier(n_jobs = -1, weights = 'distance', n_neighbors = 5))\
])


pca_svm = Pipeline([\
('scale', StandardScaler()),
('PCA', PCA(random_state = 42, n_components =.90)),
('svm_clf', SVC(random_state =42, gamma = .001, C= 5, probability=True))\
])

## analyzing error
voting_clf = VotingClassifier([('rnd_clf', rnd_clf),('extra_clf', extra_clf),\
('pca_knn', pca_knn),('svm_clf', pca_svm)], voting = 'soft')

for clf in (rnd_clf, extra_clf, pca_knn, pca_svm, voting_clf):

    clf.fit(X_train, y_train)
    pred = clf.predict(X_valid)
    print (clf.__class__.__name__ ,
    accuracy_score(clf.predict(X_train), y_train),
    accuracy_score(pred, y_valid))
    conf_matrix = confusion_matrix(pred, y_valid)
    row_sum = conf_matrix.sum(axis =1, keepdims = True )
    conf_matrix_norm = conf_matrix.astype(np.float)/row_sum
    np.fill_diagonal(conf_matrix_norm, 0)
    plt.figure()
    plt.matshow(conf_matrix_norm, cmap = 'gray')
    plt.savefig(clf.__class__.__name__ + ".png", format='png', dpi=300)
