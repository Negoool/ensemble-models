''' MNIST data set (ensemble models)'''
# load data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X_mnist = mnist['data']
y_mnist = mnist['target']
X_train, X_test = X_mnist[:60000], X_mnist[-10000:]
y_train,  y_test = y_mnist[:60000], y_mnist[-10000:]
np.random.seed(42)
shuffled_indices = np.random.permutation(len(X_train))
X_train = X_train[shuffled_indices]
y_train = y_train[shuffled_indices]

def plot_digit(data):
    image = data.reshape((28,28))
    plt.imshow(image, cmap = 'binary')
    plt.colorbar()
    plt.axis("off")
plot_digit(X_mnist[36000])


print (X_train[0,0]).nbytes

# def extenssion( X_train, add_right = 1, add_left = 1, add_down = 1, add_up =1):
#     X_train = X_train.reshape(len(X_train), 28,28)
#     X_extend = X_train[:]
#     if add_right:
#         right = np.c_[np.zeros((len(X_train),28,1)), X_train[:,:,:-1]]
#         X_extend = np.concatenate((X_extend, right), axis =2)
#     if add_left:
#         left = np.c_[X_train[:,:,1:], np.zeros((len(X_train),28,1))]
#         X_extend = np.concatenate((X_extend, left), axis =2)
#     if add_down:
#         down = np.append( np.zeros((len(X_train), 1,28)), X_train[:,:-1,], axis = 1)
#         X_extend = np.concatenate((X_extend, down), axis =2)
#     if add_up:
#         up = np.append(X_train[:,1:,], np.zeros((len(X_train), 1,28)), axis = 1)
#         X_extend = np.concatenate((X_extend, up), axis =2)
#     return X_extend
#
# X_extend = extenssion(X_train)
# X_extend = X_extend.reshape(-1,60000)
# print X_extend.shape

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pre_pipline = Pipeline([
('scaler', StandardScaler()),
])
scaler = StandardScaler()
X_train  = pre_pipline.fit_transform(X_train)
X_test = pre_pipline.transform(X_test)


from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(random_state = 42, bootstrap = True, \
 max_features = 15, n_estimators = 100) # after fine tuning
# rnd_clf.fit(X_train, y_train)
# #print rnd_clf.oob_score_
# print "RNF ACCURACY(train): ", accuracy_score(rnd_clf.predict(X_train), y_train)
# result = cross_val_score(rnd_clf, X_train, y_train, cv = 5, scoring = 'accuracy')
# print "Random Forest ACCURACY(validation): " , result.mean(), result.std()

## * comparison of decision tree and random forest
# from  sklearn.tree import DecisionTreeClassifier
# tree_clf = DecisionTreeClassifier(random_state = 42)
# tree_clf.fit(X_train, y_train)
# print "Tree ACCURACY: " , accuracy_score(tree_clf.predict(X_train), y_train)
# result = cross_val_score(tree_clf, X_train, y_train, cv = 5, scoring = 'accuracy')
# print "Tree ACCURACY(validation): " , result.mean(), result.std()

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

## getting the import features from random forest
# z =zip(np.argsort(rnd_clf.feature_importances_)[::-1][:200]/28 ,\
# np.argsort(rnd_clf.feature_importances_)[::-1][:200]%28)
# for i in z:
#     print i

from sklearn.ensemble import ExtraTreesClassifier
extra_clf = ExtraTreesClassifier(random_state = 42, n_estimators = 100)
# extra_clf.fit(X_train, y_train)
# print "extraTree ACCURACY: " , accuracy_score(extra_clf.predict(X_train), y_train)


# from sklearn.neighbors import KNeighborsClassifier # too slow
# knn_clf = KNeighborsClassifier()
# knn_clf.fit(X_train, y_train)
# print "knn ACCURACY: " , accuracy_score(knn_clf.predict(X_train), y_train)
# result = cross_val_score(knn_clf, X_train, y_train, cv = 5, scoring = 'accuracy')
# print "knn ACCURACY(validation): " , result.mean(), result.std()
#
# param_grid = [\
# {'n_neighbors': [5, 50,200], 'weights' : ['uniform', 'distance']}\
# ]
# knn_clf = KNeighborsClassifier()
# grid_search = GridSearchCV(knn_clf, param_grid, cv =5, scoring ='accuracy', verbose=3)
# grid_search.fit(X_train, y_train)
# z= zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'])
# for tup in z:
#     print tup
# print grid_search.best_params_

from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs',
 random_state = 42)
# log_clf.fit(X_train, y_train)
# print "Logistic Regression ACCURACY:", accuracy_score(log_clf.predict(X_train), y_train)
# result = cross_val_score(log_clf, X_train, y_train, cv = 5, scoring = 'accuracy')
# print "logistic ACCURACY(validation): " , result.mean(), result.std()

from sklearn.svm import SVC
svm_clf = SVC(random_state = 42, probability = True)
svm_clf.fit(X_train, y_train)
print "SVM ACCURACY:(train) ", accuracy_score(svm_clf.predict(X_train), y_train)


## analyzing error
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier([('rnd_clf', rnd_clf),('extra_clf', extra_clf),\
('log_clf', log_clf),('svm_clf', svm_clf)], voting = 'soft')
#
# for clf in (rnd_clf, extra_clf, log_clf, svm_clf, voting_clf):
#     pred = cross_val_predict(clf, X_train, y_train, cv =3)
#     print (clf.__class__.__name__ , accuracy_score(pred, y_train))
#     conf_matrix = confusion_matrix(pred, y_train)
#     row_sum = conf_matrix.sum(axis =1, keepdims = True )
#     conf_matrix_norm = conf_matrix.astype(np.float)/row_sum
#     np.fill_diagonal(conf_matrix_norm, 0)
#     plt.figure()
#     plt.matshow(conf_matrix_norm, cmap = 'gray')
#     plt.savefig(clf.__class__.__name__ + ".png", format='png', dpi=300)



plt.show()
