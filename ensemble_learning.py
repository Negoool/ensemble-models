''' different ensembel learning algorithms on moons data sets'''
import numpy as np
import matplotlib.pyplot as plt
import os
os.system('cls')
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
X,y = make_moons(n_samples=500, noise=.3, random_state=42)
X_train, X_test, y_train,y_test = train_test_split(X, y, random_state = 42)

# plt.plot(X_train[y_train  == 0,0], X_train[y_train==0,1],'sb')
# plt.plot(X_train[y_train  == 1,0], X_train[y_train==1,1],'og')
# plt.show()

''' Voting classifires'''
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
clf1 = RandomForestClassifier(random_state = 42)
clf2 = LogisticRegression()
clf3 = SVC(random_state = 42, probability=False)
voting_clf = VotingClassifier(estimators = [('rf', clf1), ('lr', clf2),
 ('svm', clf3)], voting = 'hard')
from sklearn.metrics import accuracy_score
for clf in (clf1, clf2, clf3, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print (clf.__class__.__name__, accuracy_score(y_test, y_pred))

'''Bagging and Pasting ensembles'''
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
base_clf = DecisionTreeClassifier(random_state = 42)
bagg_clf = BaggingClassifier( base_estimator = base_clf, n_estimators=500,
 max_samples=100, bootstrap=True, random_state = 42, n_jobs = 1, oob_score=True)
bagg_clf.fit(X_train, y_train)
y_pred = bagg_clf.predict(X_test)
print "\naccuracy on test data using bagging",accuracy_score(y_test, y_pred)
print "oob score", bagg_clf.oob_score_

single_tree_clf = DecisionTreeClassifier(random_state = 42)
single_tree_clf.fit(X_train, y_train)
y_pred_tree = single_tree_clf.predict(X_test)
print "accracy on test data using a single predictor", accuracy_score(y_test, y_pred_tree)

from matplotlib.colors import ListedColormap
def plot_decision_boundry(clf, X, y, axis = [-2,3,-1.5,2]):
    x1 = np.linspace(axis[0], axis[1], 100)
    x2 = np.linspace(axis[2], axis[3], 100)
    xv, yv = np.meshgrid(x1,x2)
    X_new = np.c_[xv.ravel(), yv.ravel()]
    z = clf.predict(X_new)
    zz = z.reshape(xv.shape)
    plt.contourf(xv, yv, zz, alpha =.3)
    plt.plot(X[y==0 , 0], X[y==0 , 1], 'sg', alpha =.5)
    plt.plot(X[y==1 , 0], X[y==1 , 1], 'ob', alpha = .5)

plt.figure(figsize=(11,4))
plt.subplot(121)
plot_decision_boundry(single_tree_clf, X_train, y_train, axis = [-2,3,-1.5,2])
plt.title("single decision tree")
plt.subplot(122)
plot_decision_boundry(bagg_clf, X_train, y_train, axis = [-2,3,-1.5,2])
plt.title("Desicion tree with Bagging")
plt.show()
