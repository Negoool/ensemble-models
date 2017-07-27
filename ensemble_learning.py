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
base_clf = DecisionTreeClassifier(random_state = 42, max_leaf_nodes = 16)
bagg_clf = BaggingClassifier( base_estimator = base_clf, n_estimators=500,
 max_samples=100, bootstrap=True, random_state = 42, n_jobs = 1, oob_score=True,
 max_features=1.0, bootstrap_features=False)
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

''' Random Forest'''
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=1, random_state=42)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)
print "random forest",accuracy_score(y_test, y_pred_rf) #.912

# using BaggingClassifier for randomness in feature selcetion to split on
base_clf2 = DecisionTreeClassifier(random_state = 42, splitter = 'random',
 max_leaf_nodes = 16)
bagg_clf2 = BaggingClassifier( base_estimator = base_clf2, n_estimators=500,
 max_samples=1., bootstrap=True, random_state = 42, n_jobs = 1, oob_score=True)

bagg_clf2.fit(X_train, y_train)
print accuracy_score(y_test, bagg_clf2.predict(X_test)) #.92, ~the same as up

# extra randomized trees ensemble
from sklearn.ensemble import ExtraTreesClassifier
extra_clf = ExtraTreesClassifier(n_estimators=500, max_leaf_nodes = 16,
random_state=42)
extra_clf.fit(X_train, y_train)
print "extra randomized trees ensemble",\
accuracy_score(y_test, extra_clf.predict(X_test))

''' boosting'''
## AdaBoost
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),
 n_estimators=400, learning_rate=.1, algorithm='SAMME.R', random_state=42 )
ada_clf.fit(X_train, y_train)
print "AdaBoost", accuracy_score(y_test, ada_clf.predict(X_test))

# choosing the best number of estimators
X_valid,y_valid = make_moons(n_samples=100, noise=.3, random_state = 42)
errors = [ accuracy_score(y_valid, y_pred) for y_pred in ada_clf.staged_predict(X_valid)]
plt.figure()
plt.plot(errors)
plt.xlabel("number of estimators of adaboost")
plt.ylabel("accuracy")
plt.title("choosing the best number of estimators")


from sklearn.ensemble import GradientBoostingClassifier
gbt_clf = GradientBoostingClassifier(n_estimators = 280, max_depth =1,
random_state = 42, learning_rate = .1)
gbt_clf.fit(X_train, y_train)
print "gradient boosting", accuracy_score(y_test, gbt_clf.predict(X_test))
plt.figure()
plt.subplot(131)
plot_decision_boundry(rnd_clf, X_train, y_train, axis = [-2,3,-1.5,2])
plt.title("random forest classifier")
plt.subplot(132)
plot_decision_boundry(ada_clf, X_train, y_train, axis = [-2,3,-1.5,2])
plt.title("ada boosted decision stump")
plt.subplot(133)
plot_decision_boundry(gbt_clf, X_train, y_train, axis = [-2,3,-1.5,2])
plt.title("gradient boosted trees")


## choosing the rights number of estimators
from sklearn.metrics import f1_score
gbt_clf = GradientBoostingClassifier( max_depth =2,
random_state = 42, learning_rate = .1, warm_start = True)
best_accuracy = 0.
counter = 0
a = []
for n_estimators in range(1,500):
    gbt_clf.n_estimators = n_estimators
    gbt_clf.fit(X_train, y_train)
    new_accuracy = accuracy_score(y_valid, gbt_clf.predict(X_valid))
    a.append(new_accuracy)
    if new_accuracy > best_accuracy:
        best_accuracy = new_accuracy
        counter = 0
    else:
        counter += counter
        if counter == 5:
            break

print best_accuracy
print n_estimators
plt.figure()
plt.plot(a)
plt.show()
