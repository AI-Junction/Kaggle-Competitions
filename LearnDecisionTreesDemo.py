# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:56:15 2018

@author: echtpar
"""

''''
# DECISION TREES EXAMPLE IMPLEMENTATIONS

''''


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)


from sklearn.tree import export_graphviz
export_graphviz(
                tree_clf,
                out_file="iris_tree.dot",
                feature_names=iris.feature_names[2:],
                class_names=iris.target_names,
                rounded=True,
                filled=True
                )


tree_clf.predict_proba([[5, 1.5]])
#array([[ 0. , 0.90740741, 0.09259259]])
tree_clf.predict([[5, 1.5]])
#array([1])

# min_samples_split, min_samples_leaf, min_weight_fraction_leaf, and max_leaf_nodes


'''

GINI INDEX = 1 - Sigma (i = 1 to n) of (P_ik) ^ 2

P_ik is the ratio of class k instances among the training instances in the ith node

Decision Trees make very few assumptions about the training data (as opposed to linear models, 
which obviously assume that the data is linear, for example). If left unconstrained, 
the tree structure will adapt itself to the training data, fitting it very closely, 
and most likely overfitting it.


The DecisionTreeClassifier class has a few other parameters that similarly restrict the shape of 
the Decision Tree: 

    1.) min_samples_split = the minimum number of samples a node must have before it can be split

    2.) min_samples_leaf = the minimum number of samples a leaf node must have
    3.) min_weight_fraction_leaf = same as min_samples_leaf but expressed as a fraction of the total
number of weighted instances 
    4.) max_leaf_nodes = maximum number of leaf nodes
    5.) max_features = maximum number of features that are evaluated for splitting at each node
    
Increasing min_* hyperparameters or reducing max_* hyperparameters will regularize the model.


Loss function to be optimized in decision trees = J(k, tk) = m_left/m(Gini_left) + m_right/m(Gini_right)

'''
# Common imports
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# to make this notebook's output stable across runs
np.random.seed(42)

from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris-Virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)

plt.figure(figsize=(8, 4))
plot_decision_boundary(tree_clf, X, y)
plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)
plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)
plt.text(1.40, 1.0, "Depth=0", fontsize=15)
plt.text(3.2, 1.80, "Depth=1", fontsize=13)
plt.text(4.05, 0.5, "(Depth=2)", fontsize=11)

#save_fig("decision_tree_decision_boundaries_plot")
plt.show()



from sklearn.datasets import make_moons
Xm, ym = make_moons(n_samples=100, noise=0.25, random_state=53)

deep_tree_clf1 = DecisionTreeClassifier(random_state=42)
deep_tree_clf2 = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)
deep_tree_clf1.fit(Xm, ym)
deep_tree_clf2.fit(Xm, ym)

plt.figure(figsize=(11, 4))
plt.subplot(121)
plot_decision_boundary(deep_tree_clf1, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
plt.title("No restrictions", fontsize=16)
plt.subplot(122)
plot_decision_boundary(deep_tree_clf2, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
plt.title("min_samples_leaf = {}".format(deep_tree_clf2.min_samples_leaf), fontsize=14)

#save_fig("min_samples_leaf_plot")
plt.show()


'''

RANDOM FORESTS

'''


'''

Suppose you ask a complex question to thousands of random people, then aggregate their answers. 
In many cases you will find that this aggregated answer is better than an expert’s answer. 
This is called the wisdom of the crowd. Similarly, if you aggregate the predictions of a group 
of predictors (such as classifiers or regressors), you will often get better predictions than 
with the best individual predictor. A group of predictors is called an ensemble; thus, this 
technique is called Ensemble Learning, and an Ensemble Learning algorithm is called an 
Ensemble method.


For example, you can train a group of Decision Tree classifiers, each on a different 
random subset of the training set. To make predictions, you just obtain the predictions 
of all individual trees, then predict the class that gets the most votes. Such an ensemble of 
Decision Trees is called a Random Forest, and despite its simplicity, this is one of the 
most powerful Machine Learning algorithms available today.

In fact, the winning solutions in Machine Learning competitions often involve 
several Ensemble methods (most famously in the Netflix Prize competition).

Ensemble methods, including bagging, boosting, stacking, and a few others


Voting Classifiers
===================
Suppose you have trained a few classifiers, each one achieving about 80% accuracy. You may have a
Logistic Regression classifier, an SVM classifier, a Random Forest classifier, a K-Nearest Neighbors
classifier, and perhaps a few more

A very simple way to create an even better classifier is to aggregate the predictions of 
each classifier and predict the class that gets the most votes. This majority-vote classifier 
is called a hard voting classifier

If all classifiers are able to estimate class probabilities (i.e., they have a predict_proba() 
method), then you can tell Scikit-Learn to predict the class with the highest class probability, 
averaged over all the individual classifiers. This is called soft voting. It often achieves 
higher performance than hard voting because it gives more weight to highly confident votes. 
All you need to do is replace voting="hard" with voting="soft" and ensure that all classifiers 
can estimate class probabilities.

Somewhat surprisingly, this voting classifier often achieves a higher accuracy than the 
best classifier in the ensemble. In fact, even if each classifier is a weak learner (meaning it 
does only slightly better than random guessing), the ensemble can still be a strong learner 
(achieving high accuracy), provided there are a sufficient number of weak learners and they are 
sufficiently diverse.



Bagging and Pasting
===================
One way to get a diverse set of classifiers is to use very different training algorithms. Another 
approach is to use the same training algorithm for every predictor, but to train them on different
random subsets of the training set. When sampling is performed with replacement, this method is 
called bagging (short for bootstrap aggregating). When sampling is performed without replacement, 
it is called pasting. In other words, both bagging and pasting allow training instances to be 
sampled several times across multiple predictors, but only bagging allows training instances 
to be sampled several times for the same predictor.




'''


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()


from sklearn.datasets import make_moons
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
from sklearn.model_selection import train_test_split

print(type(X))
print(y[:5])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


voting_clf = VotingClassifier(
                estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
                voting='hard'
                )
voting_clf.fit(X_train, y_train)
#Let’s look at each classifier’s accuracy on the test set:
from sklearn.metrics import accuracy_score
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
#LogisticRegression 0.864
#RandomForestClassifier 0.872
#SVC 0.888
#VotingClassifier 0.896


log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(probability=True, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft')
voting_clf.fit(X_train, y_train)



from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))



# BAGGING


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# if we use plain decision tree then accuracy is far lower as below

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_tree))



from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    
    
plt.figure(figsize=(11,4))
plt.subplot(121)
plot_decision_boundary(tree_clf, X, y)
plt.title("Decision Tree", fontsize=14)
plt.subplot(122)
plot_decision_boundary(bag_clf, X, y)
plt.title("Decision Trees with Bagging", fontsize=14)
#save_fig("decision_tree_without_and_with_bagging_plot")
plt.show()    




#Random Forests


bag_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter="random", max_leaf_nodes=16, random_state=42),
    n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1, random_state=42)


bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)


from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)


np.sum(y_pred == y_pred_rf) / len(y_pred)


# Feature importance

from sklearn.datasets import load_iris
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
rnd_clf.fit(iris["data"], iris["target"])
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)
    
    
rnd_clf.feature_importances_    


plt.figure(figsize=(6, 4))

for i in range(15):
    tree_clf = DecisionTreeClassifier(max_leaf_nodes=16, random_state=42 + i)
    indices_with_replacement = np.random.randint(0, len(X_train), len(X_train))
    tree_clf.fit(X[indices_with_replacement], y[indices_with_replacement])
    plot_decision_boundary(tree_clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.02, contour=False)

plt.show()


# Out-of-Bag evaluation


bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    bootstrap=True, n_jobs=-1, oob_score=True, random_state=40)
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_

bag_clf.oob_decision_function_



from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)



# Feature importance


from sklearn.datasets import fetch_mldata
from keras.datasets import mnist

#mnist = fetch_mldata('MNIST original')
(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train = X_train.reshape(60000,-1)
X_test = X_test.reshape(10000,-1)



rnd_clf = RandomForestClassifier(random_state=42)
rnd_clf.fit(np.array(X_train), np.array(y_train))
rnd_clf.feature_importances_.shape

#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=None, max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#            oob_score=False, random_state=42, verbose=0, warm_start=False)

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.hot,
               interpolation="nearest")
    plt.axis("off")

plot_digit(rnd_clf.feature_importances_)

cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(), rnd_clf.feature_importances_.max()])
cbar.ax.set_yticklabels(['Not important', 'Very important'])

#save_fig("mnist_feature_importance_plot")
plt.show()




