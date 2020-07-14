
# Import packages

import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


# Loading data

cancer = datasets.load_breast_cancer()

print(cancer.feature_names)
print(cancer.target_names)


# Labeling x and y and split data into train and test sets

x = cancer.data
y = cancer.target 

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2) 

print(x_train, y_test)

classes = ['maglignant', 'benign']


# Support Vector Classifier

clf = svm.SVC(kernel = "linear", C = 2)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc  = metrics.accuracy_score(y_test, y_pred)

print("SVM accuracy : ", acc)


# Comparing with KNN

clf2 = KNeighborsClassifier(n_neighbors = 3)

clf2.fit(x_train, y_train)

y_pred = clf2.predict(x_test)

acc2  = metrics.accuracy_score(y_test, y_pred)

print("KNN accuracy : ", acc2)