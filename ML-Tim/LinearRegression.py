
## Import packages

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style



data = pd.read_csv('student-mat.csv', sep = ";")

print(data.head())

# These are some of the important attributes from the data
data = data[["G1","G2","G3","studytime","failures", "absences"]]

# We want to determine attribute G3 also called label
predict = "G3"

X = np.array(data.drop([predict], 1))

y = np.array(data[predict])


# Split the data into test train (here we try to get best score for training)

best = 0

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)


'''for _ in range(30):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)

    print(acc)

# Saving model using pickle
if acc > best:
    best = acc
    with open("studentmodel.pickle", "wb") as f:
        pickle.dump(linear, f) '''

# opening the model saved using pickle(one can comment the actual training and fitting part)
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# printing different coeffecients for the line (m & b)
# Here for coefficients we get 5 values of m in 5 dimensional space.
print("Co: ", linear.coef_)       
print('Intercept: \n', linear.intercept_)


# Prediction
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], "  ", x_test[x], "  ", y_test[x])


# Plotting

p = "absences"
style.use("ggplot")
plt.scatter(data[p], data['G3'])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()
