
# linear regression example from sklearn
# data set: Boston Housing


import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, r2_score

# load data
# return_X_y option separates feature/target
boston_X_full, boston_Y = datasets.load_boston(return_X_y=True)

# one feature linear regression medv = k + b*crim
# newaxis increases the dimension
#boston_X = boston_X_full[:,np.newaxis,0]

# or, equivalently reshape the data
boston_X = boston_X_full[:, 0]
boston_X = boston_X[:, np.newaxis]

# split into train /test
boston_X_train = boston_X[:-50]
boston_X_test = boston_X[-50:]

# target
boston_Y_train = boston_Y[:-50]
boston_Y_test = boston_Y[-50:]

# object reg stores the model
# fit/test of linear regression takes 2-D arrays for both X and y
# e.g.
# X,y = [[1.5],[2],[2.5]] , [[2],[3],[4]]


reg = linear_model.LinearRegression()
reg.fit(boston_X_train, boston_Y_train)

# predict on test dataset
boston_Y_pred = reg.predict(boston_X_test)

# coef
print('Coef:\n', reg.coef_)

# mse
print('mean squared error: %.2f' % mean_squared_error(boston_Y_test, boston_Y_pred))

# determination
print('coef R squared %.2f' % r2_score(boston_Y_test, boston_Y_pred))

# plot
plt.figure()
plt.scatter(boston_X_test, boston_Y_test,color='black')
plt.plot(boston_X_test, boston_Y_pred, color='blue', linewidth=3)
plt.xlabel('crime rate %')
plt.ylabel('median value of housing prices')
plt.show()
