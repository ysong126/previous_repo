import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import neighbors
from sklearn.metrics import accuracy_score


# load dataset
X, y = datasets.load_iris(return_X_y=True)

# split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# 1st approach decision tree
Tree_classifier = tree.DecisionTreeClassifier()

# 2nd approach KNN
KNN_classifier = neighbors.KNeighborsClassifier()


# training
Tree_classifier.fit(X_train, y_train)
KNN_classifier.fit(X_train, y_train)

# testing
tree_predict = Tree_classifier.predict(X_test)
knn_predict = KNN_classifier.predict(X_test)

# printing out the result, old style string format
print("The accuracy score of decision tree is %.2f" % accuracy_score(tree_predict, y_test))
print("The accuracy score of KNN is %.2f" % accuracy_score(knn_predict, y_test))


# plotting
markers = ('s', 'x', 'o')
colors = ('blue','red','purple')
color_map = ListedColormap(colors[:len(np.unique(y_test))])

for idx, color in enumerate(np.unique(y)):
    plt.scatter(x=X[y==color,0], y=X[y==color, 1], c=color_map(idx), marker=markers[idx], label=color)
plt.xlabel("pedal length")
plt.ylabel("pedal width")
plt.title("iris dataset")
plt.show()