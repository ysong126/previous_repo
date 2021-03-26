# Credits to https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/  by Dr.Jason Brownlee

import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# load data
data_frame =pandas.read_csv("iris.csv", header=None)
dataset = data_frame.values
X=dataset[:,0:4].astype(float)
y=dataset[:,4]

# one hot encoding output
encoder = LabelEncoder()
encoder.fit(y)
encoded_y= encoder.transform(y)
dummy_y = np_utils.to_categorical(encoded_y)


# NNet model
def baseline_model():
    model=Sequential()
    model.add(Dense(12, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


clf = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=5, verbose=1)

# K Fold cross validation
kfold = KFold(n_splits=10,shuffle=True)
results = cross_val_score(clf, X, dummy_y, cv=kfold)
print("Baseline: {} {}".format(results.mean()*100, results.std()*100))