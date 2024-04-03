from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

df = pd.read_csv('train.csv')  # reading our db   

training_data, testing_data = train_test_split(df, test_size=0.3, random_state=25)  #training and testing subsets

# scale our subsets
def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)
    data = np.hstack((X, np.reshape(y, (-1, 1))))
    
    return data, X, y

#scale our subsets
train, X_train, y_train = scale_dataset(training_data, oversample=True)
test, X_test, y_test = scale_dataset(testing_data, oversample=False)

#define your cross-validation
kfold = KFold(n_splits=10)


#KNN neighbors
KNNneighbors = input("Enter the number of neighbors : ")
print("\n     -------------KNN Neighbors METHOD------------")
knn_model = KNeighborsClassifier(n_neighbors = int(KNNneighbors))
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
print(classification_report(y_test, y_pred))


# Evaluate model performance
results = cross_val_score(knn_model, X_train , y_train , cv = kfold)
print("Here is the evaluation of the model using 10-fold cross validation: " + str(results))
print("\n\n\n\n")


# Naïve Bayes classifier
print("         ------------Naive Bayes METHOD------------\n")
nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)
print(classification_report(y_test, y_pred))
print('\n\n')


# Evaluate model performance
results = cross_val_score(nb_model, X_train, y_train, cv=kfold)
print("Here is the evaluation of the model using 10-fold cross validation: " + str(results))
print("\n\n\n\n")


#SVM
print("      --------------SVM METHOD------------\n")
svc = SVC(kernel = 'linear')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(classification_report(y_test, y_pred))


# Evaluate model performance
results = cross_val_score(svc, X_train, y_train, cv=kfold)
print("Here is the evaluation of the model using 10-fold cross validation: " + str(results))
print("\n\n\n\n")



#SVM GAUSSIAN KERNEL
print("\n --------------Gaussian Kernel SVM-----------")
svc_gaussian = SVC(kernel='rbf', degree=1)
svc_gaussian.fit(X_train, y_train)
y_pred_gaussian = svc_gaussian.predict(X_test)
print(classification_report(y_test, y_pred_gaussian))


# Evaluate model performance
results = cross_val_score(svc_gaussian, X_train, y_train, cv=kfold)
print("Here is the evaluation of the model using 10-fold cross validation: " + str(results))
print("\n\n\n\n")


#NN with 1 layer and K neurons
Kneurons = input("Give the number of neurons that the hidden layer should have: ")
nn_model1 = keras.Sequential([keras.layers.Dense(256, input_shape=(X_train.shape[1],)),  # input layer (1)
    keras.layers.Dense(int(Kneurons), activation='sigmoid'),  # hidden layer (2)
    keras.layers.Dense(int(Kneurons), activation='softmax')  # output layer (3)
])
nn_model1.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# we pass the data, labels and epochs 
nn_model1.fit(X_train, y_train, epochs=10)

test_loss, test_acc = nn_model1.evaluate(X_test,  y_test , verbose = 1)
# Print the test loss and accuracy
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


# NN with 2 layers and K1,K2 neurons
Kneurons1 = input("\nGive the number of neurons for hidden layer 1: ")
Kneurons2 = input("\nGive the number of neurons for hidden layer 2: ")
nn_model2 = keras.Sequential([keras.layers.Dense(256, input_shape=(X_train.shape[1],)),  # input layer (1)
    keras.layers.Dense(int(Kneurons1), activation='sigmoid'),  # hidden layer (1)
    keras.layers.Dense(int(Kneurons2), activation='sigmoid'),  # hidden layer (2)
    keras.layers.Dense(int(Kneurons), activation='softmax')  # output layer (3)
])
nn_model2.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# we pass the data, labels and epochs 
nn_model2.fit(X_train, y_train, epochs=10)

test_loss, test_acc = nn_model2.evaluate(X_test,  y_test)
# Print the test loss and accuracy
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


