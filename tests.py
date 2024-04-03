from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
features, target = load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    df, test_size=0.3, random_state=25
)


class kNNClassifier:
    '''
    Description:
        This class contains the functions to calculate distances
    '''

    def __init__(self, k=3, distanceMetric='euclidean'):
        '''
        Description:
            KNearestNeighbors constructor
        Input    
            k: total of neighbors. Defaulted to 3
            distanceMetric: type of distance metric to be used. Defaulted to euclidean distance.
        '''
        pass

    def fit(self, xTrain, yTrain):
        '''
        Description:
            Train kNN model with x data
        Input:
            xTrain: training data with coordinates
            yTrain: labels of training data set
        Output:
            None
        '''
        assert len(xTrain) == len(yTrain)
        self.trainData = xTrain
        self.trainLabels = yTrain

    def getNeighbors(self, testRow):
        '''
        Description:
            Train kNN model with x data
        Input:
            testRow: testing data with coordinates
        Output:
            k-nearest neighbors to the test data
        '''

        calcDM = distanceMetrics()
        distances = []
        for i, trainRow in enumerate(self.trainData):
            if self.distanceMetric == 'euclidean':
                distances.append([trainRow, calcDM.euclideanDistance(
                    testRow, trainRow), self.trainLabels[i]])
            elif self.distanceMetric == 'manhattan':
                distances.append([trainRow, calcDM.manhattanDistance(
                    testRow, trainRow), self.trainLabels[i]])
            elif self.distanceMetric == 'hamming':
                distances.append([trainRow, calcDM.hammingDistance(
                    testRow, trainRow), self.trainLabels[i]])
            distances.sort(key=operator.itemgetter(1))

        neighbors = []
        for index in range(self.k):
            neighbors.append(distances[index])
        return neighbors

    def predict(self, xTest, k, distanceMetric):
        '''
        Description:
            Apply kNN model on test data
        Input:
            xTest: testing data with coordinates
            k: number of neighbors
            distanceMetric: technique to calculate distance metric
        Output:
            predicted label 
        '''
        self.testData = xTest
        self.k = k
        self.distanceMetric = distanceMetric
        predictions = []

        for i, testCase in enumerate(self.testData):
            neighbors = self.getNeighbors(testCase)
            output = [row[-1] for row in neighbors]
            prediction = max(set(output), key=output.count)
            predictions.append(prediction)

        return predictions




# # Scale the data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Apply dimensionality reduction
# pca = PCA(n_components=2)
# X_train_dim_red = pca.fit_transform(X_train_scaled)
# X_test_dim_red = pca.transform(X_test_scaled)

# # Visualise results
# fig, ax = plt.subplots(figsize=(10, 7))
# for label, color in zip(set(y_train), ('orange', 'blue', 'brown')):
#     ax.scatter(
#         X_train_dim_red[y_train == label, 0],
#         X_train_dim_red[y_train == label, 1],
#         color=color, label=f'Class {label}'
#     )

# ax.set_title('Dataset after Principal Component Analysis ')
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.legend(loc='upper right')

# # Train and evaluate a model
# model = GaussianNB()
# model.fit(X_train_dim_red, y_train)
# predictions = model.predict(X_test_dim_red)
# print(f'Model Accuracy: {accuracy_score(y_test, predictions):.2f}')


# plt.show()
