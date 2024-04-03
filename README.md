# Machine-Learning-Classification-Methods

This paper discusses the use of two datasets from the Kaggle website for experimental study of Machine Learning algorithms. The first dataset is the Mobile Price Classification dataset, which contains 2,000 data points with 20 dimensions in 4 categories. The second dataset is the Airlines Delay dataset, which consists of 539,382 6-dimensional data points in 2 categories for binary classification.

For the Mobile Price Classification dataset, the dataset will be randomly split into two subsets - a training subset and a testing subset - in a 70-30 ratio. Note that the first attribute (flight code) of the Airlines Delay dataset should be disregarded, and the remaining attributes, which are discrete alphanumerics, should be assigned integer values.

The objective of this paper is to experimentally study the performance of well-known Machine Learning algorithms for classification. I had to implement the technique of 10-fold cross-validation to address overfitting and improve generalization of the classifiers. The performance evaluation will be based on accuracy (precision or success rate) and F1-score measures on the testing set provided.

Here we have four different classification methods to be studied. The first method is k-NN Nearest Neighbors, where different values of k (k=1, 3, 5, or 10) will be tried and Euclidean distance will be used for continuous variables and Hamming distance for discrete variables. The second method is Naïve Bayes classifier, assuming normal distribution for continuous features and multinomial distribution for discrete features. The third method is Neural Networks, with sigmoidal activation function in the hidden layers, either sigmoid or tanh, and with one or two hidden layers with variable numbers of neurons. The fourth method is Support Vector Machines, using linear kernel function and Gaussian kernel function (RBF) with various parameter values.

# Results

The optimal method for Mobile Price Classification is Support Vector Machines
(SVM) with a linear kernel function with 95% accuracy while the F1-score was for category 0: 96% for category 1: 94% category 2: 95% and for category 3: 96% .So, as we can observe the accuracy and F1-score the SVM is the best to be used in the 1st
The best method for Airlines delay is k-NN Nearest Neighbors with 61% and Neural Networks with 1 layer with 61.27% .The F1-score for k-NN Nearest Neighbors was for category 0: 64% for category 1: 58%. So, as we can As we can observe the accuracy of these 2 methods are the most ideal for to be used in the 2nd
dataset with the assumption that the measurements are not included 
for SVM so it is not possible to judge the performance of the algorithm. The measurements are necessary to be able to evaluate the accuracy, specificity and F1-score of the algorithm
