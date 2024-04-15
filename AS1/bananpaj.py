# https://scikit-learn.org/stable/modules/cross_validation.html
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load data
X = pd.read_csv('data/TCGAdata.txt', sep=' ', header=0)
y = pd.read_csv('data/TCGAlabels', sep=' ', quotechar='"')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

# Create and fit the KNN model
knn = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(knn, X_train.values, np.ravel(y_train.values), cv=5)
print("Cross-validation scores:", scores)


