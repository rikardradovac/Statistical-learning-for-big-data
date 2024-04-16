# https://scikit-learn.org/stable/modules/cross_validation.html
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import auc

# Load data
X = pd.read_csv('data/TCGAdata.txt', sep=' ', header=0)
y = pd.read_csv('data/TCGAlabels', sep=' ', quotechar='"')

# Count the values and plot
# value_counts = y.value_counts()
# value_counts.plot(kind='bar')
#
# # Adding title and labels
# plt.title('Distribution of cancer type in dataset')    # Add a title to the graph
# plt.xlabel('Cancer type')         # Label for the x-axis
# plt.ylabel('Datapoints')         # Label for the y-axis
#
# # Show the plot
# plt.show()

scaler = MinMaxScaler()

# Fit and transform the data
scaled_data = scaler.fit_transform(X)

# Convert the scaled data back to a DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=X.columns)

variances = scaled_df.var()

# Sort the variances in descending order
X_sorted = variances.sort_values(ascending=False)[:1000]

# Create a bar plot of the variances
plt.figure(figsize=(10, 5))
plt.plot(X_sorted)
# plt.yscale('log')
plt.xlabel('Features')
plt.ylabel('Variance')
plt.title('Top 30 Features by Variance')
plt.xticks(rotation=90)
plt.grid()
plt.show()

all_variances = np.trapz(variances, range(len(variances)))
part = np.trapz(X_sorted, range(len(X_sorted)))

print(part/all_variances)

# Create and fit the KNN model
knn_small = KNeighborsClassifier(n_neighbors=2)
knn_big = KNeighborsClassifier(n_neighbors=20)

model = 1

for classifier in [knn_small, knn_big, model]:
    training_sizes = [0.8, 0.5, 0.2]
    for training_size in training_sizes:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-training_size), random_state=0, shuffle=True)

        scores = cross_val_score(classifier, X_train.values, np.ravel(y_train.values), cv=5)
        print("Cross-validation scores:", scores)


