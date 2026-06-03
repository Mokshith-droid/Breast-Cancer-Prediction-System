import numpy as np
import pandas as pd
import sklearn.datasets
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
dataset = sklearn.datasets.load_breast_cancer()

X = dataset.data
Y = dataset.target

# Split Data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# Standardize
scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Neural Network
model = MLPClassifier(
    hidden_layer_sizes=(20,),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=3
)

# Train
model.fit(X_train_std, Y_train)

# Accuracy
predictions = model.predict(X_test_std)

accuracy = accuracy_score(Y_test, predictions)

print("Accuracy:", accuracy)

# Save Model
with open("breast_cancer_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save Scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model Saved Successfully")