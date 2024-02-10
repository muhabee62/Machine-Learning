# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 19:16:43 2023

@author: muhab
"""
#REMOVE WARNINGS REGARDING CPU/GPU USAGE:
from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np # For numerical operations and aliases it as np.
import pandas as pd # For data manipulation and analysis and aliases it as pd.
from sklearn.model_selection import train_test_split # For splitting data into training and testing sets.
from tensorflow.keras import Sequential # A linear stack of layers for building neural network models.
from tensorflow.keras.layers import Dense # A standard fully connected neural network layer in Keras.
import matplotlib.pyplot as plt # plotting and aliases it as plt.

from tensorflow.random import set_seed
from random import seed

# For repeatable results:
SEED = 3
seed(SEED)
np.random.seed(SEED)
set_seed(SEED)

# Loading the dataset
df = pd.read_csv('flower_data.csv')
print(df.head(), '\n')
print(df.shape, '\n')

# Splitting the dataset into input and output columns
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print(X.shape, '\n')
print(y.shape, '\n')

# Spliting the input and output columns into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print(X_train.shape, '\n')
print(y_train.shape, '\n')
print(X_test.shape, '\n')
print(y_test.shape, '\n')

# Determining the number of input features
n_features = X.shape[1]
print(n_features, '\n')

# Building the model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(n_features,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

model.summary()

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model and collecting the training history
history = model.fit(X_train, y_train, epochs=56, batch_size=35, validation_data=(X_test, y_test))

# Evaluating the model 
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Making predictions
sample_input = np.random.randn(1, n_features)
prediction = model.predict(sample_input)
print(f'Prediction: {prediction[0, 0]}')
print(sample_input, prediction)

# Plotting learning curves
plt.figure(figsize=(8, 6))

# Plotting training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.show()

# Plotting learning cdurves
plt.figure(figsize=(8, 6))

# Plotting training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.tight_layout()
plt.show()
