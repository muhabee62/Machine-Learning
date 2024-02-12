# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 22:31:34 2024

@author: muhab
"""
from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np # For numerical operations and aliases it as np.
import pandas as pd # For data manipulation and analysis and aliases it as pd.
import matplotlib.pyplot as plt # plotting and aliases it as plt.
from tensorflow.keras.models import Sequential # A linear stack of layers for building neural network models.
from tensorflow.keras.layers import SimpleRNN, Dense # Standard fully connected neural network layers in Keras.
#from sklearn.model_selection import train_test_split (# Defined a split sequence function instead)
from sklearn.preprocessing import MinMaxScaler # Scaling the data to improve training performance.
from tensorflow.keras.optimizers import Adam # Optimizer for the model training

# for repeatable results:
from tensorflow.random import set_seed
from random import seed
SEED = 3
seed(SEED)
np.random.seed(SEED)
set_seed(SEED)

# For splitting the training and validation sets into input(x) and output(y) seequences
def splitSequence(seq, n_steps):    
    #Declare X and y as empty list
    X = []
    y = []    
    for i in range(len(seq)):
        #get the last index
        lastIndex = i + n_steps
        #if lastIndex is greater than length of sequence then break
        if lastIndex > len(seq) - 1:
            break
        #Create input and output sequence
        seq_X, seq_y = seq[i:lastIndex], seq[lastIndex]
        #append seq_X, seq_y in X and y list
        X.append(seq_X)
        y.append(seq_y)
        pass
    #Convert X and y into numpy array
    X = np.array(X)
    y = np.array(y)
    
    return X,y

# Reading the data from file
df = pd.read_csv('sunspots_data.csv')
print(df.head())
print(df.shape, '\n')

# Plotting the original data with intervals on the x-axis
plt.figure(figsize=(12, 6))
plt.plot(df['Month'], df['Sunspot Number'], label='Original Data')
# Rotating x-axis labels for better visibility
plt.xticks(rotation=45, ha='right')
# Seting intervals on the x-axis (on every 94th label)
interval = 94
tick_positions = range(0, len(df['Month']), interval)
tick_labels = [str(month) for month in df['Month'].iloc[tick_positions]]
plt.xticks(tick_positions, tick_labels)
plt.xlabel('Month')
plt.ylabel('Sunspot Number')
plt.legend()
plt.title('Original Monthly Mean Total Sunspot Number from 1974-01 to 2023-11 With Intervals of 7.8 Years (94 Months)')
plt.tight_layout()
plt.show()

# Spliting the data into training and validation sets
t_end = 2700
data = df.iloc[:t_end+1,1].values.reshape(-1,1) # 80% of data for training
test = df.iloc[t_end:,1].values.reshape(-1,1) #20% of data for testing
print(data.shape, '\n')
print(test.shape, '\n')

# Deciding the number of steps to be fed into model
n_steps = 5
# Creating sequences for input(x) and output(y)
X_train, y_train = splitSequence(data, n_steps)
X_test, y_test = splitSequence(test, n_steps)
print(X_train.shape, '\n')
print(y_train.shape, '\n')
print(X_test.shape, '\n')
print(y_test.shape, '\n')

# Printing the input/output pairs
print('input/output pairs:')
for i in range(10):
    print(X_train[i].T, y_train[i])
print('')

# Scaling the data and saving the scaling so we can invert later
scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)
test = scaler.transform(test)
print(data.shape, '\n')
print(test.shape, '\n')

# Reshaping the input data for RNN into [samples, timesteps, features]
n_features = 1
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 
    n_features))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 
    n_features))

# Creating the model
model = Sequential()
model.add(SimpleRNN(10, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.0000001), loss='mean_absolute_error')

# Showing model summary
model.summary()

# Training the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Making predictions on the training set
train_predictions = model.predict(X_train)

# Making predictions on the validation set
test_predictions = model.predict(X_test)

# Inverting scaling to get the actual values
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
data_actual = scaler.inverse_transform(data)
test_actual = scaler.inverse_transform(test)

# Plotting the original data with predictions for the training set and validation set
plt.figure(figsize=(12, 6))
# Plotting the original data
plt.plot(np.arange(0, t_end + 1), data_actual, label='Training Data')
# Plotting predictions for the training set
train_pred_range = np.arange(n_steps, n_steps + len(train_predictions))
plt.plot(train_pred_range, train_predictions, label='Training Data Predictions', linestyle='--')
# Plotting the validation set and predictions for the validation set
val_pred_range = np.arange(t_end + 1, t_end + 1 + len(test_predictions))
plt.plot(val_pred_range, test_actual[-len(test_predictions):], label='Validation Data')
plt.plot(val_pred_range, test_predictions, label='Validation Data Predictions', linestyle='--')
# Rotating x-axis labels for better visibility
plt.xticks(rotation=45, ha='right')
# Setting intervals on the x-axis (on every 94th label)
interval = 94
tick_positions = range(0, len(df['Month']), interval)
tick_labels = [str(month) for month in df['Month'].iloc[tick_positions]]
plt.xticks(tick_positions, tick_labels)
plt.xlabel('Month')
plt.ylabel('Sunspot Number')
plt.legend()
plt.title('Monthly Mean Total Sunspot Number with Predictions')
plt.tight_layout()
plt.show()

# Plotting the training and validation data
plt.figure(figsize=(12, 6))
# Plot the original data
plt.plot(np.arange(0, t_end + 1), data_actual, label='Training Data')
# Plotting the validation set
val_pred_range = np.arange(t_end + 1, t_end + 1 + len(test_predictions))
plt.plot(val_pred_range, test_actual[-len(test_predictions):], label='Validation Data')
# Rotating x-axis labels for better visibility
plt.xticks(rotation=45, ha='right')
# Setting intervals on the x-axis (on every 94th label)
interval = 94
tick_positions = range(0, len(df['Month']), interval)
tick_labels = [str(month) for month in df['Month'].iloc[tick_positions]]
plt.xticks(tick_positions, tick_labels)
plt.xlabel('Month')
plt.ylabel('Sunspot Number')
plt.legend()
plt.title('Monthly Mean Total Sunspot Number Training And Validation Data')
plt.tight_layout()
plt.show()

# Plotting the training and validation Prediction
plt.figure(figsize=(12, 6))
# Plotting predictions for the training set
train_pred_range = np.arange(n_steps, n_steps + len(train_predictions))
plt.plot(train_pred_range, train_predictions, label='Training Data Predictions', linestyle='--')
# Plotting predictions for the validation set
val_pred_range = np.arange(t_end + 1, t_end + 1 + len(test_predictions))
plt.plot(val_pred_range, test_predictions, label='Validation Data Predictions', linestyle='--')
# Rotate x-axis labels for better visibility
plt.xticks(rotation=45, ha='right')
# Setting intervals on the x-axis (on every 94th label)
interval = 94
tick_positions = range(0, len(df['Month']), interval)
tick_labels = [str(month) for month in df['Month'].iloc[tick_positions]]
plt.xticks(tick_positions, tick_labels)
plt.xlabel('Month')
plt.ylabel('Sunspot Number')
plt.legend()
plt.title('Monthly Mean Total Sunspot Number Training And Validation Prediction')
plt.tight_layout()
plt.show()
















