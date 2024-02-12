# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 12:56:33 2024

@author: muhab
"""
from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np # For numerical operations and aliases it as np.
import imghdr # Determine the type of an image
import os # operating system interfaces

# Useful scikit-learn functions
from sklearn.model_selection import train_test_split # For splitting data into training and testing sets.

# For brevity import specific keras objects
from tensorflow.keras import Sequential # A linear stack of layers for building neural network models.
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D # Standard fully connected neural network layers in Keras.
from tensorflow.keras.layers import Flatten # To flatten the convolutional images
from tensorflow.keras.utils import to_categorical # For encoding the labels
from tensorflow.keras.utils import image_dataset_from_directory # Loading the data from directory
from tensorflow.keras.utils import img_to_array # For converting images to numpy arrays
from tensorflow.keras.optimizers import Adam # The optimizer used whilst training the model

# For repeatable results:
from tensorflow.random import set_seed
from random import seed
SEED = 3
seed(SEED)
np.random.seed(SEED)
set_seed(SEED)

# Loading the "flower photos" dataset from directory
data_dir = 'flower_photos'

# Printing the class names of the classes within the "flower photos" folder which contain our images 
print('Class Names: %s\n' % os.listdir(data_dir))

# Printing the file names of the images in the "daisy-class"
print('Images wthin file:\n %s\n' % 
    os.listdir(os.path.join(data_dir, 'daisy')))

# Printing the extension of a file in the "daisy-class"
print('File extension of a particular file:\n %s\n' % 
    imghdr.what('flower_photos/daisy/3415180846_d7b5cced14_m.jpg'))

# Identifying and removing any non-standard images
count = 0
image_exts = ['jpeg','jpg', 'bmp', 'png']
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(
                    image_path))
                #os.remove(image_path) # uncomment out to remove
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path) # uncomment out to remove
        count += 1

print('Total number of images: %d' % count)

# Storing the total number of images in the file
num_images = 3670

# Scaling all images to the same dimensions
img_height = 200
img_width = 200

# Loading the data from directory (info printed to the console)
'''
data = image_dataset_from_directory(data_dir, 
    batch_size=500, image_size=(img_height,img_width)) #32% ACCURACY # To load only 500 of 3670 images

data = image_dataset_from_directory(data_dir, 
    batch_size=1531, image_size=(img_height,img_width)) #61% ACCURACY # To load the images of the first two classes only
'''
data = image_dataset_from_directory(data_dir, 
    batch_size=num_images, image_size=(img_height,img_width)) #71% ACCURACY # To load all images


# Printing the class names
classes = data.class_names
print(classes)

# Extracting the images and the class labels
for images, labels in data:
    print(images.shape)
    print(labels.shape)
    break # To extract a single batch
    
# Storing in diferent variables, the total number of clases in the file and their shapes
num_classes = 5
input_shape = (200, 200, 3)

# Converting class labels to numpy arrays
labels = np.asarray(labels)

# One-hot encoding of the labels
y_train_encoded = to_categorical(labels, num_classes)
y_test_encoded = to_categorical(labels, num_classes)

# Converting images to numpy arrays
img=np.zeros(images.shape)
for i in range(images.shape[0]):
    img[i] = img_to_array(images[i])

# Scaling images
img = img / 255.0

'''
# Ploting some images from the dataset
fig, ax = plt.subplots(1,4, figsize=(10,3))

for idx, img1 in enumerate(img[:4]):
    ax[idx].imshow(img1)
    ax[idx].set_title(classes[labels[idx]])

fig.tight_layout()
fig.savefig('fp_images.png')
'''

# Spliting the input and output columns into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(img, y_train_encoded, test_size=0.05)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_train.shape)


# Defining the model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Displaying the summary of the model
model.summary()

# Compiling the model (Defining loss function and optimizer)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test)) # Just 3 epochs to avaoid learning unnecessary noices around the pixels. 
    
# Evaluating the model
loss, acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {acc}')

# Saving the model to the folder models
model.save('./models/fp_classifier.keras')  
