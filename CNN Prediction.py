# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 00:20:04 2024

@author: muhab
"""

from tensorflow.keras.models import load_model # For loading the saved model.
from tensorflow.keras.preprocessing import image # Loading the image for processing.
import numpy as np # For numerical operations and aliases it as np.
import matplotlib.pyplot as plt # plotting and aliases it as plt.

# For repeatable results:
from tensorflow.random import set_seed
from random import seed
SEED = 3
seed(SEED)
np.random.seed(SEED)
set_seed(SEED)


# Loading the saved model
saved_model_path = './models/fp_classifier.keras'
loaded_model = load_model(saved_model_path)

# Creating a function to preprocess the input image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(200, 200))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 #Normalizing the image pixels digital number to a range of 256 (0-255)
    return img, img_array

# Loading the input image for prediction
new_image_path = 'flower_photos/tulips/4612075317_91eefff68c_n.jpg'
input_image, preprocessed_image = preprocess_image(new_image_path)

# Using the loaded model to make prediction
predictions = loaded_model.predict(preprocessed_image)

# Converting the prediction to human-readable labels
class_labels = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'] 
predicted_class_index = np.argmax(predictions)
predicted_class_label = class_labels[predicted_class_index]

print(f'The predicted class is: {predicted_class_label}')

# Ploting the input image
plt.imshow(input_image)
plt.title(f'Predicted Class: {predicted_class_label}')
plt.show()
