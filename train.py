"""
Created on Sat Dec 14 10:25:13 2019

@author: Aravind
"""

import os
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop  
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

# Directory with rock pictures
paper = os.path.join(r'D:\ml\deep learning\rock-paper-scissors-master\image_data\papers')
# Directory with rock pictures 
rock = os.path.join(r'D:\ml\deep learning\rock-paper-scissors-master\image_data\rocks')
# Directory with scissors pictures
scissors = os.path.join(r'D:\ml\deep learning\rock-paper-scissors-master\image_data\scissors')
# Directory with none pictures
none = os.path.join(r'D:\ml\deep learning\rock-paper-scissors-master\image_data\none')


train_paper = os.listdir(paper)
print(train_paper[:5])

train_rock = os.listdir(rock)
print(train_rock[:5])


train_scissors = os.listdir(scissors)
print(train_scissors[:5])

train_none = os.listdir(none)
print(train_none[:5])



batch_size = 64



# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(r'D:\ml\deep learning\rock-paper-scissors-master\image_data',  # This is the source directory for training images
        target_size=(227, 227),  # All images will be resized to 200 x 200
        batch_size=batch_size,
        # Specify the classes explicitly
        classes = ['none','papers','rocks','scissors'],
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')
target_size=(227,227)

#$input_shape = tuple(list(target_size)+[3])
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 200x 200 with 3 bytes color
    # The first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(227, 227, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a dense layer
    tf.keras.layers.Flatten(),
    # 128 neuron in the fully-connected layer
    tf.keras.layers.Dense(128, activation='relu'),
    # 5 output neurons for 5 classes with the softmax activation
    tf.keras.layers.Dense(4, activation='softmax')
])
model.summary()


# Optimizer and compilation
model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['acc'])#RMSprop(lr=0.001)
# Total sample count
total_sample=train_generator.n
# Training
num_epochs = 5
model.fit_generator(train_generator,steps_per_epoch=int(total_sample/batch_size),
                    epochs=num_epochs,verbose=1)







history = model.fit_generator(
        train_generator, 
        steps_per_epoch=int(total_sample/batch_size),  
        epochs=num_epochs,
        verbose=1)



# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model1.h5")
print("Saved model to disk")
