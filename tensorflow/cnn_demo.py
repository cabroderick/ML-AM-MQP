# simple cnn classification example from https://www.tensorflow.org/tutorials/images/cnn

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# plot first 25 images from the training set and display class names
plt.figure(figsize=(10,10)) # creates a figure, the outermost frame for a mpl graphic
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# create convolutional base using a stack of Conv2D and MaxPooling2D layers
model = models.Sequential() # groups a linear stack of layers into the model
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) #configure cnn to process images of given size
model.add(layers.MaxPooling2D((2, 2))) # adding the remainder of layers
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# add dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary() # displays a summary of the model