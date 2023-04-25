#%%
import tensorflow as tf
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn import classification_report


def plot_sample(X, y, index):
    plt.figure(figsize = (15, 2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])

#loading the cifar10 dataset to train_images/labels and validate it with test_images/labels
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

print(X_train.shape)
print(X_test.shape)
print(X_train[0])


y_train = y_train.reshape(-1,)
#train_labels does not need to be a 2d array, reshaped to be one dimensional
print(y_train[:5])

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


plot_sample(X_train, y_train, 1)

#normalize pixel values to be between 0 and 1
X_train = X_train / 255
X_test = X_test / 255
#testing accuracy with Artificial Neural Network (ANN)
#testing resulted in
#Epoch 1/5
#1563/1563 [==============================] -78s 49ms/step - loss: 1.8084 - accuracy: 0.3562
#Epoch 2/5
#563/1563 [==============================] - 79s 51ms/step - loss: 1.6223 - accuracy: 0.4297
#Epoch 3/5
#1563/1563 [==============================] - 81s 52ms/step - loss: 1.5404 - accuracy: 0.4572
#Epoch 4/5
#1563/1563 [==============================] - 82s 52ms/step - loss: 1.4817 - accuracy: 0.4766
#Epoch 5/5
#1563/1563 [==============================] - 83s 53ms/step - loss: 1.4323 - accuracy: 0.4953
#1563/1563 [==============================] - 19s 12ms/step - loss: 1.4017 - accuracy: 0.5053
# 50 percent was the highest result
"""ann = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (32, 32, 3)),
    #using 3000 neurons, then 1000, then 10 neurons
    tf.keras.layers.Dense(3000, activation = 'relu'),
    tf.keras.layers.Dense(1000, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'sigmoid')
])

ann.compile(optimizer = 'SGD', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
ann.fit(X_train, y_train, epochs = 5)
ann.evaluate(X_test, y_test)"""

#testing using a convolutional neural network (CNN)
## results for CNN were as follows
"""Epoch 1/10
1563/1563 [==============================] - 34s 21ms/step - loss: 1.4991 - accuracy: 0.4636
Epoch 2/10
1563/1563 [==============================] - 32s 20ms/step - loss: 1.1632 - accuracy: 0.5933
Epoch 3/10
1563/1563 [==============================] - 30s 19ms/step - loss: 1.0460 - accuracy: 0.6350
Epoch 4/10
1563/1563 [==============================] - 25s 16ms/step - loss: 0.9677 - accuracy: 0.6627
Epoch 5/10
1563/1563 [==============================] - 25s 16ms/step - loss: 0.9122 - accuracy: 0.6813
Epoch 6/10
1563/1563 [==============================] - 25s 16ms/step - loss: 0.8566 - accuracy: 0.7020
Epoch 7/10
1563/1563 [==============================] - 25s 16ms/step - loss: 0.8142 - accuracy: 0.7164
Epoch 8/10
1563/1563 [==============================] - 24s 16ms/step - loss: 0.7780 - accuracy: 0.7298
Epoch 9/10
1563/1563 [==============================] - 24s 16ms/step - loss: 0.7477 - accuracy: 0.7401
Epoch 10/10
1563/1563 [==============================] - 26s 16ms/step - loss: 0.7115 - accuracy: 0.7526
"""

cnn = tf.keras.models.Sequential([
    #CNN
    layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu", input_shape = (32, 32, 3)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu", input_shape = (32, 32, 3)),
    layers.MaxPooling2D((2,2)),
    #Dense
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])



# %%

cnn.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy', 
              metrics = ['accuracy'])

cnn.fit(X_train, y_train, epochs = 10)

cnn.evaluate(X_test, y_test)

#%%
y_test = y_test.reshape(-1, )
y_test[:5]

y_pred = cnn.predict(X_test)
y_pred[:5]

y_classes = [np.argmax(element) for element in y_pred]

y_test[:5]


for x in range (8):
    plot_sample(X_test, y_test, x)
    print(classes[y_classes[x]])

#%%