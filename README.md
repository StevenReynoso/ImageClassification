# ImageClassification
Image Classification while using a Artificial Neural Network and then after trying with a Convolutional Neural Network, using the Cifar-10 dataset

Image Classification with Convolutional Neural Networks (CNN)

In this project, we explore image classification using Convolutional Neural Networks (CNNs). We use the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes include: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

We first preprocess the images by normalizing the pixel values between 0 and 1. Then, we train and evaluate two different models: an Artificial Neural Network (ANN) and a CNN.

The ANN consists of three layers: two Dense layers with 3,000 and 1,000 neurons respectively, and a final Dense layer with 10 neurons (one for each class). We use the sigmoid activation function for the last layer and the sparse_categorical_crossentropy loss function for the optimizer.

The CNN consists of three Conv2D layers with 32, 64, and 128 filters respectively, each followed by a MaxPooling2D layer with a (2, 2) pool size. We then add two Dense layers with 64 and 10 neurons respectively. We use the softmax activation function for the last layer and the Adam optimizer with sparse_categorical_crossentropy loss function.

We train both models on the CIFAR-10 dataset and evaluate their accuracy on the test set. The CNN performs significantly better than the ANN, achieving a test accuracy of 70.91% compared to 50.53% for the ANN.

Finally, we make predictions on a sample of images from the test set and display the predicted classes along with the corresponding images.

This project demonstrates the power of CNNs in image classification tasks and provides a starting point for exploring more advanced CNN architectures and datasets.
