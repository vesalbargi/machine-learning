# CNN for Handwritten Digit Classification

This assignment builds and evaluates convolutional neural networks (CNNs) for handwritten digit recognition using the MNIST dataset. It compares a baseline CNN against a dropout-regularized version and models trained with different optimization algorithms.

## Dataset Preparation

- Load the MNIST handwritten digit dataset.
- Visualize sample digit images.
- Reshape the images to include a single grayscale channel.
- Normalize pixel values to the range \([0, 1]\).
- Convert digit labels into one-hot encoded vectors for multiclass classification.

## CNN Architecture

Build a five-stage classification network consisting of:

1. **Convolutional layer 1:** 40 filters of size $(5 \times 5)$, with ReLU activation, followed by $(2 \times 2)$ max pooling.
2. **Convolutional layer 2:** 30 filters of size $(3 \times 3)$, with ReLU activation, followed by max pooling.
3. **Convolutional layer 3:** 20 filters of size $(2 \times 2)$, with ReLU activation, followed by max pooling.
4. **Fully connected layer:** 60 neurons with ReLU activation.
5. **Output layer:** 10 neurons with softmax activation for digit classification.

Train the baseline model using categorical cross-entropy, the Adam optimizer, 10 epochs, and a batch size of 80.

## Experiments

### 1. Baseline CNN with Adam

- Train the initial convolutional neural network.
- Track training and validation performance.
- Evaluate the trained model on the MNIST test set.

### 2. Dropout Regularization

- Add dropout layers with a rate of 0.10 after each max pooling layer.
- Train and evaluate the modified architecture.
- Compare its test accuracy with the baseline CNN to assess whether dropout improves generalization.

### 3. RMSprop Optimizer

- Train the same CNN architecture using RMSprop.
- Measure test loss and accuracy.
- Compare its performance with the Adam-based model.

### 4. Stochastic Gradient Descent (SGD)

- Train the CNN using SGD while keeping the architecture and training settings consistent.
- Evaluate test performance and compare the results with Adam and RMSprop.

## Results and Comparison

The notebook reports test accuracy for four configurations:

- Baseline CNN with Adam
- CNN with Adam and dropout
- CNN with RMSprop
- CNN with SGD

Use the measured results to compare the effect of dropout regularization and optimizer choice on handwritten digit classification. Accuracy values depend on training conditions and should be taken from the actual notebook execution.

## Key Takeaways

- Prepare image data for convolutional neural networks.
- Understand how convolution and max pooling extract spatial features.
- Apply dropout to help reduce overfitting.
- Compare optimization algorithms under a consistent CNN architecture.
- Evaluate multiclass image classification using test loss and accuracy.
