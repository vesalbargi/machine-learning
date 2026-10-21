# Neural Networks for Handwritten Digit Recognition

This assignment implements a two-layer neural network from scratch to classify handwritten digits. It covers forward propagation, the regularized cost function, backpropagation, gradient checking, parameter initialization, model training, and prediction.

The network uses 400 input features representing 20 × 20 digit images, 25 hidden units, and 10 output classes.

## Implementation

### 1. Load and Visualize the Digit Dataset

- Load the handwritten digit dataset from `data/dataset.csv`.
- Separate the input features from the digit labels.
- Display a random selection of 100 digit images in a grid.

### 2. Forward Propagation and Cost Function

- Load the provided neural network weights from `data/sampleTheta1.csv` and `data/sampleTheta2.csv`.
- Implement the sigmoid activation function and its gradient.
- Compute hidden-layer and output-layer activations.
- Convert class labels into one-hot encoded targets.
- Calculate the multiclass classification cost and compare it with the expected reference value.

### 3. Regularization

- Add L2 regularization to the neural network cost function.
- Exclude bias weights from regularization.
- Compare the regularized and unregularized cost values.

### 4. Weight Initialization

- Implement random weight initialization to break symmetry between hidden units.
- Initialize weights for the input-to-hidden and hidden-to-output layers.

### 5. Backpropagation and Gradient Checking

- Implement backpropagation to calculate gradients for both weight matrices.
- Accumulate and average gradients across training examples.
- Add regularization to the gradients without modifying bias-weight gradients.
- Compare analytical gradients with numerical finite-difference approximations.
- Check gradient correctness with and without regularization.

### 6. Train the Neural Network

- Train the network using conjugate gradient optimization with `fmin_cg`.
- Use the regularized cost function and its analytical gradient.
- Experiment with the regularization parameter and optimization iterations.

### 7. Visualize Learned Features and Make Predictions

- Visualize the trained hidden-layer weights as image-like features.
- Predict digit labels for the training dataset.
- Evaluate training accuracy using the predicted labels.

## Key Takeaways

- Understand the forward pass of a feedforward neural network.
- Implement multiclass cross-entropy cost and L2 regularization.
- Derive and validate gradients using backpropagation and numerical gradient checking.
- Understand why random initialization helps neural network training.
- Train a neural network and evaluate its digit classification performance.
