# Linear Regression

This assignment explores **linear regression** and methods for estimating model parameters. It starts with a single-feature regression problem, compares gradient descent with the Normal Equation, and extends the implementation to multiple features. Further experiments investigate polynomial regression, regularization, and locally weighted linear regression.

## Implementation

### 1. One-Feature Linear Regression

Implement the hypothesis function and cost function for a single-feature dataset. Use gradient descent to optimize the model parameters and examine how the cost changes during training.

### 2. Learning Rate and Convergence

Experiment with different learning rates to observe their effects on gradient descent. Compare convergence behavior and consider how the learning rate influences the stability and speed of optimization.

### 3. Normal Equation

Implement the Normal Equation to calculate the least-squares parameters directly. Compare its solution with gradient descent and visualize the resulting regression lines against the data.

### 4. Multiple Linear Regression

Extend the model to use two input features. Train the regression model and visualize the fitted predictions as a three-dimensional regression plane. Examine the role of feature normalization when the input features have different scales.

### 5. Polynomial Regression and Regularization

Explore polynomial features to model nonlinear relationships and investigate regularization as a way to control model complexity. Relate the choice of model and regularization strength to the fitted predictions.

### 6. Locally Weighted Linear Regression

Investigate locally weighted linear regression (LWLR), which fits a locally adapted regression model by assigning different weights to training examples according to their proximity to the query point.

## Key Takeaways

* Gradient descent iteratively updates model parameters to reduce the regression cost.
* The learning rate affects the convergence speed and stability of optimization.
* The Normal Equation provides a direct least-squares solution for linear regression.
* Multiple features extend the model to higher-dimensional input spaces.
* Polynomial features can represent nonlinear relationships, while regularization helps control model complexity.
* Locally weighted linear regression adapts its fit to the region around each prediction point.