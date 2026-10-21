# Logistic Regression and Classification

This assignment explores **logistic regression** for binary and multiclass classification. It implements classification models using gradient descent, examines nonlinear decision boundaries and regularization, and evaluates model performance on several datasets, including breast cancer data and MNIST handwritten digits.

## Implementation

### 1. Sigmoid, Softmax, and Regression Models

Study the sigmoid function for binary classification and the softmax function for multiclass classification. The assignment also introduces Poisson regression for count data and discusses how different regression models are suited to different prediction tasks.

### 2. Binary Logistic Regression

Implement a vectorized logistic regression model using gradient descent. Train the classifier on a two-feature dataset and visualize the resulting decision boundary to examine how the model separates the classes.

### 3. Nonlinear Classification

Extend logistic regression with polynomial feature expansion to degree 6. Use the expanded features to represent nonlinear decision boundaries and examine the effect of feature transformations on classification.

### 4. Regularized Logistic Regression

Apply L1 and L2 regularization to logistic regression with polynomial features of degree 4. Compare the models to explore how regularization affects parameter values, decision boundaries, and overfitting.

### 5. Breast Cancer Classification

Use the breast cancer dataset to train and evaluate a logistic regression classifier. Apply feature standardization and assess performance using classification metrics.

### 6. Multiclass Classification with MNIST

Train a multiclass logistic regression model to classify handwritten digits from the MNIST dataset. Use a pipeline with feature standardization, evaluate the model with stratified 10-fold cross-validation, and save the trained model using joblib.

## Evaluation and Visualizations

The assignment uses decision-boundary plots and classification metrics to assess model behavior. Evaluation includes accuracy, precision, recall, F1-score, and confusion matrices, providing several perspectives on classification performance.

## Key Takeaways

* Logistic regression models class probabilities using the sigmoid function, while softmax supports multiclass prediction.
* Polynomial feature expansion allows linear classifiers to represent nonlinear decision boundaries.
* L1 and L2 regularization provide ways to control model complexity.
* Feature standardization is an important part of preparing data for classification.
* Cross-validation and classification metrics help assess how well a model generalizes beyond its training data.
* Saving a trained model allows it to be reused without retraining.
