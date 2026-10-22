# Machine Learning

This repository contains a collection of machine learning assignments implemented in Python using Jupyter Notebooks. The assignments explore supervised learning, generative models, tree-based classification, unsupervised learning, neural networks, convolutional neural networks, and natural language processing through practical implementations and experiments.

## Repository Structure

| # | Assignment | Description |
|---|---|---|
| 01 | Linear Regression and Gradient Descent | Implementing linear regression, gradient descent, the Normal Equation, feature normalization, polynomial regression, regularization, and locally weighted linear regression. |
| 02 | Logistic Regression and Classification | Exploring binary and multiclass logistic regression, nonlinear decision boundaries, regularization, and classification evaluation on breast cancer and handwritten digit datasets. |
| 03 | Generative Learning Algorithms | Implementing Gaussian Discriminant Analysis and Bernoulli Naive Bayes for spam classification, and deriving the sigmoid function from the exponential family. |
| 04 | Tree-Based Diabetes Classification | Comparing Decision Trees, Random Forests, AdaBoost, and XGBoost for imbalanced diabetes classification using precision-recall evaluation. |
| 05 | Clustering and Gaussian Mixture Models | Applying TF-IDF, PCA, and K-Means to health-related tweets, then exploring Gaussian Mixture Models and the Expectation-Maximization algorithm. |
| 06 | Neural Networks for Digit Recognition | Implementing a two-layer neural network from scratch, including forward propagation, backpropagation, gradient checking, regularization, and handwritten digit classification. |
| 07 | CNN for Digit Classification | Building convolutional neural networks for MNIST and comparing dropout regularization with Adam, RMSprop, and SGD optimizers. |
| 08 | IMDB Sentiment Analysis | Comparing FastText, LSTM, and DistilBERT models for binary movie-review sentiment classification. |

Each assignment is stored in its own numbered directory. Supporting datasets and other required files are kept alongside the corresponding notebooks when needed. Some notebooks download datasets automatically, while others expect local data files.

## Technologies Used

- **Python** — primary programming language for machine learning implementations.
- **Jupyter Notebook** — interactive experimentation, visualization, and analysis.
- **NumPy** — numerical computation, matrix operations, and vectorized algorithms.
- **pandas** — tabular data loading and processing.
- **Matplotlib and Seaborn** — plotting, decision boundaries, and model evaluation visualizations.
- **SciPy** — numerical optimization and scientific computing.
- **scikit-learn** — machine learning algorithms, preprocessing, cross-validation, and evaluation metrics.
- **XGBoost** — gradient-boosted tree classification.
- **Keras / TensorFlow** — neural networks, CNNs, and sequence models.
- **FastText** — supervised text classification and word representations.
- **Hugging Face Transformers and Datasets** — transformer-based sentiment classification and dataset preparation.
- **PyTorch** — tensor-based training support for transformer models.
- **Joblib** — saving and loading trained model artifacts.
- **ucimlrepo** — retrieving datasets from the UCI Machine Learning Repository.

Not every assignment requires every library. The dependencies depend on the notebook being executed, and some datasets or pretrained models may be downloaded when a notebook runs.

## Getting Started

### Requirements

Install Python and Jupyter Notebook or JupyterLab. A Python virtual environment is recommended to keep project dependencies isolated.

### Installation

Clone the repository

```bash
git clone https://github.com/vesalbargi/machine-learning.git
```

Create a Python virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment

- On Windows:

  ```bash
  .venv\Scripts\Activate.ps1
  ```

- On macOS/Linux:

  ```bash
  source .venv/bin/activate
  ```

Install required packages

```bash
pip install notebook numpy pandas matplotlib seaborn scipy scikit-learn xgboost tensorflow fasttext transformers datasets torch joblib ucimlrepo
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the numbered assignment folder and run the corresponding notebook. Keep any required local datasets and supporting files in the paths expected by that notebook.

## Learning Objectives

Through these assignments, the repository explores:

- Implementing linear and logistic regression using numerical optimization.
- Understanding feature engineering, polynomial decision boundaries, and regularization.
- Applying generative classifiers such as Gaussian Discriminant Analysis and Naive Bayes.
- Training and evaluating tree-based models on imbalanced classification problems.
- Exploring unsupervised clustering, dimensionality reduction, and Gaussian mixture modeling.
- Implementing neural network forward propagation, backpropagation, and gradient checking.
- Building CNNs for image classification and comparing regularization and optimization strategies.
- Processing text data and comparing FastText, recurrent neural networks, and transformer models.
- Evaluating model performance using suitable metrics and controlled experiments.

## License

This project is licensed under the MIT License.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
