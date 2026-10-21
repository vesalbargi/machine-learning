# Generative Learning Algorithms

This assignment explores **Gaussian Discriminant Analysis (GDA)** and **Naive Bayes** as generative learning algorithms. It implements a Gaussian-based classifier for the Iris dataset, builds a Bernoulli Naive Bayes classifier for spam detection, and derives the relationship between exponential-family distributions and the sigmoid function.

## Part I: Gaussian Discriminant Analysis

Implement a multiclass Gaussian Discriminant Analysis classifier using the Iris dataset, which contains four measurements for three iris species.

**Key tasks:**

* Load the Iris dataset and encode the species labels numerically.
* Implement the multivariate Gaussian probability density function using a mean vector and covariance matrix.
* Estimate class prior probabilities, class-specific mean vectors, and a shared covariance matrix from the training data.
* Predict the species of test samples by combining Gaussian likelihoods with class prior probabilities.
* Evaluate predictions using a confusion matrix and a heatmap.

**Result:** Build a generative classifier that models the feature distribution for each class and predicts the most likely iris species.

## Part II: Bernoulli Naive Bayes for Spam Detection

Implement a Bernoulli Naive Bayes classifier that distinguishes spam from non-spam text documents. Each file is treated as a document, and each word is represented by its presence or absence rather than its total frequency.

**Key tasks:**

* Load spam and non-spam documents from separate training and testing directories.
* Build a vocabulary and count how many documents in each class contain each word.
* Estimate word probabilities using smoothing and calculate class prior probabilities.
* Compute document scores in log space, accounting for both words that appear and words that are absent.
* Evaluate spam and non-spam classification accuracy on the test documents.

**Result:** Implement a document-level spam classifier using Bernoulli word features, smoothed probability estimates, and logarithmic scoring.

## Part III: Exponential Family and the Sigmoid Function

Derive the sigmoid function from a generative probabilistic model whose class-conditional distributions belong to the exponential family.

**Key steps:**

* Apply Bayes' rule to express the posterior probability of a binary class.
* Rewrite the posterior in terms of class odds and class-conditional likelihoods.
* Use the exponential-family representation to simplify the likelihood ratio.
* Show that the log-odds are linear in the sufficient statistics, leading to the sigmoid expression for the posterior probability.

**Result:** Establish the connection between generative probabilistic models and the sigmoid form used in binary logistic regression.