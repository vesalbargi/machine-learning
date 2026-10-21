# Tree-Based Machine Learning Models for Imbalanced Data

This assignment applies tree-based machine learning models to the **CDC Diabetes Health Indicators dataset** to predict diabetes diagnosis. It compares four classifiers and investigates how class-imbalance strategies and hyperparameter tuning affect performance, using Precision-Recall AUC (PR-AUC) and confusion matrices for evaluation.

## 1. Dataset Preparation

Retrieve the CDC Diabetes Health Indicators dataset from the UCI Machine Learning Repository and inspect its metadata, features, target labels, and class distribution.

**Key tasks:**

* Load the dataset using `ucimlrepo`.
* Examine the proportion of positive and negative diabetes diagnoses.
* Split the data into training (70%) and test (30%) sets.
* Use stratified sampling to preserve the class distribution in both sets.

## 2. XGBoost Classification

Train XGBoost classifiers with different numbers of estimators, tree depths, and learning rates.

**Key tasks:**

* Calculate `scale_pos_weight` from the ratio of negative to positive training examples.
* Compare three hyperparameter configurations.
* Evaluate each model using PR-AUC and a confusion matrix.
* Identify the configuration with the highest PR-AUC.

## 3. Decision Tree Classification

Compare an untuned baseline Decision Tree with tuned models that account for class imbalance.

**Key tasks:**

* Train a baseline classifier without class weighting or hyperparameter tuning.
* Train balanced Decision Trees with different maximum depths and minimum leaf sizes.
* Compare baseline and tuned PR-AUC scores and confusion matrices.
* Examine how class weighting and tree complexity affect classification performance.

## 4. Random Forest Classification

Evaluate Random Forest models using different ensemble sizes, tree depths, and minimum leaf sizes.

**Key tasks:**

* Apply `class_weight="balanced"` to address class imbalance.
* Compare three hyperparameter configurations.
* Evaluate each model using PR-AUC and a confusion matrix.
* Select the best configuration based on PR-AUC.

## 5. AdaBoost Classification

Train AdaBoost models using different numbers of estimators and learning rates.

**Key tasks:**

* Calculate balanced sample weights from the training labels.
* Pass the sample weights to the model during training.
* Compare the three hyperparameter configurations using PR-AUC and confusion matrices.
* Identify the best-performing configuration.

## 6. Model Comparison and Evaluation

Compare the best-performing configurations from all four classifiers and determine which model achieves the highest PR-AUC.

The assignment uses two complementary evaluation methods:

* **PR-AUC:** Assesses the precision-recall trade-off, making it useful when the positive class is relatively uncommon.
* **Confusion matrix:** Shows the distribution of correct and incorrect predictions for each class, helping reveal false positives and false negatives.

The notebook's reported comparison identifies XGBoost with `n_estimators=200`, `max_depth=5`, and `learning_rate=0.05` as the best configuration, with a PR-AUC of **0.4274**.

## Key Takeaways

* Class imbalance can make accuracy misleading when the positive class is uncommon.
* Class weighting and sample weighting provide different ways to account for imbalanced training data.
* Hyperparameters influence the complexity and predictive performance of tree-based models.
* PR-AUC and confusion matrices provide complementary insights into minority-class classification.
* Comparing tuned models against an untuned baseline helps demonstrate the effects of model configuration and imbalance handling.