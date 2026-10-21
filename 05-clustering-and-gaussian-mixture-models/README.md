# Text Clustering and Gaussian Mixture Models

This assignment explores two unsupervised learning tasks: clustering health-related tweets using **K-Means** and modeling one-dimensional data distributions using **Gaussian Mixture Models (GMMs)**. It evaluates different cluster counts, examines representative tweets, and implements the Expectation-Maximization algorithm for fitting Gaussian mixtures.

## Part I: Clustering Health-Related Tweets

Use the Health News in Twitter dataset from the UCI Machine Learning Repository to group tweets according to their textual features.

### 1. Dataset Preparation and TF-IDF

Combine the dataset's text files into a single DataFrame containing tweet IDs, dates, and tweet text. Extract the tweets and transform them into numerical feature vectors using `TfidfVectorizer`, excluding English stop words and filtering terms that appear in fewer than two documents.

### 2. Dimensionality Reduction with PCA

Apply Principal Component Analysis (PCA) to reduce the TF-IDF feature space to 500 dimensions before clustering.

### 3. K-Means Clustering and Model Selection

Fit K-Means models with the following numbers of clusters:

* \(K = 50\)
* \(K = 100\)
* \(K = 200\)
* \(K = 300\)
* \(K = 400\)

Calculate the silhouette score for each configuration and identify the cluster count with the highest score.

### 4. Exploring the Resulting Clusters

For the best-performing K-Means configuration:

* Randomly sample up to five tweets from each cluster to inspect its contents.
* Find the tweet closest to each cluster centroid using cosine distance.
* Print the representative tweet and its distance from the corresponding centroid.

These examples help interpret the clusters and explore whether tweets grouped together share meaningful topics or language.

## Part II: Gaussian Mixture Models

Implement a one-dimensional Gaussian Mixture Model using the Expectation-Maximization (EM) algorithm to estimate probability distributions from observed data.

### 1. EM Algorithm Implementation

Implement the E-step to calculate component responsibilities and the M-step to update each Gaussian component's mean, standard deviation, and mixing coefficient. Repeat the updates for 100 iterations.

### 2. Fitting Mixtures to the Original Data

Load the values from `data.txt` and fit GMMs with different numbers of components:

* \(K = 1\)
* \(K = 5\)
* \(K = 7\)
* \(K = 20\)

Plot the fitted mixture density over the data histogram to compare how the number of components affects the estimated distribution.

### 3. Synthetic Gaussian Mixture

Generate 1,000 samples from a synthetic mixture of seven Gaussian components with different means and specified weights. Normalize the provided weights so they sum to one, fit a seven-component GMM, and plot the estimated density against the synthetic data distribution.

Inspect the estimated component means, standard deviations, and mixing coefficients.

## Key Takeaways

* TF-IDF converts text into numerical features suitable for machine learning.
* PCA reduces feature dimensionality before clustering.
* The silhouette score helps compare clustering configurations.
* Representative samples and centroid-nearest tweets provide ways to interpret K-Means clusters.
* The EM algorithm iteratively estimates the parameters of a Gaussian mixture.
* Comparing fitted densities with data histograms illustrates how mixture models represent distributions with multiple peaks.
