# IMDB Movie Review Sentiment Analysis

This assignment explores binary sentiment classification using the IMDB movie review dataset. It compares three approaches to natural language processing: FastText, a Long Short-Term Memory (LSTM) network, and a pretrained DistilBERT transformer.

The goal is to classify reviews as positive (`1`) or negative (`0`) and investigate how model architecture and training parameters affect performance.

## Part I — FastText Classification

### Prepare the Review Data

- Load IMDB reviews using Keras with a vocabulary limit of 10,000 words.
- Decode token sequences into readable text.
- Create `imdb_train.txt` with the first 10,000 training reviews and `imdb_test.txt` with the first 5,000 test reviews.
- Format each line with a FastText label, such as `__label__1`, followed by the review text.

### Train and Tune FastText

- Train a baseline supervised FastText classifier using default settings.
- Evaluate its precision, recall, and classification accuracy.
- Experiment with a learning rate of `0.5`, 10 training epochs, word bigrams, and 100-dimensional word vectors.
- Compare the tuned model with the default model.

## Part II — Sentiment Classification with LSTM

### Sequence Preparation

- Load the IMDB dataset with a vocabulary limit of 200 words.
- Pad or truncate each review to 200 tokens so the sequences have a consistent length.

### Build and Train the LSTM

Construct a sequential model containing:

1. An embedding layer with a vocabulary size of 200 and 128-dimensional word vectors.
2. An LSTM layer with 64 units to learn sequential patterns.
3. A sigmoid output layer for binary sentiment classification.

Compile the model with binary cross-entropy loss and the Adam optimizer. Train for three epochs with a batch size of 64.

### Experiment with Model Capacity and Training Duration

- Change the embedding dimension and observe how the representation size affects learning.
- Increase or decrease the number of training epochs.
- Compare training behavior and consider the trade-off between model capacity, computation time, and overfitting.

## Part III — Sentiment Classification with DistilBERT

### Dataset and Tokenization

- Load the IMDB dataset using Hugging Face Datasets.
- Combine the first 100,000 training examples with 5,000 test examples, then create an 80/20 training and evaluation split.
- Tokenize reviews using the pretrained `distilbert-base-uncased` tokenizer.
- Prepare token IDs, attention masks, and sentiment labels for PyTorch training.

### Fine-Tune the Transformer

- Load `distilbert-base-uncased` using `AutoModelForSequenceClassification` with two output labels.
- Configure training through `TrainingArguments`, including the output directory and number of epochs.
- Train the model using Hugging Face `Trainer`.
- Evaluate the fine-tuned model and calculate classification accuracy.
- Investigate how increasing the number of epochs affects performance and training cost.

## Model Comparison

The notebook compares three distinct approaches:

- **FastText:** efficient supervised text classification using word representations and n-gram features.
- **LSTM:** sequence modeling with learned embeddings and recurrent connections.
- **DistilBERT:** transformer-based contextual language representations with transfer learning.

Compare the measured accuracy where available, along with training time, computational requirements, and the effect of parameter changes. The notebook does not establish a universal winner; results depend on the data split and training configuration.

## Key Takeaways

- Convert tokenized reviews into text suitable for FastText training.
- Compare lightweight text classification with recurrent neural networks and transformer fine-tuning.
- Understand embeddings, sequence padding, and contextual text representations.
- Explore the impact of learning rate, n-grams, embedding dimensions, and training epochs.
- Evaluate sentiment classifiers using classification accuracy, precision, and recall.
