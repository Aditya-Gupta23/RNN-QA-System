# RNN-Based Question Answering System using PyTorch

## Overview

This project implements a simple Question Answering (QA) system using a Recurrent Neural Network (RNN) built with PyTorch. The model is trained on a custom dataset of question-answer pairs and learns to predict the correct answer given a question.

The project demonstrates fundamental NLP concepts including:

- Text preprocessing
- Tokenization
- Vocabulary creation
- Numerical encoding of text
- Custom PyTorch Dataset and DataLoader
- Word Embeddings
- Recurrent Neural Networks (RNN)
- Model training and inference

---

## Dataset

The dataset consists of question-answer pairs stored in:

```text
100_Unique_QA_Dataset.csv
```

Example:

| Question | Answer |
|-----------|---------|
| What is the capital of France? | Paris |
| What is the capital of Germany? | Berlin |
| What is the largest planet in our solar system? | Jupiter |

The dataset is used to build a vocabulary and train the model.

---

## Project Structure

```text
.
├── RNN_Q&A_PyTorch.ipynb
├── 100_Unique_QA_Dataset.csv
└── README.md
```

---

## Workflow

### 1. Data Loading

The dataset is loaded using Pandas.

```python
df = pd.read_csv("100_Unique_QA_Dataset.csv")
```

### 2. Text Preprocessing

Questions and answers are:

- Converted to lowercase
- Cleaned by removing punctuation
- Tokenized into words

Example:

```text
"What is the capital of France?"
```

becomes

```text
["what", "is", "the", "capital", "of", "france"]
```

### 3. Vocabulary Construction

A vocabulary is created from all words appearing in both questions and answers.

Special token:

```text
<UNK>
```

is used for unknown words.

Example:

```python
{
    "<UNK>": 0,
    "what": 1,
    "is": 2,
    ...
}
```

### 4. Numerical Encoding

Each token is converted into its corresponding vocabulary index.

Example:

```text
["what", "is", "france"]
```

becomes

```text
[1, 2, 45]
```

### 5. Dataset and DataLoader

A custom PyTorch Dataset is implemented to serve question-answer pairs during training.

```python
class QADataset(Dataset):
    ...
```

### 6. Model Architecture

The model consists of:

1. Embedding Layer
2. RNN Layer
3. Fully Connected Layer

Architecture:

```text
Question
   ↓
Embedding Layer
   ↓
RNN
   ↓
Final Hidden State
   ↓
Linear Layer
   ↓
Predicted Answer
```

Implementation:

```python
Embedding(vocab_size, 50)
RNN(50, 64)
Linear(64, vocab_size)
```

---

## Training

Loss Function:

```python
nn.CrossEntropyLoss()
```

Optimizer:

```python
torch.optim.Adam()
```

Hyperparameters:

```python
learning_rate = 0.001
epochs = 20
```

Training loop performs:

1. Forward pass
2. Loss computation
3. Backpropagation
4. Parameter updates

---

## Concepts Demonstrated

This project helps understand:

- NLP preprocessing pipeline
- Word embeddings
- Sequence modeling with RNNs
- Hidden states in recurrent networks
- Classification using neural networks
- PyTorch Dataset and DataLoader APIs

---

## Requirements

Install the required libraries:

```bash
pip install pandas torch numpy
```

---

## Running the Project

Open the notebook:

```bash
jupyter notebook RNN_Q&A_PyTorch.ipynb
```

Run all cells sequentially to:

1. Load the dataset
2. Build the vocabulary
3. Create the dataset and dataloader
4. Train the RNN model
5. Evaluate predictions

---

## Limitations

This implementation is intentionally simple and intended for learning purposes.

Limitations include:

- Small dataset
- Single-layer RNN
- Limited vocabulary
- No attention mechanism
- No sequence-to-sequence architecture
- Poor generalization to unseen questions

---

## Future Improvements

Possible enhancements:

- Use LSTM or GRU instead of vanilla RNN
- Increase dataset size
- Add padding and batching support
- Implement sequence-to-sequence learning
- Use pretrained embeddings
- Integrate Transformer-based architectures
- Build a retrieval-augmented QA system

---

## Technologies Used

- Python
- Pandas
- PyTorch
- Jupyter Notebook

---

## Learning Objective

The primary goal of this project is to understand how a Question Answering system can be built from scratch using basic NLP preprocessing techniques and Recurrent Neural Networks in PyTorch.