Group 1
| Student ID | Member Name                  | Role                                                                                   |
|------------|------------------------------|----------------------------------------------------------------------------------------|
| 298762     | Maximilian Franz             | Paper: Why is this an important contribution to research and practice                 |
| 376365     | Upanishadh Prabhakar Iyer    | Paper: The research question addressed in the paper (thus, its objective)             |
| 371696     | Lalitha Kakara               | Paper: What are their results and conclusions drawn from it? What was new in this paper at the time of publication |
| 370280     | Muhammad Tahseen Khan        | Paper: What did the authors actually do (procedure, algorithms used, input/output data, etc) |
| 372268     | Dina Mohamed                 | Paper: What did the authors actually do (procedure, algorithms used, input/output data, etc) Model: Implemented live sentiment analysis in transformer & structured repo |
| 368717     | Yash Bhavneshbhai Pathak     | Model: DAN-based Encoder algorithm implementation                                     |
| 376419     | Niharika Patil               | Model: Transformer-based Encoder algorithm implementation                             |
| 373575     | Mona Pourtabarestani         | Paper: What are their results and conclusions drawn from it? What was new in this paper at the time of publication |
| 350635     | Divya Bharathi Srinivasan    | Model: DAN-based Encoder algorithm implementation                                     |
| 364131     | Siddu Vathar                 | Paper: Why is this an important contribution to research and practice                 |

# Deep Averaging Network (DAN) for Sentiment Analysis

This project implements a **Deep Averaging Network (DAN)** using **PyTorch** for binary sentiment classification. It classifies input sentences as **Positive** or **Negative**, based on real-world product and movie reviews.

---

## Overview

- Tokenization: Uses **Stanza** to split sentences into words.
- Model: DAN (Deep Averaging Network) — a simple but effective neural network.
- Dataset: Real reviews from Amazon, IMDB, and Yelp.
- Task: Binary classification — `Positive (1)` or `Negative (0)`.
- Live prediction: You can enter your own sentence to see the sentiment result.

---

## Model Architecture (DAN)

1. **Embedding Layer**: Converts word IDs into vector representations.
2. **Average Layer**: Averages the embeddings of all words in a sentence.
3. **Fully Connected Layers**: Two linear layers with ReLU and Dropout.
4. **Output Layer**: Predicts class (positive/negative).

---

## Training Details

- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Epochs: 10
- Batch Size: 64
- Accuracy and classification report generated after training.

---

## How to Run

### Requirements

- Python 3.8+
- PyTorch
- Stanza
- scikit-learn

### Navigate to the right folder

```bash
cd dan
```

### Install Dependencies

```bash
pip install torch stanza scikit-learn
```

### Download Stanza English Model

```bash
python -c "import stanza; stanza.download('en')"
```

### Run the Script

```bash
python dan.py
```

After training, you'll be able to enter your own sentence like:

```bash
Enter a sentence (or type 'exit'): I really enjoyed the movie!
Prediction: Positive
```

---

## Note

This implementation does **not use the pretrained model** from the original paper (Cer et al., 2018).  
Instead, it **reimplements the DAN model from scratch** using PyTorch and trains it on a small review dataset for educational purposes.


---

## Reference

- Original DAN paper: [Cer et al., 2018 - Universal Sentence Encoder](https://arxiv.org/abs/1803.11175)
- Dataset: UCI Sentiment Labelled Sentences
