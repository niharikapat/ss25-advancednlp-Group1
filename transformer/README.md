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


# Transformer-based Encoder

This module implements a **Transformer-based encoder** for binary sentiment classification. It classifies input sentences as **Positive** or **Negative**, based on real-world product and movie reviews.

The model is trained on the shared dataset located in the `../data/` folder, which includes labeled sentences from Amazon, IMDB, and Yelp.

## How to Run

### Requirements

- Python 3.8+
- PyTorch
- Stanza
- scikit-learn

### Navigate to the right folder

```bash
cd transformer
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
python transformer-encoder.py
```

After training, you'll be able to enter your own sentence like:

```bash
Enter a sentence (or type 'exit'): I really enjoyed the movie!
Prediction: Positive
```


After training, you'll be prompted to input your own sentences for live sentiment prediction.
