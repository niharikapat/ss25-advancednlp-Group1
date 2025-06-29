# Group 1
# Student ID        Member Name                      Role
# -------------------------------------------------------------------------------
# 298762            Maximilian Franz                Paper: Why is this an important contribution to research and practice
# 376365            Upanishadh Prabhakar Iyer       Paper: The research question addressed in the paper (thus, its objective)
# 371696            Lalitha Kakara                  Paper: What are their results and conclusions drawn from it? 
#                                                   What was new in this paper at the time of publication (with respect to the literature that existed beforehand)?
# 370280            Muhammad Tahseen Khan           Paper: What did the authors actually do (procedure, algorithms used, input/output data, etc)
# 372268            Dina Mohamed                    Paper: What did the authors actually do (procedure, algorithms used, input/output data, etc) Model: Implemented live sentiment analysis in transformer & structured repo
# 368717            Yash Bhavneshbhai Pathak        Model: DAN-based Encoder algorithm implementation
# 376419            Niharika Patil                  Model: Transformer-based Encoder algorithm implementation
# 373575            Mona Pourtabarestani            Paper: What are their results and conclusions drawn from it? 
#                                                   What was new in this paper at the time of publication (with respect to the literature that existed beforehand)?
# 350635            Divya Bharathi Srinivasan       Model: DAN-based Encoder algorithm implementation
# 364131            Siddu Vathar                    Paper: Why is this an important contribution to research and practice

import torch
import torch.nn as nn
import torch.nn.functional as F
import stanza
import os
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Download and initialize Stanza
stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True)

# Load dataset from 3 text files
def load_dataset(folder):
    texts = []
    labels = []
    files = ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']
    for filename in files:
        with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    text, label = line.split('\t')
                    texts.append(text)
                    labels.append(int(label))
    return texts, torch.tensor(labels)

# Tokenize text using Stanza
def tokenize_sentences(sentences):
    tokenized = []
    for sent in sentences:
        doc = nlp(sent)
        words = [word.text.lower() for s in doc.sentences for word in s.words]
        tokenized.append(words)
    return tokenized

# Build vocab and mapping
def build_vocab(token_lists):
    vocab = ['<PAD>', '<UNK>']
    for sent in token_lists:
        for token in sent:
            if token not in vocab:
                vocab.append(token)
    return vocab

def tokens_to_indices(token_lists, vocab):
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    indexed = []
    for sent in token_lists:
        indexed.append([word_to_idx.get(word, word_to_idx['<UNK>']) for word in sent])
    return indexed, word_to_idx

# Pad sequences
def pad_sequences(seqs, pad_idx, max_len=None):
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    padded = []
    for seq in seqs:
        seq = seq[:max_len] + [pad_idx] * (max_len - len(seq))
        padded.append(seq)
    return torch.tensor(padded)

# DAN model
class DAN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embeds = self.embedding(x)
        avg = embeds.mean(dim=1)
        x = F.relu(self.fc1(avg))
        x = self.dropout(x)
        return self.fc2(x)

# Load and preprocess full dataset
folder = './../data'
texts, labels = load_dataset(folder)
tokens = tokenize_sentences(texts)
vocab = build_vocab(tokens)
indexed, word_to_idx = tokens_to_indices(tokens, vocab)
inputs = pad_sequences(indexed, word_to_idx['<PAD>'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, stratify=labels, random_state=42)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

# Initialize model
model = DAN(vocab_size=len(vocab), embed_dim=64, hidden_dim=128, output_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Train
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        output = model(batch_x)
        loss = loss_fn(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    preds = model(X_test).argmax(dim=1)
    print("\nAccuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=["Negative", "Positive"]))

# Predict custom sentence
def predict_text(text):
    tokens = tokenize_sentences([text])
    indexed, _ = tokens_to_indices(tokens, vocab)
    padded = pad_sequences(indexed, word_to_idx['<PAD>'], max_len=X_train.shape[1])
    with torch.no_grad():
        out = model(padded)
        pred = out.argmax(dim=1).item()
    return "Positive" if pred == 1 else "Negative"

# Try it!
while True:
    user_input = input("\nEnter a sentence (or type 'exit'): ")
    if user_input.lower() == 'exit':
        break
    print("Prediction:", predict_text(user_input))
