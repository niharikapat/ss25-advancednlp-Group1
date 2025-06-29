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
import math

# Load the dataset
def dataset_load(folder):
    sentences, score = [], []
    files = ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']
    for filename in files:
        with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    sentences.append(parts[0])
                    score.append(int(parts[1]))
    return {"sentences": sentences, "score": torch.tensor(score)}

# Transformer-based sentence encoder
class transformer_sentence_encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, max_len=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = self._init_position_encoding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.embed_dim = embed_dim

    def _init_position_encoding(self, max_len, embed_dim):
        positional_encoding = torch.zeros(max_len, embed_dim) 
        positional_indices = torch.arange(start = 0, end = max_len, dtype=torch.float).unsqueeze(1) 
        angle_rates = torch.exp(torch.arange(start = 0, end = embed_dim,step = 2, dtype = torch.float) * (-math.log(10000.0)/embed_dim)) 
        sin_part = torch.sin(positional_indices * angle_rates)
        cos_part = torch.cos(positional_indices * angle_rates)
        positional_encoding[:, 0::2] = sin_part
        positional_encoding[:, 1::2] = cos_part
        return positional_encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len].to(x.device)
        x = self.transformer(x)
        x = x.sum(dim=1) / math.sqrt(seq_len)  
        return x

# Classifier using the encoder
class transformer_classifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()
        self.transform_encoder = transformer_sentence_encoder(vocab_size, embed_dim)
        self.transform_classifier = nn.Linear(embed_dim, 2)

    def forward(self, x):
        x = self.transform_encoder(x)
        return self.transform_classifier(x)

# Tokenizer and vocab builder
def build_vocabulary_and_tokenization(texts, nlp, max_len=64):
    vocabulary = {'<PAD>': 0, '<UNK>': 1}
    tokens = []
    for sent in texts:
        doc = nlp(sent)
        ids = []
        for word in doc.sentences[0].words:
            token = word.text.lower()
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
            ids.append(vocabulary[token])
        ids = ids[:max_len]
        ids += [vocabulary['<PAD>']] * (max_len - len(ids))  # pad
        tokens.append(ids)
    return torch.tensor(tokens), vocabulary

# Training and evaluation
def train_and_evaluate():
    files = dataset_load("../data")
    nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True, verbose=False)
    token, vocabulary = build_vocabulary_and_tokenization(files["sentences"], nlp)
    labels = files["score"]

    X_training, X_testing, y_training, y_testing = train_test_split(token, labels, test_size=0.2, random_state=42)
    training_loader = DataLoader(TensorDataset(X_training, y_training), batch_size=32, shuffle=True)
    testing_loader = DataLoader(TensorDataset(X_testing, y_testing), batch_size=32)

    model = transformer_classifier(vocab_size=len(vocabulary))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        model.train()
        for xb, yb in training_loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for xb, yb in testing_loader:
            out = model(xb)
            preds.extend(torch.argmax(out, dim=1).tolist())
            golds.extend(yb.tolist())

    print("Accuracy:", accuracy_score(golds, preds))
    print(classification_report(golds, preds))
    return model, vocabulary

model, vocabulary = train_and_evaluate()

# Predict custom sentence
def predict_text(text, model, vocab, max_len=64):
    nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True, verbose=False)
    doc = nlp(text)
    word_to_idx = vocab
    tokens = []
    for word in doc.sentences[0].words:
        token = word.text.lower()
        tokens.append(word_to_idx.get(token, word_to_idx['<UNK>']))
    tokens = tokens[:max_len] + [0] * (max_len - len(tokens))
    input_tensor = torch.tensor([tokens])

    with torch.no_grad():
        out = model(input_tensor)
        pred = out.argmax(dim=1).item()
    return "Positive" if pred == 1 else "Negative"

try:
    model  # check existance
    vocabulary # check existance
except NameError:
    print("Model or vocabulary not found. Please ensure training has completed.")
else:
    while True:
        user_input = input("\nEnter a sentence (or type 'exit'): ")
        if user_input.lower() == 'exit':
            break
        print("Prediction:", predict_text(user_input, model, vocabulary))
