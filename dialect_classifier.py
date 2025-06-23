# dialect_classifier.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
g
# Label mapping
label_map = {"arg": 0, "col": 1, "mex": 2, "chi": 3}

# Load and label data
files = {
    "arg": "argentinian_spanish_3000.txt",
    "col": "colombian_spanish_3000.txt",
    "mex": "mexican_spanish_3000.txt",
    "chi": "chilean_spanish_3000.txt"
}

all_sentences = []
all_labels = []

for label, fname in files.items():
    with open(fname, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        all_sentences.extend(lines)
        all_labels.extend([label_map[label]] * len(lines))

# Tokenization and vocab
tokens = [sentence.lower().split() for sentence in all_sentences]
vocab = {"<PAD>": 0, "<UNK>": 1}
for line in tokens:
    for tok in line:
        if tok not in vocab:
            vocab[tok] = len(vocab)

# Convert tokens to IDs
token_ids = [[vocab.get(tok, vocab["<UNK>"]) for tok in line] for line in tokens]

# Padding
max_len = max(len(seq) for seq in token_ids)
for i in range(len(token_ids)):
    pad_len = max_len - len(token_ids[i])
    token_ids[i] += [vocab["<PAD>"]] * pad_len

# Tensors
X = torch.tensor(token_ids, dtype=torch.long)
y = torch.tensor(all_labels, dtype=torch.long)

dataset = TensorDataset(X, y)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# Model
class DialectClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<PAD>"])
        self.fc1 = nn.Linear(embed_dim * max_len, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = DialectClassifier(len(vocab), 64, len(label_map))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
best_loss = float("inf")
patience = 5
pat_count = 0

for epoch in range(100):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    total_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = loss_fn(out, yb)
            val_loss += loss.item() * xb.size(0)
            preds = out.argmax(1)
            y_true += yb.tolist()
            y_pred += preds.tolist()

    val_loss /= len(val_loader.dataset)
    acc = sum([1 for a, b in zip(y_true, y_pred) if a == b]) / len(y_true)

    print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.2f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "best_dialect_model.pt")
        pat_count = 0
    else:
        pat_count += 1
        if pat_count >= patience:
            print("Early stopping.")
            break

# Evaluation
model.load_state_dict(torch.load("best_dialect_model.pt"))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        preds = out.argmax(1)
        y_true += yb.tolist()
        y_pred += preds.tolist()

print("\n=== Final Evaluation ===")
print(classification_report(y_true, y_pred, target_names=label_map.keys(), zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=list(label_map.values()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_map.keys()))
disp.plot(cmap=plt.cm.Blues)
plt.title("Dialect Confusion Matrix")
plt.tight_layout()
plt.show()