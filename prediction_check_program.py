import torch
import torch.nn as nn
import re
from tkinter import Tk, filedialog

# ---------------- Load Vocab and Constants ----------------
vocab = torch.load("vocab.pt")
vocab_size = len(vocab)
pad_id = vocab["<PAD>"]
embed_dim = 64
num_classes = 4

state_dict = torch.load("dialect_classifier.pt", map_location="cpu")
fc1_input_dim = state_dict["fc1.weight"].shape[1]
max_len = fc1_input_dim // embed_dim

# ---------------- Model Definition ----------------
class DialectClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, max_len, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.fc1 = nn.Linear(embed_dim * max_len, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ---------------- Load Model ----------------
model = DialectClassifier(vocab_size, embed_dim, num_classes, max_len, pad_id)
model.load_state_dict(state_dict)
model.eval()

label_map = {0: "ARG", 1: "CHL", 2: "COL", 3: "MEX"}

def predict(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(ids) < max_len:
        ids += [pad_id] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    input_tensor = torch.tensor([ids])
    with torch.no_grad():
        logits = model(input_tensor)
        pred = torch.argmax(logits, dim=1).item()
    return label_map[pred]

# ---------------- Upload and Predict ----------------
def upload_and_predict():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not file_path:
        return

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        print(predict(line))

upload_and_predict()