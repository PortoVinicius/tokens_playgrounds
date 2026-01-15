import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModel.from_pretrained("bert-base-multilingual-cased")

def embedding(word):
    tokens = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state[0][1]

def compare(w1, w2):
    e1 = embedding(w1)
    e2 = embedding(w2)
    sim = cosine_similarity(e1, e2, dim=0).item()
    print(f"\nSimilaridade entre '{w1}' e '{w2}': {sim:.4f}")

while True:
    w1 = input("\nPalavra 1 (ENTER para sair): ")
    if not w1:
        break
    w2 = input("Palavra 2: ")
    compare(w1, w2)
