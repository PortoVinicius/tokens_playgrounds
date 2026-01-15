import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModel.from_pretrained("bert-base-multilingual-cased")

def show_embedding(word):
    tokens = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)

    # embedding do primeiro token real
    embedding = outputs.last_hidden_state[0][1]

    print(f"\nPalavra: {word}")
    print("Token ID:", tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)))
    print("Dimensão do vetor:", embedding.shape)
    print("Primeiros 10 valores do embedding:")
    print(embedding[:10])

while True:
    w = input("\nDigite uma palavra (ENTER para sair): ")
    if not w:
        break
    show_embedding(w)
