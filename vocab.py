from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# tamanho do vocabulário
print("Tamanho do vocabulário:", tokenizer.vocab_size)

print("\nExemplos do vocabulário:")
count = 0
for token, token_id in tokenizer.vocab.items():
    print(f"{token:15} -> {token_id}")
    count += 1
    if count == 20:
        break
