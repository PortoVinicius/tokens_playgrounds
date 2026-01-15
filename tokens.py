from transformers import AutoTokenizer

# tokenizer de um modelo real
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

while True:
    text = input("\nDigite um texto (ou ENTER para sair): ")
    if not text:
        break

    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    print("\nTokens:")
    for t in tokens:
        print(f"  {t}")

    print("\nIDs numéricos:")
    print(token_ids)

    print("\nTotal de tokens:", len(tokens))
