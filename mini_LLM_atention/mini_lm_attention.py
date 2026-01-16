import numpy as np
from collections import defaultdict

np.set_printoptions(precision=3, suppress=True)

# --- 1. Texto de treino ---
texto = "tudo bem tudo certo tudo tranquilo bem tranquilo certo tranquilo bem"

# --- 2. Tokenização ---
tokens = texto.lower().split()
vocab = list(sorted(set(tokens)))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}

# --- 3. Construção de trigramas ---
trigrams = []
for i in range(len(tokens)-2):
    trigrams.append((tokens[i], tokens[i+1], tokens[i+2]))

# --- 4. Embeddings simples ---
embedding_dim = 4
embeddings = np.random.rand(len(vocab), embedding_dim)

# --- 5. Função de attention ---
def attention(context_vecs):
    # context_vecs: array de shape (2, embedding_dim)
    Q = context_vecs[-1]      # token atual
    K = context_vecs          # todos os tokens do contexto
    V = context_vecs
    scores = Q @ K.T
    exp_scores = np.exp(scores)
    weights = exp_scores / exp_scores.sum()
    out = weights @ V
    return out, weights

# --- 6. Função de softmax com temperatura ---
def softmax(x, T=1.0):
    x = x / T
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# --- 7. Geração ---
def gerar(context, max_tokens=5, T=1.0):
    generated = list(context)
    for _ in range(max_tokens):
        context_vecs = np.array([embeddings[word2idx[w]] for w in generated[-2:]])
        attn_out, weights = attention(context_vecs)
        # calcular scores simples: dot com todos os embeddings
        scores = embeddings @ attn_out
        probs = softmax(scores, T)
        next_idx = np.random.choice(len(vocab), p=probs)
        next_word = idx2word[next_idx]
        generated.append(next_word)
        print(f"Context: {generated[-2:]} -> Pesos: {weights} -> Next: {next_word}")
    return generated

# --- 8. Rodar exemplo ---
contexto_inicial = ["tudo", "bem"]
saida = gerar(contexto_inicial, max_tokens=10, T=1.0)
print("\nTexto gerado:", " ".join(saida))
