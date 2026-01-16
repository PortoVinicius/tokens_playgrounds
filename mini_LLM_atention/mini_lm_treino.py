import numpy as np
from collections import defaultdict

np.set_printoptions(precision=3, suppress=True)

# --- 1. Texto de treino ---
texto = "tudo bem tudo certo tudo tranquilo bem tranquilo certo tranquilo bem"
tokens = texto.lower().split()
vocab = list(sorted(set(tokens)))
word2idx = {w:i for i,w in enumerate(vocab)}
idx2word = {i:w for i,w in enumerate(vocab)}

# --- 2. Trigramas ---
trigrams = []
for i in range(len(tokens)-2):
    trigrams.append((tokens[i], tokens[i+1], tokens[i+2]))

# --- 3. Embeddings ---
embedding_dim = 6
vocab_size = len(vocab)
embeddings = np.random.rand(vocab_size, embedding_dim) * 0.1

# --- 4. Função de attention ---
def attention(context_vecs):
    Q = context_vecs[-1]
    K = context_vecs
    V = context_vecs
    scores = Q @ K.T
    exp_scores = np.exp(scores)
    weights = exp_scores / exp_scores.sum()
    out = weights @ V
    return out, weights

# --- 5. Softmax com temperatura ---
def softmax(x, T=1.0):
    x = x / T
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# --- 6. Treinamento simples ---
lr = 0.05
epochs = 200

for epoch in range(epochs):
    loss = 0
    for w1, w2, w3 in trigrams:
        # context embeddings
        context_vecs = np.array([embeddings[word2idx[w1]], embeddings[word2idx[w2]]])
        attn_out, _ = attention(context_vecs)
        
        # logits
        scores = embeddings @ attn_out
        probs = softmax(scores)
        
        # one-hot do alvo
        target = np.zeros(vocab_size)
        target[word2idx[w3]] = 1
        
        # cross-entropy loss
        l = -np.sum(target * np.log(probs + 1e-8))
        loss += l
        
        # gradiente simples (embeddings apenas)
        grad = probs - target  # shape (vocab_size,)
        for i in range(vocab_size):
            embeddings[i] -= lr * grad[i] * attn_out

    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# --- 7. Função de geração ---
def gerar(context, max_tokens=5, T=1.0):
    generated = list(context)
    for _ in range(max_tokens):
        context_vecs = np.array([embeddings[word2idx[w]] for w in generated[-2:]])
        attn_out, weights = attention(context_vecs)
        scores = embeddings @ attn_out
        probs = softmax(scores, T)
        next_idx = np.random.choice(len(vocab), p=probs)
        generated.append(idx2word[next_idx])
        print(f"Context: {generated[-2:]} -> Pesos: {weights} -> Next: {idx2word[next_idx]}")
    return generated

# --- 8. Rodar geração ---
contexto_inicial = ["tudo", "bem"]
saida = gerar(contexto_inicial, max_tokens=10, T=1.0)
print("\nTexto gerado:", " ".join(saida))
