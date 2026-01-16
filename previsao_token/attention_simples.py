import numpy as np

np.set_printoptions(precision=3, suppress=True)

# ---- Tokens fictícios ----
tokens = ["joao", "pegou", "carro", "pai"]

# ---- Embeddings simples (4 tokens, 3 dimensões) ----
embeddings = {
    "joao":  np.array([1.0, 0.0, 0.0]),
    "pegou": np.array([0.8, 0.2, 0.0]),
    "carro": np.array([0.0, 1.0, 0.2]),
    "pai":   np.array([0.0, 0.9, 0.1]),
}

# ---- Matriz de projeção (fixa para simplicidade) ----
W = np.array([
    [1.0, 0.2, 0.0],
    [0.0, 1.0, 0.1],
    [0.1, 0.0, 1.0]
])

# ---- Escolha do token atual (ex: "carro") ----
token_atual = "pai"

Q = embeddings[token_atual] @ W

print(f"\nToken atual: {token_atual}")
print("Query (Q):", Q)

# ---- Calcula attention ----
scores = []
values = []

for t in tokens:
    K = embeddings[t] @ W
    V = embeddings[t]
    score = Q @ K  # produto escalar
    scores.append(score)
    values.append(V)

scores = np.array(scores)
values = np.array(values)

# ---- Softmax ----
exp_scores = np.exp(scores)
attention_weights = exp_scores / exp_scores.sum()

print("\nPesos de attention:")
for t, w in zip(tokens, attention_weights):
    print(f"{t}: {w:.3f}")

# ---- Vetor final ----
output = attention_weights @ values

print("\nVetor de saída (contexto):")
print(output)
