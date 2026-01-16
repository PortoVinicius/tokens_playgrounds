## Como funciona na prática

1. Cada trigram é usado para treinar implicitamente relações (não há treino real, só embeddings aleatórios)

2. Attention olha para os últimos 2 tokens

3. Softmax + temperatura decide probabilidades

4. Amostragem escolhe o próximo token

5. Você vê vetores, pesos e palavras na hora

## Rodando
```bash
python mini_lm_attention.py
```

Você vai ver:

```bash
Context: ['bem', 'tudo'] -> Pesos: [0.45 0.55] -> Next: tranquilo
Context: ['tudo', 'tranquilo'] -> Pesos: [0.52 0.48] -> Next: bem
...
Texto gerado: tudo bem tranquilo ...
```

## test para mini_lm_attention.py

Alterar temperatura para 0.5 ou 1.5 → comportamento muda
Alterar embedding_dim → atenção fica diferente
Alterar contexto inicial → gera frases novas


## test para mini_lm_treino

Alterar T → temperatura diferente muda criatividade
Alterar embedding_dim → vetores mais ricos ou pobres
Alterar contexto_inicial → inicia frase diferente
Aumentar epochs → embeddings aprendem melhor
