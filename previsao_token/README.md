##  Objetivo:
Entender como a previsão do próximo token funciona
sem redes neurais.

Ideia:
Contar quais palavras costumam vir depois de outras.

Aprendizado:
Mesmo sem "inteligência", padrões emergem.

## 👉 escrever um script que:
recebe um texto de treino
constrói uma tabela:

```bash
palavra_atual -> {proxima_palavra: contagem}
```

## dado um token, prevê o próximo

- uma LLM prevê o próximo token
- tokens podem ser palavras ou números
- probabilidades surgem de repetição
- não existe "pensamento", só estatística

next_token_model.py ==> “cérebro primitivo”

## Pipeline real:

- tokens → embeddings → camadas → logits → softmax → probabilidades

## Pipeline final (mental)

scores
 ↓
divisão por temperatura
 ↓
softmax
 ↓
filtragem (top-k / top-p)
 ↓
amostragem

## Um embedding é:

um vetor de números em um espaço de alta dimensão

```bash

"tudo"       → [ 0.2, -0.4,  0.9 ]
"bem"        → [ 0.21, -0.38, 0.88 ]
"carro"      → [ -0.7, 0.1, -0.2 ]
```

Você já vê:

"tudo" perto de "bem"
"carro" longe

```bash
texto
 ↓
tokenização
 ↓
índices
 ↓
embeddings (geometria)
 ↓
transformações lineares
 ↓
scores
 ↓
softmax + temperatura
 ↓
amostragem
```

## Cada token vira três vetores:

Query (Q) → o que eu estou procurando
Key (K) → o que eu tenho
Value (V) → a informação que eu entrego