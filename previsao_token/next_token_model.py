from collections import defaultdict
import random
from collections import Counter

# Texto de treino (pequeno e repetitivo de propósito)
texto = """
tudo bem
tudo certo
tudo tranquilo
tudo errado
"""

# 1. Tokenização simples (palavras)
tokens = texto.lower().split()

# 2. Criar tabela de transições
modelo = defaultdict(list)

for i in range(len(tokens) - 1):
    atual = tokens[i]
    proximo = tokens[i + 1]
    modelo[atual].append(proximo)

# 3. Função de previsão
def prever_proximo(token):
    if token not in modelo:
        return None

    contagem = Counter(modelo[token])
    total = sum(contagem.values())

    print("\nProbabilidades:")
    for palavra, qtd in contagem.items():
        print(f"{palavra}: {qtd / total:.2f}")

    return contagem.most_common(1)[0][0]

# 4. Teste interativo
while True:
    entrada = input("\nDigite um token (ou 'sair'): ").lower()
    if entrada == "sair":
        break

    previsao = prever_proximo(entrada)
    print("Próximo token previsto:", previsao)
