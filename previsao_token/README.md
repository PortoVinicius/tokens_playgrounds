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