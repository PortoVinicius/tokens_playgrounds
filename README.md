# Estudos sobre LLMs e Tokens

Este diretório contém programas e experimentos criados com o objetivo de **entender como Large Language Models (LLMs) funcionam**, com foco especial no conceito de **tokens**.

## Objetivo

Os arquivos desta pasta não têm como finalidade criar aplicações finais, mas sim servir como material de estudo para compreender:

- O que são tokens
- Como textos são divididos em tokens
- Como os LLMs processam tokens internamente
- Como a quantidade e a ordem dos tokens influenciam as respostas dos modelos

## O que são tokens?

Tokens são as menores unidades de texto que um LLM consegue processar.  
Eles podem representar:

- Palavras inteiras  
- Partes de palavras  
- Símbolos  
- Pontuação  

Os LLMs não “leem” texto como humanos, mas sim como sequências de tokens que são convertidos em números.

## Estrutura da pasta

Cada programa dentro desta pasta explora algum aspecto do funcionamento de tokens, como:

- Tokenização de textos
- Contagem de tokens
- Comparação entre texto original e tokens gerados
- Simulações simples do funcionamento interno de um LLM

## Observação

Este material é **educacional** e experimental, voltado para aprendizado e exploração dos conceitos fundamentais por trás dos modelos de linguagem.

---

## Comando ideal no seu caso

```bash
pip install -r requirements.txt
```

## Se der problema com torch pesado no PC velho:

```bash
pip install torch --no-cache-dir
```

📚 Ideal para quem está começando a estudar LLMs ou quer aprofundar o entendimento de como os modelos processam texto internamente.
# tokens_playgrounds
