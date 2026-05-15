# Predictive Analysis of Cancer — FIAP Tech Challenge

Classificação de tumores como benignos ou malignos com base no [Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data), utilizando quatro modelos de machine learning: Logistic Regression, Random Forest, Gradient Boosting e KNN.

## Estrutura do projeto

```
├── main.py                  # Entry point — executa o pipeline completo
├── config.py                # Constantes centralizadas (URL, features, hiperparâmetros)
├── requirements.txt
└── src/
    ├── data_loader.py       # Carregamento e renomeação de colunas
    ├── preprocessing.py     # Seleção de features e divisão treino/teste
    ├── models.py            # Definição dos 4 modelos
    ├── evaluation.py        # Métricas e cross-validation
    └── explainability.py    # Análise SHAP e importância de features
```

## Como executar

```bash
# Crie e ative o ambiente virtual
python -m venv .venv
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # Linux/macOS

pip install -r requirements.txt
python main.py
```

## Modelos avaliados

| Modelo | Observação |
|---|---|
| Logistic Regression | Melhor desempenho em Recall e F1 na cross-validation |
| Random Forest | Baseado em árvores com 200 estimadores |
| Gradient Boosting | Boosting sequencial com sklearn |
| KNN | Classificação por proximidade (k=5), requer normalização |

## Features selecionadas

`area_pior`, `textura_pior`, `pontos_concavos_pior`, `concavidade_pior`

## Métricas de avaliação

- **Accuracy** — proporção de predições corretas
- **Precision** — dos classificados como malignos, quantos realmente eram
- **Recall** — dos malignos reais, quantos foram identificados *(métrica principal)*
- **F1-Score** — equilíbrio entre Precision e Recall
- **ROC-AUC** — capacidade de separação entre classes

O **Recall** é a métrica mais crítica neste contexto: um Falso Negativo significa não detectar um tumor maligno, impedindo o início do tratamento.

## Explicabilidade

A análise SHAP (via `LinearExplainer`) é aplicada sobre a Logistic Regression para identificar o impacto de cada feature nas predições. A feature `area_pior` apresenta o maior peso no modelo.
