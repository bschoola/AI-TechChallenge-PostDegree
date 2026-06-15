# 🎗️ Predictive Analysis of Cancer — FIAP Tech Challenge

Classificação de tumores como benignos ou malignos com base no [Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data), utilizando Regressão Logística com otimização de hiperparâmetros via **Algoritmo Genético**.


## 👥 Team

| Name |
|---|
| [Bruno Gouveia Schoola](https://github.com/bschoola) |
| [Ricardo Stebulaitis](https://github.com/stebulaitis) |


## 🎯 Objetivo

Apoiar profissionais de saúde no diagnóstico de câncer de mama com um modelo de classificação supervisionada. O sistema classifica tumores como **benignos (B)** ou **malignos (M)** e aplica um Algoritmo Genético para encontrar a combinação ideal de hiperparâmetros da Regressão Logística, maximizando o **F1-Score** — métrica central para equilibrar Precisão e Recall em contexto clínico.


## 🗂️ Estrutura do projeto

```
├── main.py                  # Entry point — executa o pipeline completo
├── config.py                # Constantes, espaço de busca e configurações do AG
├── requirements.txt
└── src/
    ├── data_loader.py       # Carregamento e renomeação de colunas
    ├── preprocessing.py     # Seleção de features e divisão treino/teste
    ├── models.py            # Regressão Logística (baseline e otimizada)
    ├── evaluation.py        # Métricas e cross-validation
    ├── genetic_algorithm.py # Algoritmo Genético para otimização de hiperparâmetros
    └── explainability.py    # Análise SHAP e importância de features
```

## 🚀 Como executar

```bash
# Crie e ative o ambiente virtual
python -m venv .venv
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # Linux/macOS

pip install -r requirements.txt
python main.py
```

## 🤖 Modelo

| Modelo | Observação |
|---|---|
| Logistic Regression | Melhor desempenho em F1 e Recall; pipeline com StandardScaler |

## 🔬 Features selecionadas

`area_pior`, `textura_pior`, `pontos_concavos_pior`, `concavidade_pior`

## 🧬 Algoritmo Genético

O AG busca os melhores hiperparâmetros da Regressão Logística sem depender de bibliotecas externas — implementado from-scratch com `numpy` e `random`.

### Codificação dos genes

Cada indivíduo é representado por 4 genes (índices inteiros):

| Gene | Hiperparâmetro | Valores possíveis |
|---|---|---|
| 0 | `C` (regularização) | 0.001, 0.01, 0.1, 1, 10, 100 |
| 1 | `penalty` | l1, l2 |
| 2 | `max_iter` | 100, 200, 500, 1000, 2000 |
| 3 | `class_weight` | None, balanced |

> `penalty=l1` usa `solver=liblinear`; `penalty=l2` usa `solver=lbfgs` (compatibilidade automática).

### Operadores implementados

| Operador | Estratégia |
|---|---|
| **Seleção** | Torneio (tournament_size=3) |
| **Cruzamento** | Ponto único com taxa configurável |
| **Mutação** | Por gene com probabilidade configurável |
| **Elitismo** | Melhor indivíduo preservado a cada geração |

### Função fitness

F1-Score médio em cross-validation (`cv=5`) sobre os dados de **treino** — o conjunto de teste não é tocado durante a busca, evitando vazamento de dados.

### Experimentos

Três configurações diferentes do AG são executadas e comparadas:

| Experimento | Pop. | Gerações | Mutação | Cruzamento |
|---|---|---|---|---|
| Exp1 — Pop. Pequena / Mutação Alta  | 10 | 20 | 0.30 | 0.80 |
| Exp2 — Pop. Média / Mutação Média   | 20 | 15 | 0.20 | 0.80 |
| Exp3 — Pop. Grande / Mutação Baixa  | 30 | 25 | 0.10 | 0.90 |

## 📊 Métricas de avaliação

- **Accuracy** — proporção de predições corretas
- **Precision** — dos classificados como malignos, quantos realmente eram
- **Recall** — dos malignos reais, quantos foram identificados
- **F1-Score** — equilíbrio entre Precision e Recall *(métrica principal do AG)*
- **ROC-AUC** — capacidade de separação entre classes

O **Recall** é a métrica mais crítica neste contexto: um Falso Negativo significa não detectar um tumor maligno, impedindo o início do tratamento. O **F1-Score** é usado como fitness do AG por equilibrar Precision e Recall simultaneamente.

## 📓 Explicabilidade

A análise SHAP (via `LinearExplainer`) pode ser aplicada sobre o modelo otimizado para identificar o impacto de cada feature nas predições. A feature `area_pior` apresenta o maior peso no modelo.
