# Tech Challenge — OncoLab: Predição de Câncer de Mama com IA

Sistema completo de apoio ao diagnóstico médico composto por três projetos integrados: otimização de modelo com Algoritmo Genético, API de predição em Python e interface clínica em Angular.

---

## Arquitetura

```
┌──────────────────────────────────────────────────────────────────┐
│  1.GenAIPrediction                                               │
│  Algoritmo Genético otimiza hiperparâmetros da Regressão         │
│  Logística → salva o modelo vencedor em ChoosenModel/            │
└──────────────────────────────┬───────────────────────────────────┘
                               │ cancer_model.joblib
┌──────────────────────────────▼───────────────────────────────────┐
│  2.PredictionApi  (FastAPI · localhost:8000)                     │
│  POST /predict/breastCancer                                      │
│  Carrega o modelo → classifica → retorna diagnóstico + confiança │
└──────────────────────────────┬───────────────────────────────────┘
                               │ JSON { diagnostico, confianca, ... }
┌──────────────────────────────▼───────────────────────────────────┐
│  3.Front  (Angular 17 · localhost:4200)                          │
│  Formulário médico → chama a API → exibe resultado ao médico     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Estrutura do Repositório

```
AI-TechChallenge-PostDegree/
│
├── 1.GenAIPrediction/
│   ├── main.py                    # Pipeline principal: treina, compara e salva o melhor modelo
│   ├── config.py                  # Espaço de busca do AG e configurações dos 3 experimentos
│   ├── requirements.txt
│   └── src/
│       ├── data_loader.py         # Carrega o dataset Wisconsin Breast Cancer (CSV remoto)
│       ├── preprocessing.py       # Feature engineering e divisão treino/teste
│       ├── models.py              # Pipeline: StandardScaler + LogisticRegression
│       ├── genetic_algorithm.py   # Seleção, cruzamento, mutação e elitismo
│       └── evaluation.py          # Métricas: F1, Recall, Accuracy, ROC-AUC
│
├── 2.PredictionApi/
│   ├── main.py                    # FastAPI: endpoints, schemas Pydantic, CORS
│   ├── requirements.txt
│   └── PredictionApi.md           # Documentação detalhada da API
│
├── ChoosenModel/                  # Gerado ao rodar o projeto 1
│   ├── cancer_model.joblib        # Pipeline serializado (scaler + modelo fitados)
│   └── model_metadata.json        # Experimento vencedor, hiperparâmetros e métricas
│
└── 3.Front/
    ├── src/
    │   ├── styles.css             # Tema médico global (variáveis CSS)
    │   └── app/
    │       ├── app.component.*    # Formulário de entrada + tela de resultado
    │       └── services/
    │           └── prediction.service.ts  # HttpClient → POST /predict/breastCancer
    └── angular.json / package.json / tsconfig.json
```

---

## Como rodar o projeto completo

> Execute os passos na ordem abaixo. O modelo precisa existir antes de subir a API.

### Passo 1 — Gerar o modelo (Fase 1)

```bash
cd 1.GenAIPrediction
pip install -r requirements.txt
python main.py
```

Ao terminar, a pasta `ChoosenModel/` será criada com `cancer_model.joblib` e `model_metadata.json`.

### Passo 2 — Subir a API (Fase 2)

```bash
cd 2.PredictionApi
pip install -r requirements.txt
uvicorn main:app --reload
```

API disponível em `http://localhost:8000` · Swagger UI em `http://localhost:8000/docs`

### Passo 3 — Subir o frontend (Fase 3)

```bash
cd 3.Front
npm install
ng serve
```

Interface disponível em `http://localhost:4200`

---

## Fase 1 — Otimização com Algoritmo Genético

### Objetivo

Encontrar os melhores hiperparâmetros para um modelo de Regressão Logística que classifica tumores de mama como **Maligno** ou **Benigno** com máximo Recall — minimizando falsos negativos.

### Dataset

- **Fonte:** [Wisconsin Breast Cancer Dataset](https://github.com/bschoola/FIAP-Pos-AI/blob/main/Data/data.csv)
- **Features selecionadas:** `area_pior`, `textura_pior`, `pontos_concavos_pior`, `concavidade_pior`
- **Target:** `diagnostico` → Maligno (1) / Benigno (0)
- **Divisão:** 80% treino / 20% teste, estratificado

### Espaço de busca

| Hiperparâmetro | Valores possíveis |
|---|---|
| `C` (regularização) | 0.001 · 0.01 · 0.1 · 1 · 10 · 100 |
| `penalty` | l1 · l2 |
| `max_iter` | 100 · 200 · 500 · 1000 · 2000 |
| `class_weight` | None · balanced |

O AG representa cada combinação como um **cromossomo de 4 genes** (índices nas listas acima). O **fitness** é o F1-Score médio em cross-validation de 5 folds, avaliado apenas nos dados de treino.

### 3 Experimentos

| | Exp 1 | Exp 2 | Exp 3 |
|---|---|---|---|
| População | 10 | 20 | 30 |
| Gerações | 20 | 15 | 25 |
| Taxa de mutação | 30% | 20% | 10% |
| Taxa de cruzamento | 80% | 80% | 90% |
| Estratégia | Alta exploração | Balanceado | Alta explotação |
| Avaliações totais | 200 | 300 | 750 |

### Métricas

| Métrica | Descrição |
|---|---|
| **Recall** | % de tumores malignos reais detectados — métrica mais crítica clinicamente |
| **F1-Score** | Equilíbrio entre Precision e Recall — usado como fitness do AG |
| **ROC-AUC** | Capacidade de separação entre as duas classes |
| **Accuracy** | Taxa geral de acerto |

### Saída

O experimento com maior F1-Score no conjunto de teste é serializado em:

```
ChoosenModel/
├── cancer_model.joblib     # Pipeline completo pronto para uso em produção
└── model_metadata.json     # { experiment, hyperparameters, features, metrics, trained_at }
```

---

## Fase 2 — API de Predição

Documentação completa: [2.PredictionApi/PredictionApi.md](2.PredictionApi/PredictionApi.md)

### Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Redireciona para `/docs` |
| `POST` | `/predict/breastCancer` | Predição com grau de confiança |
| `GET` | `/health` | Status da API e métricas do modelo carregado |

### Exemplo de requisição

```json
POST /predict/breastCancer
{
  "area_pior": 880.5,
  "textura_pior": 25.38,
  "pontos_concavos_pior": 0.2654,
  "concavidade_pior": 0.3001
}
```

### Exemplo de resposta

```json
{
  "diagnostico": "Maligno",
  "confianca": 94.73,
  "probabilidade_maligno": 0.9473,
  "probabilidade_benigno": 0.0527
}
```

### Stack

- **Framework:** FastAPI + Uvicorn
- **Serialização do modelo:** joblib
- **Validação:** Pydantic v2
- **CORS:** habilitado para `http://localhost:4200`

---

## Fase 3 — Frontend Angular (OncoLab)

### Telas

**Formulário de entrada**
- 4 campos com descrição clínica, hint de valores típicos e validação
- Botão desabilitado até todos os campos estarem preenchidos

**Tela de resultado**
- Badge colorido: **MALIGNO** (vermelho) ou **BENIGNO** (verde)
- Percentual de confiança em destaque
- Barras de probabilidade para cada classe
- Resumo dos parâmetros informados com data/hora
- Disclaimer clínico
- Botão "Nova Análise"

### Stack

- **Framework:** Angular 17 standalone (sem módulos)
- **HTTP:** `HttpClient` via `PredictionService`
- **Estilo:** CSS customizado, tema hospitalar azul, responsivo
- **Fonte:** Inter (Google Fonts)

---

## Aviso

Este sistema é um projeto acadêmico de pós-graduação (FIAP). Os resultados gerados pela IA **não substituem** a avaliação clínica do profissional de saúde e não devem ser usados como única base para decisões diagnósticas.
