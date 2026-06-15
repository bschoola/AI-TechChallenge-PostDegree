# Fase 2 — Prediction API

API REST em Python (FastAPI) que carrega o modelo treinado na Fase 1 e expõe um endpoint de predição de câncer de mama com grau de confiança.

---

## Pré-requisito

Execute o projeto [1.GenAIPrediction](../1.GenAIPrediction/) ao menos uma vez para que o modelo seja gerado em `ChoosenModel/cancer_model.joblib`.

---

## Instalação e execução

```bash
cd 2.PredictionApi
pip install -r requirements.txt
uvicorn main:app --reload
```

A API sobe em `http://localhost:8000`.

---

## Documentação interativa

| Interface | URL |
|---|---|
| Swagger UI | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |

---

## Endpoints

### `POST /predict/breastCancer`

Classifica um tumor como **Maligno** ou **Benigno** com base nos parâmetros clínicos.

**Request body:**

```json
{
  "area_pior": 880.5,
  "textura_pior": 25.38,
  "pontos_concavos_pior": 0.2654,
  "concavidade_pior": 0.3001
}
```

| Campo | Tipo | Descrição |
|---|---|---|
| `area_pior` | float > 0 | Pior valor de área do tumor (worst area) |
| `textura_pior` | float > 0 | Pior valor de textura (desvio padrão da escala de cinza) |
| `pontos_concavos_pior` | float ≥ 0 | Pior número de pontos côncavos no contorno |
| `concavidade_pior` | float ≥ 0 | Pior severidade das partes côncavas do contorno |

**Response `200`:**

```json
{
  "diagnostico": "Maligno",
  "confianca": 94.73,
  "classe": 1,
  "probabilidade_maligno": 0.9473,
  "probabilidade_benigno": 0.0527
}
```

| Campo | Descrição |
|---|---|
| `diagnostico` | `"Maligno"` ou `"Benigno"` |
| `confianca` | Grau de confiança da predição em % |
| `classe` | `1` = Maligno, `0` = Benigno |
| `probabilidade_maligno` | Probabilidade bruta de ser maligno (0.0 – 1.0) |
| `probabilidade_benigno` | Probabilidade bruta de ser benigno (0.0 – 1.0) |

**Response `503`:** modelo não carregado (execute o projeto 1 primeiro).

---

### `GET /health`

Verifica o status da API e informa qual experimento/modelo está em uso.

**Response `200`:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_experiment": "Exp3 — Pop. Grande / Mutação Baixa",
  "model_metrics": {
    "f1": 0.9756,
    "recall": 0.9762,
    "accuracy": 0.9737,
    "roc_auc": 0.9961
  }
}
```

---

## Estrutura

```
2.PredictionApi/
├── main.py           # Aplicação FastAPI (endpoints, schemas, carregamento do modelo)
├── requirements.txt
└── PredictionApi.md  # Esta documentação
```

---

## Aviso

Esta API é parte de um projeto acadêmico. Os resultados **não substituem** o diagnóstico médico profissional.
