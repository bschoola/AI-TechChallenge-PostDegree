# Tech Challenge — Predição de Câncer de Mama com IA

Sistema completo de apoio ao diagnóstico médico usando GenAI para otimização de modelos, API de predição com laudo gerado por LLM, e frontend Angular para uso clínico.

---

## Visão Geral da Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│  Fase 1 — Pesquisa                                              │
│  GenAI + Algoritmo Genético → Melhor modelo de classificação    │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ modelo treinado / hiperparâmetros
┌─────────────────────────────────▼───────────────────────────────┐
│  Fase 2 — API (FastAPI / Python)                                │
│  Recebe parâmetros clínicos → classifica → LLM gera laudo       │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ laudo técnico (JSON)
┌─────────────────────────────────▼───────────────────────────────┐
│  Fase 3 — Frontend (Angular)                                    │
│  Sistema de consulta médica → exibe laudo ao médico             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Estrutura do Repositório

```
AI-TechChallenge-PostDegree/
├── 1.GenAIPrediction/          # Fase 1 — Otimização do modelo
│   ├── main.py                 # Orquestrador do pipeline
│   ├── config.py               # Hiperparâmetros e configurações
│   ├── requirements.txt
│   └── src/
│       ├── data_loader.py      # Carga dos dados (Wisconsin Breast Cancer)
│       ├── preprocessing.py    # Feature engineering e split treino/teste
│       ├── models.py           # Regressão Logística com Pipeline
│       ├── genetic_algorithm.py # Otimização por algoritmo genético
│       ├── evaluation.py       # Métricas: F1, Recall, ROC-AUC
│       └── explainability.py
├── 2.PredictionAPI/            # Fase 2 — API de predição (a criar)
└── 3.MedicalFrontend/          # Fase 3 — Frontend Angular (a criar)
```

---

## Fase 1 — Otimização com GenAI (Algoritmo Genético)

### Objetivo

Encontrar os melhores hiperparâmetros para um modelo de Regressão Logística que classifica tumores de mama como **malignos (M)** ou **benignos (B)**, usando o dataset Wisconsin Breast Cancer.

### Dataset

- **Fonte:** [Wisconsin Breast Cancer Dataset](https://github.com/bschoola/FIAP-Pos-AI/blob/main/Data/data.csv)
- **Features selecionadas:** `area_pior`, `textura_pior`, `pontos_concavos_pior`, `concavidade_pior`
- **Target:** `diagnostico` → Maligno (1) / Benigno (0)
- **Divisão:** 80% treino / 20% teste, estratificado

### Como funciona

O Algoritmo Genético busca a melhor combinação de hiperparâmetros no espaço:

| Hiperparâmetro | Valores possíveis |
|---|---|
| `C` (regularização) | 0.001, 0.01, 0.1, 1, 10, 100 |
| `penalty` | l1, l2 |
| `max_iter` | 100, 200, 500, 1000, 2000 |
| `class_weight` | None, balanced |

**Cada indivíduo** da população é uma configuração de hiperparâmetros. O **fitness** é o F1-Score em cross-validation de 5 folds. O AG usa seleção por torneio, cruzamento de ponto único e elitismo.

### 3 Experimentos

| | Exp1 | Exp2 | Exp3 |
|---|---|---|---|
| População | 10 | 20 | 30 |
| Gerações | 20 | 15 | 25 |
| Taxa de mutação | 30% | 20% | 10% |
| Estratégia | Alta exploração | Balanceado | Alta explotação |

### Métricas de avaliação

O **Recall** é a métrica mais crítica: um falso negativo significa classificar um tumor maligno como benigno.

| Métrica | Descrição |
|---|---|
| Recall | % de malignos reais detectados corretamente |
| F1-Score | Equilíbrio entre Precision e Recall (fitness do AG) |
| ROC-AUC | Capacidade de separação entre as classes |
| Accuracy | Taxa geral de acerto |

### Como rodar

```bash
cd 1.GenAIPrediction
pip install -r requirements.txt
python main.py
```

---

## Fase 2 — API de Predição com Laudo por LLM

### Objetivo

API REST que recebe os parâmetros clínicos do tumor, aplica o modelo otimizado na Fase 1 e usa uma LLM (Claude) para transformar o resultado em um **laudo técnico** estruturado.

### Fluxo

```
POST /predict
  └─ recebe: { area_pior, textura_pior, pontos_concavos_pior, concavidade_pior }
       └─ modelo classifica → { resultado: "Maligno", probabilidade: 0.94 }
            └─ LLM gera laudo técnico com:
                 • classificação e grau de confiança
                 • interpretação clínica das features
                 • recomendações de conduta
                 • disclaimers legais
```

### Stack

- **Framework:** FastAPI (Python)
- **Modelo:** Regressão Logística treinada com hiperparâmetros da Fase 1
- **LLM:** Claude (Anthropic) via API
- **Formato de resposta:** JSON com campos `classificacao`, `probabilidade`, `laudo`

---

## Fase 3 — Frontend Angular

### Objetivo

Interface clínica para que médicos possam inserir os dados do exame e visualizar o laudo gerado automaticamente.

### Funcionalidades

- Formulário de entrada com os parâmetros do tumor
- Validação dos campos antes do envio
- Exibição do laudo técnico formatado
- Indicador visual de risco (Maligno / Benigno)
- Histórico de consultas (opcional)

### Stack

- **Framework:** Angular 17+
- **UI:** Angular Material
- **HTTP:** `HttpClient` consumindo a API da Fase 2

---

## Aviso

Este sistema é um projeto acadêmico de pós-graduação. Os resultados gerados **não substituem** o diagnóstico médico profissional e não devem ser usados como única base para decisões clínicas.
