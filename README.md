# Tech Challenge — OncoLab: Predição de Câncer de Mama com IA

Sistema completo de apoio ao diagnóstico médico composto por três projetos integrados: otimização de modelo com Algoritmo Genético, API de predição em Python com geração de laudo via LLM local, e interface clínica em Angular.

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
│  POST /predict/breastCancer       → diagnóstico + confiança      │
│  POST /predict/breastCancer/laudo → diagnóstico + laudo (LLM)    │
└──────────────────────────────┬───────────────────────────────────┘
                               │                    │
                    JSON result│          LLM prompt│
                               │    ┌───────────────▼──────────────┐
                               │    │  Ollama  (localhost:11434)   │
                               │    │  Modelo: llama3.2:3b         │
                               │    │  Gera laudo médico em PT-BR  │
                               │    └──────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────┐
│  3.Front  (Angular 17 · localhost:4200)                          │
│  Formulário → classificação imediata → laudo gerado por LLM      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Estrutura do Repositório

```
AI-TechChallenge-PostDegree/
│
├── 1.GenAIPrediction/
│   ├── main.py                    # Pipeline: treina, compara e salva o melhor modelo
│   ├── config.py                  # Espaço de busca do AG e configurações dos experimentos
│   ├── requirements.txt
│   └── src/
│       ├── data_loader.py         # Carrega o dataset Wisconsin Breast Cancer
│       ├── preprocessing.py       # Feature engineering e divisão treino/teste
│       ├── models.py              # Pipeline: StandardScaler + LogisticRegression
│       ├── genetic_algorithm.py   # Seleção, cruzamento, mutação e elitismo
│       └── evaluation.py          # Métricas: F1, Recall, Accuracy, ROC-AUC
│
├── 2.PredictionApi/
│   ├── main.py                    # FastAPI: endpoints, schemas Pydantic, CORS, integração Ollama
│   ├── requirements.txt
│   ├── Dockerfile
│   └── PredictionApi.md           # Documentação detalhada da API
│
├── 3.Front/
│   ├── src/app/
│   │   ├── app.component.*        # Formulário + resultado + bloco de laudo
│   │   └── services/
│   │       └── prediction.service.ts
│   ├── nginx.conf                 # Config nginx para container Docker
│   └── Dockerfile
│
├── ChoosenModel/                  # Gerado ao rodar o projeto 1
│   ├── cancer_model.joblib
│   └── model_metadata.json
│
├── docker-compose.yml             # Orquestra: ollama + api + front
├── start.sh                       # Script de inicialização (Linux/Mac)
└── start.ps1                      # Script de inicialização (Windows)
```

---

## Como rodar — Docker (recomendado)

> Requer apenas **Docker Desktop** instalado e em execução. Nenhuma outra dependência.

### Opção A — Script interativo (com progresso visível)

O script sobe cada serviço em ordem, aguarda o health check de cada um e exibe o status em tempo real.

**Windows (PowerShell):**
```powershell
.\start.ps1
```

**Linux / Mac:**
```bash
chmod +x start.sh
./start.sh
```

Na primeira execução o modelo `llama3.2:3b` (~2 GB) será baixado automaticamente com barra de progresso. Nas execuções seguintes o modelo já estará em cache no volume Docker.

### Opção B — docker compose direto

```bash
docker compose up --build
```

> O modelo é baixado em background pelo serviço `ollama-setup`. Pode parecer parado por alguns minutos durante o download — use `docker compose logs -f` em outro terminal para acompanhar.

### URLs após inicialização

| Serviço | URL |
|---|---|
| Frontend | http://localhost:4200 |
| API (Swagger) | http://localhost:8000/docs |
| Ollama | http://localhost:11434 |

### Parar todos os serviços

```bash
docker compose down
```

---

## Como rodar — Localmente (sem Docker)

> Siga os passos na ordem. O modelo precisa existir antes de subir a API.

### Pré-requisitos

- Python 3.10+
- Node.js 20+ e Angular CLI (`npm install -g @angular/cli`)
- [Ollama](https://ollama.com/download) instalado localmente

### Passo 1 — Gerar o modelo

```bash
cd 1.GenAIPrediction
pip install -r requirements.txt
python main.py
```

Ao terminar, a pasta `ChoosenModel/` será criada com `cancer_model.joblib` e `model_metadata.json`.

### Passo 2 — Iniciar o Ollama e baixar o modelo

```bash
ollama serve                    # inicia o servidor (porta 11434)
ollama pull llama3.2:3b         # baixa o modelo (~2 GB, só na primeira vez)
```

### Passo 3 — Subir a API

```bash
cd 2.PredictionApi
pip install -r requirements.txt
uvicorn main:app --reload
```

API disponível em `http://localhost:8000` · Swagger UI em `http://localhost:8000/docs`

### Passo 4 — Subir o frontend

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

O AG representa cada combinação como um **cromossomo de 4 genes**. O **fitness** é o F1-Score médio em cross-validation de 5 folds.

### 3 Experimentos

| | Exp 1 | Exp 2 | Exp 3 |
|---|---|---|---|
| População | 10 | 20 | 30 |
| Gerações | 20 | 15 | 25 |
| Taxa de mutação | 30% | 20% | 10% |
| Taxa de cruzamento | 80% | 80% | 90% |
| Estratégia | Alta exploração | Balanceado | Alta explotação |

### Métricas

| Métrica | Descrição |
|---|---|
| **Recall** | % de tumores malignos reais detectados — métrica mais crítica clinicamente |
| **F1-Score** | Equilíbrio entre Precision e Recall — usado como fitness do AG |
| **ROC-AUC** | Capacidade de separação entre as duas classes |
| **Accuracy** | Taxa geral de acerto |

---

## Fase 2 — API de Predição + LLM

Documentação completa: [2.PredictionApi/PredictionApi.md](2.PredictionApi/PredictionApi.md)

### Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Redireciona para `/docs` |
| `POST` | `/predict/breastCancer` | Predição rápida (classificador) |
| `POST` | `/predict/breastCancer/laudo` | Predição + laudo médico gerado por LLM |
| `GET` | `/health` | Status da API, modelo e conexão com Ollama |

### Exemplo — predição simples

```json
POST /predict/breastCancer
{
  "area_pior": 880.5,
  "textura_pior": 25.38,
  "pontos_concavos_pior": 0.2654,
  "concavidade_pior": 0.3001
}
```

```json
{
  "diagnostico": "Maligno",
  "confianca": 94.73,
  "classe": 1,
  "probabilidade_maligno": 0.9473,
  "probabilidade_benigno": 0.0527
}
```

### Exemplo — predição com laudo

```json
POST /predict/breastCancer/laudo
{ ...mesmos campos... }
```

```json
{
  "diagnostico": "Maligno",
  "confianca": 94.73,
  "classe": 1,
  "probabilidade_maligno": 0.9473,
  "probabilidade_benigno": 0.0527,
  "laudo": "LAUDO DE APOIO AO DIAGNÓSTICO\n\nAchados morfológicos: Os parâmetros ..."
}
```

### Stack

- **Framework:** FastAPI + Uvicorn
- **Serialização do modelo:** joblib
- **Validação:** Pydantic v2
- **LLM:** Ollama via cliente OpenAI-compatível (`openai` Python SDK)
- **Modelo LLM:** llama3.2:3b (roda localmente, sem dependência de nuvem)

---

## Fase 3 — Frontend Angular (OncoLab)

### Fluxo de uso

1. Médico preenche os 4 parâmetros do exame de imagem
2. O resultado do classificador aparece imediatamente (rápido)
3. O laudo médico em linguagem clínica é gerado pela LLM e exibido logo em seguida

### Telas

**Formulário de entrada** — 4 campos com descrição clínica, hint de valores típicos e validação

**Tela de resultado**
- Badge colorido: **MALIGNO** (vermelho) ou **BENIGNO** (verde) com percentual de confiança
- Barras de probabilidade para cada classe
- Bloco de laudo médico gerado por IA com indicador de carregamento

### Stack

- **Framework:** Angular 17 standalone
- **HTTP:** `HttpClient` via `PredictionService`
- **Estilo:** CSS customizado, tema hospitalar azul, responsivo

---

## Aviso

Este sistema é um projeto acadêmico de pós-graduação (FIAP). Os resultados gerados pela IA **não substituem** a avaliação clínica do profissional de saúde e não devem ser usados como única base para decisões diagnósticas.
