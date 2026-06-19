import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from openai import OpenAI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "ChoosenModel" / "cancer_model.joblib"
METADATA_PATH = Path(__file__).resolve().parent.parent / "ChoosenModel" / "model_metadata.json"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

_model = None
_metadata: dict = {}
_llm: OpenAI | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _metadata, _llm
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Modelo não encontrado em {MODEL_PATH}. "
            "Execute o projeto 1.GenAIPrediction primeiro para gerar o modelo."
        )
    _model = joblib.load(MODEL_PATH)
    if METADATA_PATH.exists():
        with open(METADATA_PATH, encoding="utf-8") as f:
            _metadata = json.load(f)
    _llm = OpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama", timeout=120.0)

    # Warmup em background: carrega o modelo na memória sem bloquear o startup da API
    async def _warmup():
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _llm.chat.completions.create(
                    model=OLLAMA_MODEL,
                    messages=[{"role": "user", "content": "ok"}],
                    max_tokens=1,
                ),
            )
        except Exception:
            pass

    asyncio.create_task(_warmup())

    yield


app = FastAPI(
    title="Breast Cancer Prediction API",
    description="""
API de predição de **câncer de mama** usando modelo de Regressão Logística
otimizado por Algoritmo Genético (Projeto 1 — GenAIPrediction).

### Como usar
1. Envie os parâmetros clínicos via `POST /predict/breastCancer`
2. Receba o diagnóstico (`Maligno` / `Benigno`) e o grau de confiança
3. Para laudo médico completo gerado por LLM, use `POST /predict/breastCancer/laudo`

### Parâmetros do modelo
Os valores correspondem às **piores medições** registradas nas células do tumor
no exame de imagem (Wisconsin Breast Cancer Dataset).
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:80", "http://localhost"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class BreastCancerInput(BaseModel):
    area_pior: float = Field(
        ...,
        gt=0,
        description="Pior valor de área do tumor (worst area)",
        examples=[880.5],
    )
    textura_pior: float = Field(
        ...,
        gt=0,
        description="Pior valor de textura (desvio padrão da escala de cinza — worst texture)",
        examples=[25.38],
    )
    pontos_concavos_pior: float = Field(
        ...,
        ge=0,
        description="Pior número de pontos côncavos no contorno (worst concave points)",
        examples=[0.2654],
    )
    concavidade_pior: float = Field(
        ...,
        ge=0,
        description="Pior severidade das partes côncavas do contorno (worst concavity)",
        examples=[0.3001],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "area_pior": 1.5,
                    "concavidade_pior": 0.3001,
                    "pontos_concavos_pior": 0.2654,
                    "textura_pior": 25.38 
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    diagnostico: str = Field(description="Resultado da classificação: 'Maligno' ou 'Benigno'")
    confianca: float = Field(description="Grau de confiança da predição (%)")
    classe: int = Field(description="Classe numérica: 1 = Maligno, 0 = Benigno")
    probabilidade_maligno: float = Field(description="Probabilidade bruta de ser maligno (0.0 – 1.0)")
    probabilidade_benigno: float = Field(description="Probabilidade bruta de ser benigno (0.0 – 1.0)")


class LaudoResponse(BaseModel):
    diagnostico: str = Field(description="Resultado da classificação: 'Maligno' ou 'Benigno'")
    confianca: float = Field(description="Grau de confiança da predição (%)")
    classe: int = Field(description="Classe numérica: 1 = Maligno, 0 = Benigno")
    probabilidade_maligno: float = Field(description="Probabilidade bruta de ser maligno (0.0 – 1.0)")
    probabilidade_benigno: float = Field(description="Probabilidade bruta de ser benigno (0.0 – 1.0)")
    laudo: str = Field(description="Laudo médico gerado pela LLM em linguagem clínica")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    llm_available: bool
    ollama_url: str
    ollama_model: str
    model_experiment: str | None
    model_metrics: dict | None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _run_prediction(data: BreastCancerInput) -> PredictionResponse:
    """Executa o classificador e retorna o resultado."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    features = [[
        data.area_pior,
        data.textura_pior,
        data.pontos_concavos_pior,
        data.concavidade_pior,
    ]]

    classe = int(_model.predict(features)[0])
    probabilidades = _model.predict_proba(features)[0]
    prob_benigno = round(float(probabilidades[0]), 4)
    prob_maligno = round(float(probabilidades[1]), 4)
    confianca = round(float(probabilidades[classe]) * 100, 2)

    return PredictionResponse(
        diagnostico="Maligno" if classe == 1 else "Benigno",
        confianca=confianca,
        classe=classe,
        probabilidade_maligno=prob_maligno,
        probabilidade_benigno=prob_benigno,
    )


def _gerar_laudo(data: BreastCancerInput, pred: PredictionResponse) -> str:
    """Chama a LLM local (Ollama) e retorna o laudo médico."""
    if _llm is None:
        raise HTTPException(status_code=503, detail="LLM não inicializada.")

    prompt = f"""Você é um assistente médico em oncologia mamária.
Redija um parecer clínico em português para leitura por um médico especialista.

REGRAS OBRIGATÓRIAS:
- NÃO inclua campos em branco nem placeholders como "Nome:", "Médico:", "CRM:", "Paciente:", "Assinatura:", "Data:" ou similares.
- Escreva apenas o texto corrido do parecer, sem formulários.
- Use **negrito** para destacar termos clínicos importantes.
- Máximo de 200 palavras.

DADOS DO EXAME (Wisconsin Breast Cancer Dataset):
- Área do tumor (pior): {data.area_pior}
- Textura (pior): {data.textura_pior}
- Pontos côncavos (pior): {data.pontos_concavos_pior}
- Concavidade (pior): {data.concavidade_pior}

RESULTADO DO CLASSIFICADOR:
- Diagnóstico: **{pred.diagnostico}** (confiança: {pred.confianca}%)
- P(maligno): {pred.probabilidade_maligno * 100:.1f}% | P(benigno): {pred.probabilidade_benigno * 100:.1f}%

Estruture em quatro parágrafos com os títulos em negrito:
**Achados morfológicos**, **Interpretação**, **Conduta sugerida**, **Observação importante**.
"""

    try:
        response = _llm.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Erro ao comunicar com a LLM ({OLLAMA_URL}): {exc}",
        ) from exc


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.post(
    "/predict/breastCancer",
    response_model=PredictionResponse,
    summary="Predição de câncer de mama",
    tags=["Predição"],
    responses={
        200: {"description": "Predição realizada com sucesso"},
        503: {"description": "Modelo não carregado"},
    },
)
def predict_breast_cancer(data: BreastCancerInput) -> PredictionResponse:
    return _run_prediction(data)


@app.post(
    "/predict/breastCancer/laudo",
    response_model=LaudoResponse,
    summary="Predição + laudo médico gerado por LLM",
    tags=["Predição"],
    responses={
        200: {"description": "Predição e laudo gerados com sucesso"},
        502: {"description": "Erro ao comunicar com a LLM"},
        503: {"description": "Modelo ou LLM não carregados"},
    },
)
def predict_breast_cancer_laudo(data: BreastCancerInput) -> LaudoResponse:
    pred = _run_prediction(data)
    laudo = _gerar_laudo(data, pred)
    return LaudoResponse(
        diagnostico=pred.diagnostico,
        confianca=pred.confianca,
        classe=pred.classe,
        probabilidade_maligno=pred.probabilidade_maligno,
        probabilidade_benigno=pred.probabilidade_benigno,
        laudo=laudo,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Status da API e do modelo",
    tags=["Sistema"],
)
def health() -> HealthResponse:
    llm_ok = _llm is not None
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        llm_available=llm_ok,
        ollama_url=OLLAMA_URL,
        ollama_model=OLLAMA_MODEL,
        model_experiment=_metadata.get("experiment"),
        model_metrics=_metadata.get("metrics"),
    )
