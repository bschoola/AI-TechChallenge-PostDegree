import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "ChoosenModel" / "cancer_model.joblib"
METADATA_PATH = Path(__file__).resolve().parent.parent / "ChoosenModel" / "model_metadata.json"

_model = None
_metadata: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _metadata
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Modelo não encontrado em {MODEL_PATH}. "
            "Execute o projeto 1.GenAIPrediction primeiro para gerar o modelo."
        )
    _model = joblib.load(MODEL_PATH)
    if METADATA_PATH.exists():
        with open(METADATA_PATH, encoding="utf-8") as f:
            _metadata = json.load(f)
    yield


app = FastAPI(
    title="Breast Cancer Prediction API",
    description="""
API de predição de **câncer de mama** usando modelo de Regressão Logística
otimizado por Algoritmo Genético (Projeto 1 — GenAIPrediction).

### Como usar
1. Envie os parâmetros clínicos via `POST /predict/breastCancer`
2. Receba o diagnóstico (`Maligno` / `Benigno`) e o grau de confiança

### Parâmetros do modelo
Os valores correspondem às **piores medições** registradas nas células do tumor
no exame de imagem (Wisconsin Breast Cancer Dataset).
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
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


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_experiment: str | None
    model_metrics: dict | None


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


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Status da API e do modelo",
    tags=["Sistema"],
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        model_experiment=_metadata.get("experiment"),
        model_metrics=_metadata.get("metrics"),
    )
