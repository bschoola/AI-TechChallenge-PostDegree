DATA_URL = (
    "https://raw.githubusercontent.com/bschoola/FIAP-Pos-AI"
    "/refs/heads/main/Data/data.csv"
)

COLUMN_RENAME_MAP = {
    "diagnosis": "diagnostico",
    "radius_mean": "raio_medio",
    "texture_mean": "textura_media",
    "perimeter_mean": "perimetro_medio",
    "area_mean": "area_media",
    "smoothness_mean": "suavidade_media",
    "compactness_mean": "compacidade_media",
    "concavity_mean": "concavidade_media",
    "concave points_mean": "pontos_concavos_media",
    "symmetry_mean": "simetria_media",
    "fractal_dimension_mean": "dimensao_fractal_media",
    "radius_se": "raio_erro_padrao",
    "texture_se": "textura_erro_padrao",
    "perimeter_se": "perimetro_erro_padrao",
    "area_se": "area_erro_padrao",
    "smoothness_se": "suavidade_erro_padrao",
    "compactness_se": "compacidade_erro_padrao",
    "concavity_se": "concavidade_erro_padrao",
    "concave points_se": "pontos_concavos_erro_padrao",
    "symmetry_se": "simetria_erro_padrao",
    "fractal_dimension_se": "dimensao_fractal_erro_padrao",
    "radius_worst": "raio_pior",
    "texture_worst": "textura_pior",
    "perimeter_worst": "perimetro_pior",
    "area_worst": "area_pior",
    "smoothness_worst": "suavidade_pior",
    "compactness_worst": "compacidade_pior",
    "concavity_worst": "concavidade_pior",
    "concave points_worst": "pontos_concavos_pior",
    "symmetry_worst": "simetria_pior",
    "fractal_dimension_worst": "dimensao_fractal_pior",
}

SELECTED_FEATURES = [
    "area_pior",
    "textura_pior",
    "pontos_concavos_pior",
    "concavidade_pior",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 10
