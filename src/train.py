import os
import json
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from core.path import CAMINHO_MODELO, CAMINHO_METRICAS, PASTA_ARTIFACTS
from .dataset import criar_dataset
from .features import ORDEM_FEATURES

def main():
    # 1) cria pasta artifacts/
    PASTA_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # 2) carrega dados
    caminho_csv = os.getenv("SIGPS_DATASET_CSV", r"data\raw\sigps_dataset.csv")
    X, y = criar_dataset(caminho_csv)

    # 3) split treino/teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) modelo
    modelo = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    # 5) treino
    modelo.fit(X_treino, y_treino)

    # 6) predição
    pred = modelo.predict(X_teste)

    # 7) métricas
    mae = float(mean_absolute_error(y_teste, pred))
    r2 = float(r2_score(y_teste, pred))

    # 8) salva modelo
    joblib.dump(modelo, CAMINHO_MODELO)

    # 9) salva métricas + contrato
    payload = {
        "mae": mae,
        "r2": r2,
        "ordem_features": ORDEM_FEATURES,
        "n_treino": int(len(X_treino)),
        "n_teste": int(len(X_teste)),
        "modelo": type(modelo).__name__,
    }
    CAMINHO_METRICAS.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("✅ Treino concluído")
    print(f"Modelo salvo em: {CAMINHO_MODELO}")
    print(f"Métricas salvas em: {CAMINHO_METRICAS}")
    print("MAE:", mae, "R²:", r2)

if __name__ == "__main__":
    main()