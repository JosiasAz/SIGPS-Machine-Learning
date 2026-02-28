import os
import json
from core.path import CAMINHO_AVALIACAO, CAMINHO_MODELO
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

from .dataset import criar_dataset

def main():
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError("Modelo não encontrado. Rode: python -m src.train")

    modelo = joblib.load(CAMINHO_MODELO)

    caminho_csv = os.getenv("SIGPS_DATASET_CSV", "data/raw/sigps_dataset.csv")
    X, y = criar_dataset(caminho_csv)
    pred = modelo.predict(X)

    mae = float(mean_absolute_error(y, pred))
    r2 = float(r2_score(y, pred))

    payload = {"mae": mae, "r2": r2, "n_eval": int(len(X))}
    CAMINHO_AVALIACAO.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("✅ Avaliação concluída")
    print(f"Avaliação salva em: {CAMINHO_AVALIACAO}")
    print("MAE:", mae, "R²:", r2)

if __name__ == "__main__":
    main()