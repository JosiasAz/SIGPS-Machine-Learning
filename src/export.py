import shutil
import sys
from pathlib import Path
from core.path import CAMINHO_MODELO


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Uso: python -m src.export <caminho_para_sigps-backend>")

    caminho_backend = Path(sys.argv[1]).resolve()

    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError("Modelo não existe. Rode: python -m src.train")

    if not caminho_backend.exists():
        raise FileNotFoundError(f"Backend não encontrado: {caminho_backend}")

    destino = caminho_backend / "app" / "ml" / "model.pkl"
    destino.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(CAMINHO_MODELO, destino)

    print("✅ Exportação concluída")
    print(f"Modelo copiado para: {destino}")

if __name__ == "__main__":
    main()