import numpy as np
import pandas as pd

from .features import ORDEM_FEATURES

def criar_dataset(
    caminho_csv: str = "data/raw/sigps_prod_fake.csv",
    usar_heuristica_se_nao_tiver_y: bool = False,
):
    """
    Carrega dataset real (simulado) de um CSV e devolve:
      X: (n,4) na ordem ORDEM_FEATURES
      y: (n,) em 0..100

    Requisitos:
      CSV deve conter colunas ORDEM_FEATURES.
      Alvo esperado: score_prioridade (0..100).
    """
    df = pd.read_csv(caminho_csv)

    # 1) valida features
    faltando = [c for c in ORDEM_FEATURES if c not in df.columns]
    if faltando:
        raise ValueError(f"CSV não possui colunas obrigatórias: {faltando}")

    # 2) monta X na ordem do contrato
    X = df[ORDEM_FEATURES].to_numpy(dtype=np.float32)

    # 3) monta y
    if "score_prioridade" in df.columns:
        y = df["score_prioridade"].to_numpy(dtype=np.float32)
    else:
        if not usar_heuristica_se_nao_tiver_y:
            raise ValueError(
                "CSV não tem 'score_prioridade'. "
                "Adicione essa coluna ou chame criar_dataset(..., usar_heuristica_se_nao_tiver_y=True)."
            )

        # Heurística fallback (igual MVP) — útil para “dataset realista sem rótulo”
        urg = X[:, 0]
        hist_cons = X[:, 1]
        hist_falt = X[:, 2]
        min_fila = X[:, 3]
        raw = urg * 18.0 + min_fila * 0.20 + hist_cons * 0.25 - hist_falt * 6.0
        y = np.clip(raw, 0, 100).astype(np.float32)

    # 4) sanidade do y
    y = np.clip(y, 0, 100).astype(np.float32)

    # 5) garantias
    assert X.shape[1] == 4
    assert y.shape == (X.shape[0],)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 100.0

    return X, y