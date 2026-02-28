# SIGPS ML Training

Repositório dedicado ao treinamento, validação e exportação do modelo de Machine Learning do SIGPS.

## Objetivo
- Treinar o modelo de priorização (score 0..100)
- Gerar artefatos versionáveis em `artifacts/`
- Exportar `model.pkl` para o `sigps-backend` (somente inferência no backend)

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```



