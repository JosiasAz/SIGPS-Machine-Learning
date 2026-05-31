# Docker — SIGPS Machine Learning

| Item | Valor |
|------|--------|
| **Imagem** | `sigps-ml` |
| **Container** | `sigps-ml` |
| **Porta** | `8000` |

## Subir só o ML

```bash
docker compose up -d --build
docker compose -f docker-compose.prod.yml up -d --build
```

## Build da imagem

```bash
docker build -t sigps-ml:latest .
```

Coloque o modelo em `artifacts/model.pkl` antes do build (opcional — sem ele usa fallback por regras).

## Health

```bash
curl http://localhost:8000/health
```

O backend acessa via rede Docker: `http://sigps-ml:8000/predict`
