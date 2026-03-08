# SIGPS ML Training

Repositório dedicado ao treinamento, validação e exportação do modelo de Machine
Learning do SIGPS.

## Objetivo

- Treinar o modelo de priorização (score 0..100).
- Gerar artefatos versionáveis em `artifacts/`.
- Exportar `model.pkl` para o `sigps-backend` (somente inferência no backend).

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Estrutura do repositório

```
src/
├── dataset.py     – carregamento e preparação dos dados brutos.
├── features.py    – transformação / engenharia de atributos.
├── train.py       – rotina de treinamento (fit do modelo, persistência).
├── evaluate.py    – cálculo de métricas, gráficos e relatórios de validação.
├── export.py      – empacota o modelo treinado em `model.pkl` e gera metadados.
├── __pycache__/   – caches do Python.
artifacts/
└── ...            – pasta onde são armazenados os artefatos versionáveis
README.md          – este guia.
requirements.txt   – dependências do projeto.
```

### Descrição dos módulos

- **dataset.py**  
  Implementa funções para ler os dados de origem (CSV, banco, API...) e
  aplicar limpeza básica. Retorna `pandas.DataFrame` prontos para o
  processamento.

- **features.py**  
  Contém a lógica de engenharia de atributos: codificação de categóricas,
  normalização, agregações etc. É chamado por `train.py` e por `evaluate.py`
  para garantir que os dados são transformados de forma idêntica no treino
  e na inferência.

- **train.py**  
  Executa o pipeline completo de treinamento. Usa `dataset.py` para carregar os
  dados, passa pelo `features.py`, treina um `sklearn` (ou outro framework),
  grava o objeto em `artifacts/` com número de versão e, opcionalmente, salva
  o melhor modelo local para inspeção.

- **evaluate.py**  
  Roda métricas de desempenho (ROC‑AUC, MSE, etc.), plota curvas e gera
  relatórios. Normalmente usado após o treino para validar se o modelo atende
  aos critérios antes de um novo deploy.

- **export.py**  
  A partir do artefato gerado por `train.py`, empacota o modelo em
  `model.pkl` (pickle compatível com o backend) e agrega um arquivo de
  metadados (versão, data, features usadas). É essa saída que será
  consumida pelo repositório do backend.

## Fluxo de trabalho

1. Ative o virtualenv e instale as dependências.
2. Execute `python src/train.py` para treinar e gerar artefatos.
3. (Opcional) `python src/evaluate.py` para ver métricas e validar o modelo.
4. Quando satisfeito, rode `python src/export.py` para criar o `model.pkl`.
5. Copie/commite o `model.pkl` gerado em `artifacts/` e envie para o
   repositório `sigps-backend` ou para o pipeline de CI/CD responsável pelo
   deploy.

## Integração com o backend

O backend não treina nem valida modelos; ele só faz **inferência**. O
procedimento de integração é simples:

1. O `model.pkl` exportado aqui é versionado (ex.: `artifacts/model_v1.2.3.pkl`)
   e disponibilizado ao repositório `sigps-backend` por meio de um submódulo,
   pacote interno, artefato do CI ou caminho de rede.
2. No código do backend, há uma função de inicialização que carrega o modelo:

   ```python
   import pickle

   with open('path/to/model.pkl', 'rb') as f:
       model = pickle.load(f)

   def score(features: dict) -> float:
       X = preprocess(features)  # mesma lógica de features.py, simplificada
       return model.predict_proba(X)[0, 1] * 100
   ```

3. O backend chama essa função sempre que precisa gerar um `score` para um
   novo registro. O preprocess deve replicar os passos de `features.py` para
   garantir consistência.

### Observações

- Não há dependência direta entre este repositório e o backend; a única
  “ponte” é o arquivo serializado `model.pkl`.
- Qualquer alteração na engenharia de features ou no formato de entrada deve
  ser comunicada e sincronizada com o código do backend.

Com essas informações, qualquer desenvolvedor novo consegue entender o que
cada script faz, como rodar o treinamento e como o artefato resultante é
utilizado pelo sistema de produção.



