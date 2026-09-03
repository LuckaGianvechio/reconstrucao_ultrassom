# Reconstrução de Ultrassom

Este repositório reúne o código para processamento, extração de embeddings, treinamento/evaluação e inferência do fluxo de reconstrução ultrassônica.

## Estrutura proposta

- `src/` — código-fonte do projeto
  - `src/preprocessing/` — pré-processamento de imagens e dados
  - `src/models/` — modelos, treino e experimentos
  - `src/inference/` — código de inferência e serve
  - `src/evaluation/` — métricas e avaliação
- `serve/` — serviço FastAPI/serving
- `notebooks/` — notebooks de exploração e experimentação
- `configs/` — arquivos de configuração
- `data/` — dados locais, organizados em raw/processed
- `artifacts/` — checkpoints e artefatos gerados
- `docs/` — documentação adicional

## Regras de versionamento

O repositório deve versionar apenas:
- código-fonte
- documentação
- configs
- scripts de automação
- notebooks limpos e sem outputs pesados

Não devem entrar no Git:
- dados brutos e processados pesados
- checkpoints e pesos de modelos
- embeddings intermediários
- imagens/plots gerados em execução
- ambientes virtuais (`.venv`)

## Setup rápido

1. Crie um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Instale dependências:

```bash
pip install -r serve/requirements.txt
```

3. Use as pastas `data/` e `artifacts/` como armazenamento local de execução.

## Observação

Os repositórios externos em `mae/` e `USF-MAE/` devem ser tratados como dependências e, preferencialmente, mantidos em submodulos ou fora do controle principal de artefatos gerados.
