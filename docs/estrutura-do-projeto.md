# Organização do projeto

## Objetivo

Separar código reprodutível de artefatos pesados e outputs executados, mantendo o repositório Git limpo e fácil de manter.

## Diretrizes

### Versionar
- código Python
- notebooks versionados, sem outputs pesados
- README e docs
- scripts e configs

### Não versionar
- dados brutos e processados grandes
- checkpoints
- embeddings calculados
- imagens / plots / debug
- `.venv`

## Estrutura final sugerida

```text
reconstrucao_ultrassom/
├── README.md
├── .gitignore
├── .env.example
├── requirements.txt
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── inference/
│   └── evaluation/
├── serve/
├── notebooks/
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   ├── checkpoints/
│   ├── embeddings/
│   └── reports/
├── docs/
├── mae/
├── USF-MAE/
└── .venv/
```

> Observação: `mae/` e `USF-MAE/` devem ser tratados como dependências externas ou submodulos, não como dados de execução.
