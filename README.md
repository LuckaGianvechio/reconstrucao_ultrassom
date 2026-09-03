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

1. Ative a env do projeto:

```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
pyenv activate ultrassom
```

2. Instale as dependências do ambiente reproduzível:

```bash
pip install -r requirements-ultrassom.txt
```

3. Se for necessário rodar o serviço FastAPI:

```bash
pip install -r serve/requirements.txt
```

4. Use as pastas `data/` e `artifacts/` como armazenamento local de execução.

## Ambiente reproduzível e MAE customizado

Este projeto usa uma stack fixa e validada na env `ultrassom`, registrada em `requirements-ultrassom.txt`.

Esse arquivo é importante porque as correções aplicadas em `mae/` não são ajustes arbitrários: elas refletem incompatibilidades reais da stack atual, especialmente com:

- `timm` moderno
- `numpy` >= 2
- `torch` recente

Exemplos de compatibilidade necessárias:

- remoção/ajuste de `qk_scale` em `mae/models_mae.py`
- troca de `np.float` por `float` em `mae/util/pos_embed.py`

Isso significa que o diretório `mae/` deve ser tratado como uma dependência customizada, idealmente como fork ou submodule do repositório upstream, e não como um clone genérico sem controle de versão.

## Observação

Os repositórios externos em `mae/` e `USF-MAE/` devem ser tratados como dependências do projeto, preferencialmente em fork/submodule, e não como artefatos gerados dentro do repositório principal.
