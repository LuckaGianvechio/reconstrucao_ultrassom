# Onboarding — Predição de embeddings a partir de dados tabulares (BrEaST)

## Objetivo do projeto

Este repositório usa o autoencoder **USF-MAE** (um Masked Autoencoder pré-treinado em imagens de ultrassom) para gerar embeddings de imagens do dataset **BrEaST** (lesões de mama) e reconstruir imagens a partir dessas embeddings via decoder.

A linha de investigação recente testou se é possível **prever a embedding da imagem a partir apenas de dados tabulares clínicos** (idade, BI-RADS, ecogenicidade, etc.), sem ver a imagem — e por quê isso falha. Este documento resume o pipeline e as conclusões para replicar o experimento com outro tipo de modelo de embedding (não-MAE).

## Ambiente

- pyenv virtualenv `ultrassom` (Python 3.12), dependências em `requirements-ultrassom.txt` (torch 2.13, timm 1.0.28, numpy 2.5.2, catboost, lightgbm, xgboost, scikit-learn).
- Ativar com:
  ```bash
  export PATH="$HOME/.pyenv/bin:$PATH"
  eval "$(pyenv init -)"
  eval "$(pyenv virtualenv-init -)"
  pyenv activate ultrassom
  ```
- Checkpoint do USF-MAE já deve estar na raiz do projeto (`USF-MAE_full_pretrain_43dataset_100epochs.pt`) ou apontado via env var `MAE_CHECKPOINT`.
- **Específico do MAE** (não se aplica a outro modelo): o submódulo `mae/` (fork do facebookresearch/mae) precisa estar populado com 2 patches de compatibilidade — remoção de `qk_scale` em `mae/models_mae.py` e troca de `np.float` por `float` em `mae/util/pos_embed.py` (documentado no `README.md` raiz).

## Dataset

- Imagens/máscaras: `dataset_breast/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks/caseNNN.png` (ignore `*_tumor.png` e `*_other*.png`, que são máscaras).
- Dados clínicos: `dataset_breast/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx` (coluna `Image_filename` liga a linha à imagem; `CaseID` é o identificador numérico, ex: `CaseID=1` → `case001.png`).
- Diretório de dados é git-ignored — precisa ser baixado/extraído manualmente (ver `dataset_BrEaST.pdf` no mesmo diretório para documentação do dataset original).

## Pipeline (ordem de execução)

1. **Gerar embeddings por patch** — `python -m serve.generate_embeddings`
   Codifica cada imagem em embeddings por token (CLS + patches) e salva `embeddings_per_patch.json` ao lado das imagens. **Específico do MAE**: usa `serve/inference.py::encode_image`, que assume ViT-Base/patch16 (197 tokens = 1 CLS + 196 patches 14×14, 768-dim cada). Para outro modelo de embedding, seria preciso reimplementar essa função conforme a arquitetura (nº de tokens, dimensão, se há token de resumo global como o CLS).

2. **Construir dataset tabular** — `python -m src.preprocessing.build_tabular_dataset`
   Faz one-hot encoding dos dados clínicos (reproduz `notebooks/data_exploration.ipynb`) e junta com as embeddings por token, gerando `dataset_breast/processed_dataset_embeddings.csv` (1 linha por token: `patient_id`, `x_pos`, `y_pos`, colunas clínicas, `emb_0`..`emb_(D-1)`). **Genérico** — só precisa saber a dimensionalidade `D` da embedding do novo modelo e como mapear posição de patch (`x_pos`,`y_pos`; token especial tipo CLS = `(-1,-1)`).

3. **Treinar regressor** — `python -m src.models.train_catboost`
   Treina CatBoost (MultiOutputRegressor, `MultiRMSE`) para prever as `D` dimensões da embedding a partir das features tabulares, split por paciente (evita vazamento entre tokens da mesma imagem). Salva modelo (`.joblib`) e previsões no formato `infered_embeddings.json` (mesmo schema de `embeddings_per_patch.json`), para permitir decodificar/comparar depois.

4. **Reconstruir e comparar visualmente** — `python -m serve.reconstruct_and_compile --limit N`
   Decodifica embedding real vs. predita e monta imagens lado a lado. **Específico do MAE** (usa o decoder do MAE via `serve/inference.py::decode_embedding`) — para outro modelo, seria preciso o decoder/gerador equivalente daquele modelo (se existir).

## Scripts de avaliação (`src/evaluation/`)

Todos genéricos, desde que se ajuste a dimensão `D` e o nº de tokens por imagem:

- `compare_prediction_distance.py` — mede o erro da predição em unidades de desvio-padrão por dimensão (permite comparar magnitude do erro do modelo com testes de perturbação sintética).
- `compare_token_variance.py` — compara a variância entre tokens (real vs. predita) por dimensão. ⚠️ Foi o teste que sustentava a antiga "causa raiz", mas rodou sobre tokens embaralhados (ver Conclusões e `LOGS.md`); precisa ser reexecutado antes de suas medições serem citadas.
- `plot_dimension_distributions.py`, `plot_embedding_divergence.py`, `plot_qq_divergence.py` — visualizações complementares da distribuição/dispersão da predição vs. real.
- `pilot_single_vs_multi_target.py` — compara treinar 1 modelo multi-output vs. N modelos single-target para uma única posição de patch.

Testes de sensibilidade do decoder (específicos do MAE, mas o *conceito* é reaplicável): `serve/perturbation_test.py` (perturbação gaussiana calibrada por fração do desvio-padrão) e `serve/cls_replacement_test.py` (substituição do token CLS por vetor nulo/aleatório).

Modelo alternativo por posição (não recomendado, ver Conclusões): `src/models/train_catboost_per_position.py`. TabPFN via API (`src/models/train_tabpfn_sample.py`) — requer `TABPFN_TOKEN` (conta gratuita em priorlabs.ai; cota diária limitada, ~23 dimensões/dia no plano free).

## Conclusões principais (o "porquê" por trás dos resultados)

> ⚠️ **Revisado em 2026-09-15 após a correção do bug de `ids_restore`** (ver `LOGS.md`).
> Os itens 1, 2 e 4 abaixo foram obtidos com os patch tokens **embaralhados**: o MAE
> permuta a sequência de patches por imagem mesmo com `mask_ratio=0.0`, e o pipeline em
> `src/` ignorava a permutação salva. A conclusão prática se mantém, mas a explicação
> registrada estava errada. O texto abaixo já reflete a revisão.

1. **Prever a embedding a partir do tabular não serve para reconstrução** — confirmado
   visualmente, com um teto íntegro para comparar. Nas reconstruções de casos do split
   de teste (`outputs/reconstructions/compiled/`), o decode da embedding **verdadeira**
   é fiel (lesão, margens e camadas visíveis), enquanto o decode da embedding
   **inferida** é um gradiente cinza sem estrutura — colapso para a média.

   Os números de R², porém, mudaram com a correção (mesmo código, mesmo split, mesmos
   hiperparâmetros — só a ordenação dos tokens):

   | Dataset | MAE | RMSE | R² |
   |---|---|---|---|
   | Antigo (tokens embaralhados) | 0.133 | 0.168 | 0.015 |
   | Corrigido (ordem espacial) | 0.113 | 0.145 | **0.185** |

   **R² subiu 12× sem que a reconstrução melhorasse em nada.** A leitura mais provável é
   que o ganho vem de estrutura global de baixa frequência (gradiente de brilho,
   geometria da sonda), que só passou a ficar consistentemente posicionada depois da
   correção — nada disso é conteúdo diagnóstico. É um bom lembrete de que R² agregado
   sobre 768 dimensões não mede capacidade de reconstrução.

2. **Causa raiz — versão corrigida**: o dado clínico tabular determina apenas
   **estatísticas globais** da imagem, não o conteúdo local da lesão. Cada imagem gera N
   linhas com as mesmas features clínicas, diferindo só em `x_pos`/`y_pos`, e o modelo
   converge para a média por posição.

   ⚠️ A formulação anterior — "a posição de um patch não determina seu conteúdo entre
   imagens diferentes" — **descrevia o bug, não o fenômeno**: as posições literalmente
   eram patches diferentes em cada imagem. A medição que a sustentava ("a posição explica
   apenas ~0.3% da variância entre patches", via `compare_token_variance.py`) foi feita
   sobre dados embaralhados e **precisa ser refeita** antes de ser citada de novo.

   Isso é corroborado por outra medição independente, com embeddings *pooled* (imunes ao
   bug): tabular → embedding pooled de 768 dims dá R² ≈ 0.08 (ver
   `docs/tabular-to-umap-log.md`). Sinal fraco, mas real, nas duas medições.

3. **O token tipo CLS (resumo global) é bem mais previsível que os tokens de patch** — mas o decoder do MAE **ignora quase completamente** o CLS na reconstrução espacial (confirmado substituindo-o por ruído extremo, vetor nulo ou vetor aleatório: reconstrução não muda). Ou seja, mesmo que um modelo acerte bem o "resumo global", isso não ajuda a reconstruir a imagem.

4. **Fragmentar o modelo não resolve** (nem 197 modelos por posição, nem 197×768 modelos single-target, nem trocar CatBoost por TabPFN) — o gargalo é a **ausência de sinal real**, não a arquitetura do regressor. Modelos mais sofisticados (TabPFN) tiveram desempenho *pior*, não melhor.

   ⚠️ Estes experimentos leem `processed_dataset_embeddings.csv` e portanto rodaram sobre
   tokens embaralhados. A conclusão é plausível (é consistente com o item 1, que foi
   reverificado), mas os números específicos **não foram refeitos** após a correção.

## O que muda para um modelo de embedding diferente (não-MAE)

- Reimplementar o equivalente de `serve/inference.py::load_model`/`encode_image` para o novo modelo (arquitetura, checkpoint, pré-processamento de imagem).
- Confirmar a estrutura de tokens do novo modelo: quantos tokens por imagem, se existe um token de resumo global (tipo CLS), dimensionalidade da embedding.
- Se o novo modelo tiver um decoder/gerador (para reconstruir imagem a partir da embedding), adaptar `decode_embedding` e os scripts de reconstrução/perturbação.
- Os passos 2–4 do pipeline e os scripts de `src/evaluation/` devem funcionar quase sem alteração, parametrizando apenas `D` (dimensão) e a lógica de `x_pos`/`y_pos` (ou removendo-a, se o novo modelo não tiver noção de patches espaciais).
- Vale replicar a mesma checagem de variância explicável por posição (`compare_token_variance.py`) **antes** de investir em regressores mais complexos — se o resultado for parecido (posição não é preditiva entre imagens), o mesmo problema estrutural provavelmente se repete independentemente do modelo de embedding.
