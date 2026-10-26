# Representações de imagens de ultrassom mamário e variáveis clínicas

Projeto da disciplina MAC 5921 (Deep Learning, IME-USP). O código mede quanta informação
sobre as variáveis anotadas pelo especialista (BI-RADS, forma, margem, ecogenicidade...)
as representações de imagens de ultrassom do conjunto **BrEaST** carregam, comparando o
**USF-MAE** (especializado em ultrassom) com o VAE do Stable Diffusion (genérico), e
controlando o fator de confusão dos lotes de aquisição.

A primeira fase do projeto tentava o caminho inverso (gerar imagens a partir das anotações)
e foi abandonada; o código dela continua no repositório e está descrito no fim deste
documento. O histórico completo de experimentos, decisões e correções está em
[`LOGS.md`](LOGS.md).

---

## 1. Requisitos

- Linux x86-64 (testado no Ubuntu, kernel 6.8), **Python 3.12.3**.
- Só CPU: nenhum passo exige GPU.
- **Memória:** ~10 GB de RAM livres. Os experimentos rodam vários processos em paralelo
  (`--n-jobs`); numa máquina com 16 GB e sem swap, use `--n-jobs 4` ou menos. Para evitar
  que um estouro de memória trave a máquina, recomendamos rodar os passos longos com teto
  rígido (systemd):

  ```bash
  systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0 python -m <módulo> ...
  ```
- **Disco:** ~0,6 GB de dados, checkpoint e representações, mais ~6 GB do ambiente Python (PyTorch).
- **Tempo total:** ~8 h de CPU para reproduzir todos os resultados (tempos por passo abaixo,
  medidos num notebook com 4–6 processos).

## 2. Instalação

```bash
git clone --recursive https://github.com/LuckaGianvechio/reconstrucao_ultrassom.git
cd reconstrucao_ultrassom

# MAE oficial (facebookresearch/mae) + correção de compatibilidade com timm/numpy atuais
git -C mae apply ../patches/mae-compat.patch

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-ultrassom.txt
```

Se já tiver clonado sem `--recursive`, rode `git submodule update --init` antes do patch.

O patch em [`patches/mae-compat.patch`](patches/mae-compat.patch) muda três linhas do MAE
oficial: remove o argumento `qk_scale` (que não existe mais no `timm` moderno) e troca
`np.float` por `float` (removido no `numpy` 2). Não altera o modelo.

`requirements-ultrassom.txt` fixa as versões exatas usadas nos resultados (inclui as rodas
CUDA do PyTorch, que só são baixadas, não usadas).

Todos os comandos abaixo são executados **na raiz do repositório**, com o ambiente ativado.

## 3. Dados e pesos

Nada disso é versionado (ver `.gitignore`); é preciso baixar.

**Conjunto de dados BrEaST** (Breast-Lesions-USG, *The Cancer Imaging Archive*,
DOI [10.7937/9WKK-Q141](https://doi.org/10.7937/9WKK-Q141)). Baixe os dois arquivos da
versão de 15/12/2023 e organize assim:

```
dataset_breast/
├── BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx
└── BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/
    └── BrEaST-Lesions_USG-images_and_masks/      # conteúdo do .zip
        ├── case001.png
        ├── case001_tumor.png
        └── ...
```

```bash
mkdir -p dataset_breast/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023
unzip BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip \
      -d dataset_breast/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/
mv BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx dataset_breast/
```

**Checkpoint do USF-MAE** (100 épocas, o recomendado pelos autores): link na seção
*Checkpoints* de [`USF-MAE/README.md`](https://github.com/Yusufii9/USF-MAE#-checkpoints).
Salve na raiz do repositório como `USF-MAE_full_pretrain_43dataset_100epochs.pt`, ou
aponte a variável `MAE_CHECKPOINT` para o arquivo.

**VAE do Stable Diffusion** (`stabilityai/sd-vae-ft-mse`): baixado automaticamente do
Hugging Face na primeira execução.

## 4. Pipeline

Os passos devem ser rodados **na ordem**: cada um lê o que os anteriores gravaram. Todos os
resultados vão para `outputs/embedding_to_tabular_experiments/<pasta>/`. As sementes são
fixas (`random_state=42` nas partições e nos modelos), então os números devem coincidir com
os do relatório.

### Passo 1 — Representações das imagens (CPU, tempo não cronometrado)

```bash
python -m serve.generate_embeddings       # USF-MAE → embeddings_per_patch.json (197 tokens × 768)
python -m serve.generate_embeddings_vae   # SD-VAE  → embeddings_vae.json (latente 4×32×32)
```

Os dois arquivos são gravados na pasta das imagens. Nota: o MAE sorteia uma ordem dos
*patches* a cada execução (`ids_restore` é salvo junto e o código desfaz a permutação), por
isso o JSON muda entre execuções; o token CLS e a média dos *patches*, que são o que os
experimentos usam, não dependem dessa ordem.

### Passo 2 — Comparação de representações, reduções e modelos (~2 h 30)

Grade: 4 representações × {bruta, PCA, UMAP com k = 10/30/50} × {CatBoost, XGBoost,
LightGBM, MLP} × 39 alvos, validação cruzada estratificada de 5 *folds*.

```bash
python -m src.models.train_embedding_to_clinical_gbm --folder-name gbm_pca_umap_sweep
python -m src.evaluation.analyze_embedding_to_clinical_sweep --folder-name gbm_pca_umap_sweep
python -m src.evaluation.metadata_only_baseline --folder-name gbm_pca_umap_sweep
```

Para testar a instalação antes da execução completa:
`python -m src.models.train_embedding_to_clinical_gbm --folder-name smoke_test --quick`.

→ `gbm_pca_umap_sweep/`: `results_long.csv` (tudo, por *fold*), `summary_by_combo.csv`,
`comparison_*.csv` e `paired_tests.csv` (MAE vs. VAE, PCA vs. UMAP, modelos),
`metadata_only_baseline.csv` (linha de base de aquisição).

### Passo 3 — Fator de confusão de aquisição (~1 h)

Representação (CLS do USF-MAE) vs. linha de base de aquisição (largura, altura,
`Pixel_size`), em validação cruzada estratificada e agrupada por lote (tamanho da imagem e
`Pixel_size`), mais os testes de significância.

```bash
python -m src.evaluation.acquisition_confound --folder-name acquisition_confound
python -m src.evaluation.history_from_annotations --folder-name acquisition_confound
python -m src.evaluation.significance_tests
```

→ `acquisition_confound/`: `results_folds.csv`, `results.csv`, `summary_by_group.csv`,
`history_from_annotations.csv`, `significance_tests.csv`.

### Passo 4 — Relatório consolidado dos passos 2 e 3 (rápido, sem treino)

```bash
python -m src.evaluation.analyze_best_per_variable
python -m src.evaluation.build_report
```

→ `report/`: `report.md`, `relatorio.html`, `figures/`, `tables/`.

### Passo 5 — BI-RADS (~1 h 30)

Multiclasse, ordinal, categorias e cortes binários (≥4a, ≥4b, ≥4c), com aquisição,
pré-ultrassom (idade, sinais, sintomas), representação e fusões precoce e tardia.

```bash
python -m src.evaluation.birads_analysis --folder-name birads_analysis
python -m src.evaluation.birads_figures
```

→ `birads_analysis/`: `results_folds.csv`, `results.csv`, `oof_predictions.csv`,
`significance_tests.csv`, `figures/fig1–fig4`.

### Passo 6 — Malignidade vs. BI-RADS do médico (~25 min)

Usa os mesmos *folds* do passo 5.

```bash
python -m src.evaluation.malignancy_vs_birads --folder-name malignancy_vs_birads
```

→ `malignancy_vs_birads/`.

### Passo 7 — Pré-ultrassom + representação para as 23 variáveis de imagem (~3 h)

```bash
python -m src.evaluation.image_targets_with_pre_us --folder-name image_targets_with_pre_us
# piloto com LightGBM (~20 min), opcional:
python -m src.evaluation.image_targets_with_pre_us --folder-name image_targets_with_pre_us_lgbm --model lightgbm
```

→ `image_targets_with_pre_us/`.

Os passos com testes estatísticos aceitam uma opção para refazer só os testes a partir dos
CSVs já salvos, sem retreinar: `--retest` (passos 5 e 7), `--rewithin` (passo 6),
`--resummarize` (passo 2).

### Onde está cada resultado do relatório (Tarefa 2)

| Resultado | Arquivo |
|---|---|
| MAE vs. VAE, bruta vs. PCA vs. UMAP, comparação de modelos | `gbm_pca_umap_sweep/comparison_*.csv`, `paired_tests.csv` |
| Tabela 1 (aquisição, representação, repr. + aquisição) | `acquisition_confound/summary_by_group.csv` |
| Tabela 1 (pré-US, fusão tardia) | `image_targets_with_pre_us/` |
| Ganho sobre a aquisição por variável, valores de *q* | `acquisition_confound/significance_tests.csv` |
| Tabela 2, linhas de BI-RADS | `birads_analysis/results.csv`, `significance_tests.csv` |
| Tabela 2, linha de malignidade; BI-RADS 5 | `malignancy_vs_birads/` |

---

## 5. Fase 1 (abandonada): reconstrução a partir das anotações

Resultados citados na seção 1 do relatório. Detalhes em [`LOGS.md`](LOGS.md) (2026-09-15)
e em `docs/`.

```bash
# anotações → representação por patch (R² ≈ 0,185) e reconstrução pelo decodificador
python -m src.preprocessing.build_tabular_dataset
python -m src.models.train_catboost
python -m serve.reconstruct_and_compile --cases case007.png case080.png case207.png case229.png

# anotações → UMAP → UMAP inverso (R² = −0,158 no espaço de 768 dimensões)
python -m src.models.train_tabular_to_umap --pooling mean
python -m src.models.train_tabular_to_umap_inverse --pooling mean
```

Saídas em `outputs/reconstructions/compiled/` e `outputs/tabular_to_umap/`. Atenção: o
`inverse_transform` do `umap-learn` depende de quais pontos são invertidos juntos (ver
`LOGS.md`), então os valores do UMAP inverso não são reproduzíveis com outro tamanho de lote.

## 6. Estrutura

```
src/preprocessing/   carregamento das anotações, alvos clínicos e pooling das representações
src/models/          experimentos de treino (varredura principal e fase 1)
src/evaluation/      análises, testes estatísticos, figuras e relatórios
serve/               carregamento do MAE/VAE, geração de representações e reconstrução
mae/, USF-MAE/       submódulos: MAE oficial (com patch) e repositório do USF-MAE
patches/             correção de compatibilidade do MAE
docs/                registros detalhados por tema
LOGS.md              diário cronológico dos experimentos
```

## Referências

- A. Pawłowska et al. *A curated benchmark dataset for ultrasound-based breast lesion
  analysis (Breast-Lesions-USG)*. The Cancer Imaging Archive, 2024. DOI 10.7937/9WKK-Q141.
- Y. Megahed et al. *USF-MAE: Ultrasound self-supervised foundation model with masked
  autoencoding*. Biomedical Signal Processing and Control, v. 122, 2026.
- K. He et al. *Masked autoencoders are scalable vision learners*. CVPR, 2022.
