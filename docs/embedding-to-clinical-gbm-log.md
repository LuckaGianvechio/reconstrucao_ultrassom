# Log — informação clínica nas embeddings: USF-MAE vs. SD-VAE, PCA vs. UMAP, 4 modelos

Após os experimentos de reconstrução (ver `LOGS.md`, 2026-09-15) não produzirem
imagens com anatomia, o projeto mudou de pergunta: em vez de reconstruir a imagem
a partir da embedding, medir **quanta informação a embedding carrega sobre as
variáveis que o especialista anota olhando a própria imagem** (BIRADS, forma,
margem, ecogenicidade, diagnóstico...). A medida é o desempenho de classificadores
treinados sobre a embedding.

Continuação de `docs/embedding-to-clinical-log.md`, que já mostrou que há sinal
nessa direção, mas usava split único 80/20, só CatBoost + MLP, sem PR-AUC e com PCA
por fração de variância (nº de componentes diferente do UMAP). Aquele doc continua
válido como registro da primeira rodada.

## Perguntas

1. Que informação clínica anotada a partir da imagem as embeddings carregam?
2. USF-MAE ou SD-VAE carrega mais?
3. Reduzir dimensionalidade ajuda? PCA ou UMAP?
4. O resultado depende do modelo (CatBoost, XGBoost, LightGBM, MLP)?

## Metodologia

Script: `src/models/train_embedding_to_clinical_gbm.py`. Saídas em
`outputs/embedding_to_tabular_experiments/{folder_name}/`.

- **Embeddings** (256 imagens = 256 pacientes, então CV por imagem já é CV por paciente):
  - `mae_mean` — média dos 196 patch tokens do USF-MAE (768-d)
  - `mae_cls` — token CLS (768-d)
  - `vae_mean` — média espacial do latente do SD-VAE (4-d). Só entra em `raw`:
    não há como reduzir 4 dims para 10.
  - `vae_flat` — latente do SD-VAE achatado, `4×32×32 = 4096-d`
    (`embedding_pooling.py::load_vae_flat`). **Novo**: o `vae_mean` descarta toda a
    informação espacial, o que tornava a comparação MAE vs. VAE da rodada anterior
    injusta com o VAE. O achatamento é válido porque o latente do VAE é espacialmente
    alinhado entre imagens (sem o embaralhamento de tokens do MAE).
- **Redução de dimensionalidade**: `raw`, `pca` e `umap`, ambos com
  n_components ∈ {10, 30, 50} — o mesmo nº nos dois métodos, para comparação direta.
  Ajustados só no fold de treino. UMAP com `StandardScaler` antes e repetido com 3
  seeds por fold (média), por ser estocástico.
- **Modelos**: CatBoost, XGBoost, LightGBM (150 árvores, profundidade 4, lr 0.08,
  classes balanceadas — `auto_class_weights`, `class_weight` e `sample_weight`
  balanceado no XGBoost) e MLP `(64,)` com `StandardScaler`, igual à rodada anterior.
  Baseline: `DummyClassifier(most_frequent)` / `DummyRegressor(mean)`.
- **Validação**: `StratifiedKFold(5)` (classificação) / `KFold(5)` (Age), `shuffle`,
  `random_state=42` — as mesmas partições para todas as embeddings, redutores e modelos
  de um alvo, então as comparações são pareadas por fold.
- **Alvos** (`clinical_targets.py`): Age (regressão), colunas single-label (BIRADS,
  Shape, Echogenicity, Posterior_features, Halo, Calcifications, Skin_thickening,
  Verification) e um alvo binário por valor das colunas multi-label (Interpretation,
  Margin, Signs, Symptoms, Tissue_composition, Diagnosis). Classes com < 5 amostras
  são descartadas. Colunas single-label que ficam com 2 classes após o filtro
  (Halo, Calcifications, Skin_thickening, Verification) passam a ser tratadas como
  **binárias** (precision/recall na classe positiva) — na rodada anterior eram
  tratadas como multiclasse com média macro.
- **Métricas** (média ± desvio entre folds): ROC-AUC, PR-AUC (average precision;
  multiclasse = one-vs-rest com média macro), precision, recall, F1 (binário: classe
  positiva; multiclasse: macro), accuracy, balanced accuracy; R², MAE, RMSE para Age.
  `delta_*` = métrica do modelo − baseline **no mesmo fold**. Note que o PR-AUC do
  baseline é a prevalência da classe positiva, então `delta_pr_auc` é o ganho sobre
  o acaso nessa métrica; o ROC-AUC do baseline é sempre 0.5.
- **Alvos-sentinela**: rótulos derivados de "not available"/"not applicable"
  (`Signs_not available`, `Diagnosis_not applicable`...) descrevem se o campo do laudo
  foi preenchido, não a lesão — a rodada anterior mostrou que alguns deles têm sinal
  alto, provavelmente por metadado. Ficam em `results_long.csv` com
  `is_sentinel_target=True`, mas **fora** das médias de `summary_by_combo.csv`.

### Arquivos de saída

| Arquivo | Conteúdo |
|---|---|
| `results_folds.csv` | uma linha por (alvo, embedding, redutor, n_components, modelo, fold), seeds do UMAP já em média |
| `results_long.csv` | média e desvio entre folds de cada combinação |
| `summary_by_combo.csv` | média sobre os alvos de classificação não-sentinela, por (embedding, redutor, n_components, modelo) |
| `best_by_target.csv` | melhor combinação por alvo (ROC-AUC; R² para Age) — máximo sobre 91 combinações, portanto **otimista** |
| `config.json` | grade, nº de folds/seeds, versões das bibliotecas, tempo de execução |
| `comparison_{embeddings,reducers,models}.csv`, `comparison_models_friedman.csv`, `paired_tests.csv` | `src/evaluation/analyze_embedding_to_clinical_sweep.py` |
| `metadata_only_baseline.csv` | `src/evaluation/metadata_only_baseline.py` (ver "Fator de confusão") |

**Grupos de alvo** (`target_group` no script): `image` — anotado pelo especialista a
partir da imagem (BIRADS, Shape, Echogenicity, Posterior_features, Halo,
Calcifications, Skin_thickening, Margin_*, Tissue_*, Interpretation_*; 23 alvos);
`pathology` — Diagnosis_* e Verification (6); `history` — Signs_*, Symptoms_*, Age
(5 de classificação), **não visíveis na imagem** e usados como controle negativo;
`sentinel` (4). As perguntas 1–4 se referem ao grupo `image`.

## Resultados — `outputs/embedding_to_tabular_experiments/gbm_pca_umap_sweep/`

Rodada de 2026-09-26: 156 tarefas (4 embeddings × 39 alvos), 5 folds, 3 seeds de
UMAP, 2h25 com 4 workers. Comparações abaixo pareadas por alvo (Wilcoxon, n=23),
com a média dos 4 modelos por alvo quando o fator comparado não é o modelo. São ~80
testes: tratar 0.01 < p < 0.05 como sugestivo, não como conclusivo.

### ⚠️ Fator de confusão: aquisição (leia antes das perguntas 1–4)

Duas medições mostram que boa parte do que as embeddings "sabem" sobre os alvos vem
do **aparelho/protocolo de aquisição**, não do conteúdo da lesão:

**1. O controle negativo falha.** Os alvos `history` (histórico hormonal, histórico
familiar, lesão palpável) não estão na imagem, mas são previstos **melhor** que os de
imagem:

| grupo | ROC-AUC `mae_cls` bruto + CatBoost |
|---|---|
| `history` | **0.825** |
| `image` | 0.714 |
| `pathology` | 0.636 |

**2. Três números de aquisição já preveem muito.** CatBoost só com largura, altura e
`Pixel_size` da imagem, com os mesmos folds (`metadata_only_baseline.py`):

| grupo | só metadados | embedding (`mae_cls`) | ganho da embedding |
|---|---|---|---|
| `history` | 0.718 | 0.825 | +0.107 |
| `image` | 0.634 | 0.714 | +0.080 |
| `pathology` | 0.655 | 0.636 | −0.019 |

Exemplo extremo: os 28 casos com imagem de 433×469 px têm **todos**
`Signs = not available` — tamanho de imagem identifica um lote de aquisição, e cada
lote tem seu padrão de preenchimento do laudo e sua população de pacientes.

**Consequência.** O ganho da embedding sobre os metadados é **maior** nos alvos
invisíveis (+0.107) do que nos de imagem (+0.080). Logo, nem o ganho residual pode
ser atribuído ao conteúdo da lesão: a embedding captura a aquisição com mais detalhe
do que três números conseguem (textura característica do aparelho, bordas, texto e
marcações gravados na imagem). **Este desenho mede "o que a embedding permite prever",
não "que informação clínica visual ela carrega".** As comparações relativas abaixo
(MAE vs. VAE, PCA vs. UMAP, modelos) continuam válidas como comparações, mas os valores
absolutos são um teto contaminado.

Alvos de imagem onde a embedding supera os metadados com folga — os candidatos mais
fortes a sinal de conteúdo real: `Skin_thickening` (0.79 vs. 0.44), `Mammary duct
ectasia` (0.76 vs. 0.50), `Posterior_features` (0.80 vs. 0.60), `Calcifications`
(0.61 vs. 0.43), `Margin_spiculated` (0.82 vs. 0.69), `Echogenicity` (0.64 vs. 0.54).
Onde **não** supera: `Shape` (0.686 vs. 0.674), `Intraductal papilloma` (0.83 vs.
0.82), `Interpretation_Fibroadenoma` (0.75 vs. 0.78) — nesses, o sinal é
explicável por aquisição.

### 1. Que informação os alvos de imagem mostram

Melhor configuração média: `mae_cls` bruto + CatBoost, ROC-AUC **0.714**, PR-AUC
0.505 (Δ sobre a prevalência **+0.244**), balanced accuracy 0.563 (baseline 0.468).
Variáveis citadas no escopo (`mae_cls` bruto + CatBoost, média ± desvio entre folds):

| alvo | ROC-AUC | PR-AUC | Δ PR-AUC | precision | recall | só metadados |
|---|---|---|---|---|---|---|
| `Margin_spiculated` | 0.815 ± 0.164 | 0.504 | +0.375 | 0.500 | 0.305 | 0.688 |
| `Posterior_features` (4 cl.) | 0.798 ± 0.069 | 0.560 | +0.310 | 0.489 | 0.457 | 0.602 |
| `Halo` | 0.791 ± 0.046 | 0.625 | +0.406 | 0.643 | 0.527 | 0.748 |
| `Margin_circumscribed` | 0.765 ± 0.043 | 0.717 | +0.267 | 0.660 | 0.661 | 0.682 |
| `Margin_indistinct` | 0.743 ± 0.041 | 0.732 | +0.283 | 0.662 | 0.652 | 0.657 |
| `Margin_angular` | 0.700 ± 0.072 | 0.419 | +0.255 | 0.435 | 0.281 | 0.647 |
| `Shape` (3 cl.) | 0.686 ± 0.054 | 0.546 | +0.213 | 0.405 | 0.410 | 0.674 |
| `BIRADS` (6 cl.) | 0.669 ± 0.053 | 0.375 | +0.208 | 0.314 | 0.323 | 0.593 |
| `Margin_microlobulated` | 0.659 ± 0.107 | 0.343 | +0.202 | 0.227 | 0.132 | 0.636 |
| `Echogenicity` (6 cl.) | 0.637 ± 0.110 | 0.370 | +0.203 | 0.282 | 0.255 | 0.541 |
| `Diagnosis_Invasive carcinoma NST` | 0.729 ± 0.106 | 0.521 | +0.263 | 0.628 | 0.409 | 0.608 |
| `Diagnosis_Invasive lobular carcinoma` | 0.731 ± 0.149 | 0.167 | +0.116 | 0.000 | 0.000 | 0.769 |
| `Diagnosis_DCIS` | 0.513 ± 0.089 | 0.097 | +0.042 | 0.000 | 0.000 | 0.683 |
| `Diagnosis_Fibroadenoma` | 0.448 ± 0.080 | 0.133 | +0.016 | 0.000 | 0.000 | 0.639 |

- Diagnósticos raros (13–30 positivos) têm recall 0 no limiar padrão: o modelo
  ordena razoavelmente (ROC-AUC) mas nunca prevê a classe. Para esses, ROC-AUC e PR-AUC
  são as métricas informativas; precision/recall exigiriam calibrar o limiar.
- `Age`: R² = 0.126 ± 0.187 (`mae_mean` bruto + CatBoost) — praticamente nada.
- `accuracy` engana aqui: o baseline que chuta a classe majoritária já tem 0.759.

### 2. USF-MAE vs. SD-VAE — **MAE, com folga**

| embedding | ROC-AUC (bruto) | ROC-AUC (melhor redutor) | PR-AUC (melhor) |
|---|---|---|---|
| `mae_cls` | 0.700 | 0.700 (bruto) | 0.495 |
| `mae_mean` | 0.693 | 0.693 (bruto) | 0.494 |
| `vae_flat` | 0.562 | 0.592 (PCA 10) | 0.385 |
| `vae_mean` | 0.568 | 0.568 (bruto) | 0.364 |

- MAE > VAE em **22 e 23 de 23 alvos** (p < 0.001), +0.10 de ROC-AUC e +0.11 de PR-AUC.
- `mae_mean` ≈ `mae_cls` (p = 0.39 em ROC-AUC, p = 0.94 em PR-AUC).
- Achatar o latente do VAE (4096-d) **não ajuda por si só** (0.562 vs. 0.568 do
  `vae_mean`); só com PCA 10 chega a 0.592 (p = 0.045, sugestivo). A hipótese de que o
  `vae_mean` perdia o sinal espacial não se confirma: o latente do SD-VAE, que codifica
  aparência de baixo nível para reconstrução, carrega pouca informação clínica em
  qualquer formato.

### 3. PCA vs. UMAP — **nenhum dos dois ajuda no MAE; com o mesmo nº de componentes, PCA > UMAP**

ROC-AUC, média dos 4 modelos (`comparison_reducers.csv`):

| | bruto | PCA 10 | PCA 30 | PCA 50 | UMAP 10 | UMAP 30 | UMAP 50 |
|---|---|---|---|---|---|---|---|
| `mae_cls` | **0.700** | 0.635 | 0.660 | 0.643 | 0.618 | 0.623 | 0.621 |
| `mae_mean` | **0.693** | 0.633 | 0.652 | 0.646 | 0.613 | 0.607 | 0.617 |
| `vae_flat` | 0.562 | **0.592** | 0.588 | 0.583 | 0.568 | 0.566 | 0.565 |

- **MAE**: a embedding bruta vence toda redução em 20–23 de 23 alvos (p < 0.001).
  Custo: −0.04 a −0.06 com PCA, −0.08 com UMAP.
- **PCA > UMAP** em todas as 9 comparações de mesmo k (MAE e VAE): p ≤ 0.002 para
  k = 30 no MAE; nos demais, entre 0.002 e 0.08. A direção é sempre a mesma.
- **VAE**: PCA ajuda um pouco (+0.03 sobre o bruto, p = 0.005 com k = 10); UMAP não.
- ⚠️ Corrige `docs/embedding-to-clinical-log.md`, que achou o UMAP "menos destrutivo"
  que o PCA. Lá o PCA era por fração de variância (poucos componentes) e o UMAP tinha
  10–100; com o **mesmo** nº de componentes, o PCA preserva mais. A outra conclusão
  daquele doc se mantém: reduzir a dimensionalidade do MAE piora a predição.

### 4. Modelos — **CatBoost ligeiramente à frente; XGBoost ≈ LightGBM; MLP atrás**

ROC-AUC na embedding bruta (`comparison_models_friedman.csv`):

| embedding | CatBoost | XGBoost | LightGBM | MLP | Friedman p |
|---|---|---|---|---|---|
| `mae_cls` | **0.714** | 0.707 | 0.703 | 0.674 | 0.006 |
| `mae_mean` | **0.708** | 0.696 | 0.694 | 0.672 | < 0.001 |
| `vae_flat` | 0.567 | 0.558 | 0.559 | 0.564 | 0.51 |

- Os 3 GBMs estão a ~0.01 um do outro. CatBoost vs. XGBoost/LightGBM: p entre 0.05 e
  0.23 — não significativo com este nº de testes. XGBoost vs. LightGBM: p ≈ 0.5–0.8.
- MLP fica 0.02–0.04 abaixo de qualquer GBM no MAE (p ≤ 0.033 em todos os pares;
  < 0.001 contra o CatBoost).
- No VAE nenhum modelo se destaca: com o sinal perto do acaso, não há o que separar.
- Custo computacional na embedding bruta (1 fold, 1 thread, BIRADS em `vae_flat`):
  CatBoost 106 s, XGBoost 62 s, LightGBM 14 s, MLP 0.1 s. LightGBM entrega quase o
  mesmo desempenho do CatBoost com ~8× menos tempo.

### Ressalvas desta rodada

- `Calcifications` saiu com a classe positiva invertida (`no` = 1, pelo LabelEncoder):
  seu PR-AUC/precision/recall se referem a "sem calcificação". ROC-AUC e balanced
  accuracy não são afetados; nas médias por grupo o efeito é o mesmo para todas as
  combinações, então as comparações não mudam. **Corrigido no código** (classe positiva
  = minoritária) para as próximas rodadas.
- `best_by_target.csv` escolhe o máximo de 91 combinações por alvo: otimista,
  sobretudo em alvos com poucos positivos (ex.: `Invasive lobular carcinoma` 0.903 com
  `vae_flat` + XGBoost, 13 positivos, desvio 0.11 entre folds).
- 5 folds sobre 256 imagens: desvio entre folds de 0.04–0.18 de ROC-AUC por alvo.
- Precision/recall usam o limiar padrão (0.5) com classes balanceadas; não houve
  ajuste de limiar.

## Próximos passos sugeridos

1. **Isolar a aquisição** antes de qualquer conclusão sobre conteúdo clínico:
   (a) comparar *metadados + embedding* vs. *só metadados* como medida do ganho da
   embedding; (b) CV agrupada por lote de aquisição (tamanho de imagem / `Pixel_size`
   como proxy de aparelho), para que o modelo não possa usar o lote.
2. **Embeddings só da lesão**: recortar a imagem pela máscara (`Mask_tumor_filename`)
   antes de extrair a embedding, removendo bordas, texto e marcações gravadas — e
   medir se o controle negativo (`history`) cai para perto de 0.5.
3. Rodar a próxima versão já com a correção da classe positiva de `Calcifications`.
