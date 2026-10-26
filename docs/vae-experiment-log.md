# Log de experimentos — SD-VAE como modelo de embedding alternativo

Este documento registra a investigação de usar o VAE do Stable Diffusion
(`stabilityai/sd-vae-ft-mse`, formato diffusers) como alternativa ao
USF-MAE para gerar/reconstruir embeddings de imagens do dataset BrEaST.
Ver `ONBOARDING.md` para o pipeline e as conclusões da linha de investigação
com o MAE (que esta trilha busca comparar/repetir).

Ambiente: mesmo `requirements-ultrassom.txt` do projeto principal, com
`diffusers==0.40.0`, `accelerate==1.15.0` e `scikit-image==0.26.0`
adicionados. Modelo baixado do Hugging Face Hub (`AutoencoderKL.from_pretrained`),
sem necessidade de checkpoint local.

## Fase 1 — Fidelidade do autoencoder (encode → decode direto)

**Script**: `serve/vae_reconstruction_baseline.py`
**Comando**: `python -m serve.vae_reconstruction_baseline --batch-images 20`
**Pré-processamento**: resolução nativa de cada imagem, só com padding até o
múltiplo de 8 mais próximo (sem resize). Latente `(1, 4, H/8, W/8)`.

**Resultado (20 imagens, `outputs/vae_reconstruction_v0/summary.csv`)**:

| métrica | média |
|---|---|
| MSE | 48.93 |
| PSNR | 31.60 dB |
| SSIM | 0.816 |

Variação entre imagens: PSNR de ~28.8 a ~36.1 dB, SSIM de ~0.70 a ~0.93 —
sem nenhum caso de colapso total (diferente do que se via com o token CLS
do MAE sendo ignorado pelo decoder).

**Observação qualitativa**: inspecionando `case002_compiled.png`, a
reconstrução é visualmente quase indistinguível do original — bordas,
textura de ruído speckle e estruturas anecoicas (ex.: o "buraco" escuro na
imagem) são preservadas. Isso é um resultado bem melhor do que a
reconstrução via MAE, cujo decoder é fortemente dependente do masking usado
no pré-treino e não foi ajustado para reconstrução fiel pixel-a-pixel.

**Conclusão da fase 1**: o VAE do SD, mesmo nunca tendo visto imagens de
ultrassom no pré-treino, reconstrói essas imagens com fidelidade alta
(PSNR ~30+dB, SSIM ~0.8). Isso valida seguir para a fase 2 (tolerância a
ruído) e depois para a fase 3 (regressão tabular).

## Fase 2 — Tolerância a perturbação gradual do latente

**Script**: `serve/vae_perturbation_test.py`
**Comando (single-image)**: `python -m serve.vae_perturbation_test --image case001.png`
**Comando (batch)**: `python -m serve.vae_perturbation_test --batch-images 15`
**Metodologia**: ruído gaussiano por canal, escalado pelo desvio-padrão
daquele canal ao longo das posições espaciais da própria imagem
(`N(0, (fraction * std_c)^2)`), sweep de frações
`[0.1, 0.25, 0.5, 0.75, 1.0, 5.0, 10.0]` — mesma grade de calibração usada
no teste equivalente do MAE (`serve/perturbation_test.py`). Sem modo
`--target cls`: o latente do VAE não tem token de resumo global, então só
existe o modo "perturbar tudo".

**Resultado (case001, `outputs/vae_perturbation_v0/metrics.csv`)**:

| fraction | MSE | PSNR | SSIM |
|---|---|---|---|
| 0.00 (baseline) | 75.7 | 29.34 | 0.756 |
| 0.10 | 87.1 | 28.73 | 0.741 |
| 0.25 | 150.3 | 26.36 | 0.675 |
| 0.50 | 446.7 | 21.63 | 0.513 |
| 0.75 | 1209.3 | 17.31 | 0.344 |
| 1.00 | 2589.2 | 14.00 | 0.223 |
| 5.00 | 11062.0 | 7.69 | 0.042 |
| 10.00 | 12229.6 | 7.26 | 0.028 |

**Resultado agregado (batch de 15 imagens, `python -m serve.vae_perturbation_test --batch-images 15`,
médias em `outputs/vae_perturbation_v0/metrics.csv`)** — confirma o padrão de
case001 acima, com a mesma curva de degradação suave e monotônica:

| fraction | MSE (média) | PSNR (média) | SSIM (média) |
|---|---|---|---|
| 0.00 (baseline) | 55.2 | 30.90 | 0.795 |
| 0.10 | 64.6 | 30.18 | 0.780 |
| 0.25 | 116.2 | 27.55 | 0.717 |
| 0.50 | 368.2 | 22.53 | 0.547 |
| 0.75 | 1024.4 | 18.09 | 0.368 |
| 1.00 | 2152.3 | 14.84 | 0.246 |
| 5.00 | 11971.1 | 7.36 | 0.043 |
| 10.00 | 13916.3 | 6.72 | 0.032 |

**Observação qualitativa** (`case001_grid.png`): a estrutura anatômica
permanece reconhecível até fração ~0.25; entre 0.5 e 0.75 a imagem degrada
visivelmente (textura tipo blocos, perda de contraste); a partir de 1.0 a
reconstrução vira essencialmente ruído — mas, diferente do ruído em tons de
cinza do MAE, aqui aparecem **artefatos de cor** (o decoder é nativamente
RGB, então ruído no latente se traduz em desvios de cor, não só de
luminância). Isso é uma diferença estrutural importante frente ao MAE e
deve ser levada em conta ao interpretar reconstruções da fase 3.

**Conclusão da fase 2**: a degradação é suave e monotônica (sem os saltos
abruptos observados na sensibilidade ao token CLS do MAE) — o espaço latente
do VAE responde de forma mais "contínua" a perturbações, o que é esperado de
um VAE regularizado por KL. O ponto de calibração ~0.25–0.5 (SSIM caindo de
0.82→0.67→0.51) é uma referência útil para avaliar, na fase 3, se o erro do
regressor tabular está dentro ou muito acima da faixa "tolerável" pelo
decoder.

## Fase 3 — Regressão tabular (dados clínicos → latente do VAE)

**Scripts**:
- `serve/generate_embeddings_vae.py` — gera `embeddings_vae.json` (256 imagens,
  resize fixo 256×256 → latente `(1,4,32,32)` por imagem).
- `src/preprocessing/build_tabular_dataset_vae.py` — gera
  `dataset_breast/processed_dataset_embeddings_vae.csv` (262144 linhas =
  256 imagens × 1024 posições espaciais, 135 colunas, sem token CLS).
- `src/models/train_catboost_vae.py` — mesmo CatBoost `MultiRMSE` do
  pipeline do MAE (só que com 4 saídas em vez de 768), split por paciente
  (204 treino / 52 teste).
- `serve/reconstruct_and_compile_vae.py`, `src/evaluation/compare_token_variance_vae.py`.

**Resultado do regressor**:

| métrica | valor |
|---|---|
| MAE | 0.517 |
| RMSE | 0.668 |
| R² | 0.054 |

**Colapso de variância** (`case002.png`, `compare_token_variance_vae.py`):
razão `std_pred/std_real` por canal = `[0.24, 0.36, 0.19, 0.25]` (média 0.26)
— ou seja, ~74% da variância posição-a-posição do latente real se perde na
predição.

**Observação qualitativa** (`outputs/vae_reconstructions/compiled/case002_compiled.png`):
a reconstrução a partir da embedding predita (`infered`) é um borrão quase
uniforme, sem nenhuma estrutura anatômica reconhecível — visualmente
equivalente ao colapso já visto com o MAE.

**Conclusão da fase 3**: o R² (0.054) é **numericamente melhor que zero** e a
razão de colapso de variância (~0.26, i.e. ~26% de variância preservada) é
menos severa que a observada com o MAE (onde a maioria das dimensões tinha
razão < 0.10). Ainda assim, a diferença não é suficiente para produzir uma
reconstrução útil: 26% da variância real está muito abaixo do que o teste de
perturbação da fase 2 tolera sem degradação visível (SSIM só cai de
~0.82→0.67 até fração≈0.25, i.e. ruído com ~25% do desvio-padrão real — aqui
o "erro" do regressor equivale a um ruído de magnitude muito maior que isso
em ~74% da variância). **A causa raiz é a mesma identificada para o MAE**:
dados clínicos tabulares (idade, BI-RADS, ecogenicidade etc.) não carregam
informação espacial sobre o conteúdo visual de cada posição — a posição
`(x,y)` não determina o mesmo conteúdo entre pacientes diferentes. Trocar o
espaço de embedding (MAE → VAE do SD) melhora ligeiramente a fidelidade base
de reconstrução (fases 1–2) mas **não resolve** o problema estrutural da
regressão tabular, que independe do autoencoder usado.

## Resumo geral

| fase | resultado | vs. MAE |
|---|---|---|
| 1. Fidelidade encode→decode | PSNR ~31.6 dB, SSIM ~0.82 (20 imagens) | Muito melhor — reconstrução quase indistinguível a olho nu; o decoder do MAE nunca foi otimizado para fidelidade pixel-a-pixel |
| 2. Tolerância a ruído gradual | Degradação suave e monotônica; estrutura reconhecível até fração ~0.25, colapso entre 0.5–0.75 | Mais "gracioso"/contínuo que o MAE (sem o comportamento de "ignorar o token CLS" observado lá) |
| 3. Regressão tabular (clínico → embedding) | R² = 0.054, ~74% da variância posição-a-posição perdida, reconstrução final é um borrão sem estrutura | Levemente melhor (R²≈0 no MAE) mas **mesma falha estrutural**: dado tabular clínico não carrega sinal espacial |

**Conclusão final**: o VAE do SD é um autoencoder de imagem bem melhor que o
USF-MAE para este dataset (fases 1–2), mas isso não é suficiente para viabilizar
a predição de embeddings a partir de dados tabulares clínicos (fase 3) — o
gargalo nunca foi o autoencoder, é a ausência de sinal espacial nas features
tabulares disponíveis. Qualquer investigação futura nessa linha precisaria de
alguma fonte de sinal espacial (ex.: um subconjunto de pixels/regiões da
própria imagem, não só metadados clínicos) para ter chance de funcionar,
independente do modelo de embedding escolhido.
