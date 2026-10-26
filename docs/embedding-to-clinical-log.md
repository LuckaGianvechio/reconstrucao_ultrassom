# Log de experimentos — prever features clínicas a partir das embeddings (direção inversa)

As duas investigações anteriores mostraram que **prever a embedding da imagem
a partir de dados tabulares clínicos falha** — MAE (`ONBOARDING.md`, R²≈0) e
SD-VAE (`docs/vae-experiment-log.md`, R²=0.054) — porque a posição de um
patch/pixel latente não carrega o mesmo conteúdo visual entre pacientes
diferentes.

Este documento testa a direção **inversa**: usar a embedding da imagem (uma
única embedding por imagem, via pooling) para prever features clínicas
individuais. Ao contrário da tentativa anterior, aqui não há mais "posição"
para colapsar — o alvo é por imagem, não por token.

## Metodologia

- **Embeddings**: reaproveitadas as já geradas em sessões anteriores —
  `embeddings_per_patch.json` (USF-MAE, 197 tokens × 768-dim) e
  `embeddings_vae.json` (SD-VAE, latente fixo `(4,32,32)`).
- **Pooling**: `serve/inference.py`/`src/preprocessing/embedding_pooling.py`
  - MAE **mean**: média dos 196 patch tokens (exclui CLS) — mesma convenção
    já usada em `serve/inference.py::encode_image`.
  - MAE **cls**: token índice 0 isolado.
  - VAE **mean**: média espacial dos 1024 pixels do latente, por canal
    (vetor 4-dim). Sem análogo de CLS (decisão já validada: não inventar um).
- **Alvos clínicos** (`src/preprocessing/clinical_targets.py`):
  regressão (`Age`), multiclasse single-label (`BIRADS`, `Shape`,
  `Echogenicity`, `Posterior_features`, `Halo`, `Calcifications`,
  `Skin_thickening`, `Verification`), e binário multi-label (rótulos
  individuais de `Interpretation`, `Margin`, `Signs`, `Symptoms`,
  `Tissue_composition`, `Diagnosis`, filtrados a taxa de positivos entre
  5%–95%).
- **Modelos**: dois modelos treinados e comparados lado a lado em cada alvo —
  `CatBoostClassifier`/`CatBoostRegressor` (tree-based, mesma família já usada
  no resto do projeto) e um **MLP** (`MLPClassifier`/`MLPRegressor` do
  scikit-learn, com `StandardScaler` antes — o MLP precisa de entrada
  normalizada, diferente de árvores — uma hidden layer pequena `(64,)`,
  `alpha=1e-3` e `early_stopping=True`, dado o dataset pequeno: ~200 amostras
  de treino, até 768-dim de entrada no MAE). Ambos comparados contra o mesmo
  baseline trivial (`DummyClassifier`/`DummyRegressor`) — comparação sempre
  lado a lado (`delta_catboost`/`delta_mlp = metric_model - metric_baseline`),
  já que vários alvos são desbalanceados (ex. `Calcifications` ~88% "no").
- **Métricas reportadas** (por alvo, por modelo e para o baseline): accuracy,
  precision, recall, F1 (`average='binary'` para os alvos binários,
  `average='macro'` para multiclasse) e **ROC AUC** (`predict_proba` — one-vs-rest
  com média macro para multiclasse). `metric`/`delta` continuam usando
  **balanced accuracy** especificamente (é a que fica robusta a desbalanceamento
  e é usada para ranquear os alvos); todas as outras métricas ficam disponíveis
  no CSV completo para inspeção lado a lado.
- **Script**: `python -m src.models.train_embedding_to_clinical --model {mae,vae} --pooling {mean,cls}`.
  Resultados completos em `outputs/embedding_to_clinical/{model}_{pooling}_results.csv`.

## Resultado principal: **ao contrário da direção anterior, aqui há sinal real**

Comparação dos alvos com maior ganho sobre o baseline (`delta` de balanced
accuracy para classificação, de R² para `Age`), nas 3 combinações testadas
(CatBoost — o MLP é comparado na seção seguinte):

| alvo | mae+mean | mae+cls | vae+mean |
|---|---|---|---|
| `Shape` (multiclasse, 3 classes) | 0.292 | 0.261 | 0.110 |
| `Posterior_features` (multiclasse, 4 classes) | 0.233 | **0.338** | -0.008 |
| `Halo` (binário) | 0.202 | **0.326** | 0.127 |
| `Verification` (binário) | **0.248** | 0.169 | 0.077 |
| `BIRADS` (multiclasse, 6 classes) | **0.170** | 0.050 | 0.084 |
| `Age` (regressão, R²) | **0.137** | 0.140 | 0.003 |
| `Echogenicity` (multiclasse, 6 classes) | 0.050 | 0.019 | 0.097 |
| `Calcifications` (multiclasse, quase degenerado) | 0.000 | 0.000 | -0.089 |

**Conclusão imediata**: diferente da direção "tabular → embedding" (onde
nenhum regressor testado saiu de R²≈0), aqui **vários alvos clínicos
mostram sinal consistente e substancial nas embeddings de imagem** —
principalmente `Shape`, `Posterior_features`, `Halo`, `Verification`,
`BIRADS` e `Age`. Isso confirma a hipótese do usuário: a embedding da imagem
carrega informação visual real que se relaciona com essas características
(faz sentido clinicamente — forma da lesão, halo, features posteriores e
BI-RADS são justamente características *visuais* que um radiologista lê
direto da imagem).

**MAE (mean e cls) supera o VAE em quase todos os alvos com sinal forte** —
esperado, já que o MAE foi pré-treinado especificamente em imagens de
ultrassom (domínio-específico), enquanto o VAE do SD nunca viu esse domínio
e sua embedding tem só 4 dimensões (vs. 768 do MAE) após o pooling, uma
capacidade muito menor para carregar informação discriminativa.

**MAE mean vs. cls**: nenhum dos dois pooling domina uniformemente — `cls`
vence em `Posterior_features` e `Halo` (talvez por serem características
mais "globais"/de contorno, que um token de resumo captura bem), `mean`
vence em `Verification` e `BIRADS`. Vale considerar concatenar os dois (ou
usar ambos como features) em uma investigação futura.

### Métricas completas (accuracy, precision, recall, F1, ROC AUC) — CatBoost

Detalhamento dos alvos-chave (baseline sempre no mesmo split; ROC AUC do
baseline trivial é sempre 0.5, como esperado de um classificador constante):

| alvo | combo | accuracy (baseline) | precision | recall | F1 | ROC AUC |
|---|---|---|---|---|---|---|
| `Shape` | mae+mean | 0.765 (0.549) | 0.701 | 0.625 | 0.650 | **0.831** |
| `Shape` | mae+cls | 0.706 (0.549) | 0.641 | 0.594 | 0.612 | 0.742 |
| `Shape` | vae+mean | 0.627 (0.549) | 0.423 | 0.443 | 0.433 | 0.636 |
| `Halo` | mae+mean | 0.843 (0.784) | 0.789 | 0.702 | 0.730 | 0.902 |
| `Halo` | mae+cls | 0.882 (0.784) | 0.826 | 0.826 | 0.826 | **0.943** |
| `Halo` | vae+mean | 0.725 (0.784) | 0.613 | 0.627 | 0.619 | 0.723 |
| `Posterior_features` | mae+mean | 0.647 (0.627) | 0.423 | 0.483 | 0.442 | 0.851 |
| `Posterior_features` | mae+cls | 0.725 (0.627) | 0.493 | 0.588 | 0.521 | **0.890** |
| `Posterior_features` | vae+mean | 0.294 (0.627) | 0.228 | 0.242 | 0.214 | 0.576 |
| `Verification` | mae+mean | 0.863 (0.784) | 0.817 | 0.748 | 0.774 | **0.820** |
| `Verification` | mae+cls | 0.843 (0.784) | 0.824 | 0.669 | 0.703 | 0.818 |
| `Verification` | vae+mean | 0.647 (0.784) | 0.559 | 0.577 | 0.557 | 0.614 |
| `BIRADS` | mae+mean | 0.333 (0.196) | 0.398 | 0.337 | 0.355 | **0.642** |
| `BIRADS` | mae+cls | 0.216 (0.196) | 0.203 | 0.216 | 0.208 | 0.602 |
| `BIRADS` | vae+mean | 0.255 (0.196) | 0.282 | 0.251 | 0.251 | 0.671 |

**Leitura importante da ROC AUC**: ela confirma o sinal em todos os casos
acima (sempre bem acima de 0.5, o valor do baseline), inclusive quando a
accuracy/precision/recall parecem menos impressionantes — ex. `BIRADS`
(multiclasse com 6 classes, accuracy só 0.333) ainda tem ROC AUC macro
0.60-0.67, mostrando que o modelo discrimina bem entre classes mesmo sem
acertar a classe exata na maioria das vezes (esperado num problema
multiclasse difícil com poucas amostras por classe). Já em `Posterior_features`
no VAE, a ROC AUC (0.576) confirma que o sinal ali é realmente fraco — bate
com o `delta` quase nulo (-0.008) visto na tabela anterior.

## CatBoost vs. MLP

Cada alvo foi treinado com os dois modelos (mesma matriz de treino/teste,
mesmo baseline). `delta` = ganho sobre o baseline trivial:

| alvo | delta catboost (mean/cls/vae) | delta mlp (mean/cls/vae) |
|---|---|---|
| `Shape` | 0.292 / 0.261 / 0.110 | 0.169 / -0.031 / -0.067 |
| `Posterior_features` | 0.233 / 0.338 / -0.008 | 0.221 / 0.016 / 0.028 |
| `Halo` | 0.202 / 0.326 / 0.127 | 0.248 / 0.223 / 0.000 |
| `Verification` | 0.248 / 0.169 / 0.077 | 0.152 / 0.165 / 0.000 |
| `BIRADS` | 0.170 / 0.050 / 0.084 | -0.007 / 0.119 / 0.059 |
| `Age` (R²) | 0.137 / 0.140 / 0.003 | -0.169 / -0.233 / **-0.815** |
| `Echogenicity` | 0.050 / 0.019 / 0.097 | 0.044 / 0.142 / 0.011 |

**Média do delta em todos os alvos, por combinação** (`outputs/embedding_to_clinical/{combo}_results.csv`):

| combinação | delta médio CatBoost | delta médio MLP |
|---|---|---|
| mae + mean | 0.141 | 0.091 |
| mae + cls | 0.150 | 0.108 |
| vae + mean | 0.098 | **-0.013** |

**CatBoost supera o MLP na maioria dos alvos e em todas as combinações**,
principalmente em `Age` (regressão) — o MLP **piora em relação ao baseline**
nas três combinações, com colapso severo no VAE (R²=-0.81, o MLP claramente
sobreajusta com só ~170 amostras de treino e early stopping não foi
suficiente para evitar isso). O padrão é o esperado para este regime de
dados: MLPs geralmente precisam de mais exemplos para superar métodos
baseados em árvore como CatBoost, que lidam melhor com poucas centenas de
amostras e não exigem normalização de entrada. Em alguns alvos pontuais o MLP
chega perto ou levemente à frente do CatBoost (`Halo` e `Verification` no
`mae+mean`, `BIRADS` e `Echogenicity` no `mae+cls`), mas não há um padrão
consistente de vantagem do MLP em nenhuma combinação — **CatBoost é a escolha
mais robusta neste dataset** (256 imagens, splits de ~50 amostras de teste).

## ⚠️ Ressalva importante: possível vazamento de metadado nos alvos "not available"/"no"

Vários dos alvos com maior `delta` nas três combinações são rótulos do tipo
`Signs_not available`, `Symptoms_not available`, `Tissue_not available`,
`Symptoms_HRT/hormonal contraception` — ou seja, prevendo se um campo
clínico **estava disponível ou não**, não uma característica visual da
lesão em si. É plausível que isso reflita um viés de coleta de dados (ex.:
imagens de um determinado sub-conjunto/protocolo de aquisição tendem a ter
menos metadados preenchidos), que pode estar correlacionado com
características de imagem *não relacionadas ao diagnóstico* (equipamento,
configuração do aparelho, etc.) — um tipo de vazamento de metadado, não sinal
diagnóstico real. **Os resultados mais interessantes e clinicamente
relevantes são os alvos que descrevem a lesão em si**: `Shape`,
`Posterior_features`, `Halo`, `BIRADS`, `Verification`, `Age` — esses são os
que merecem mais confiança e investigação futura.

## PCA nas embeddings (50/60/70/80% da variância) — não ajuda

Testamos reduzir a embedding com PCA antes de treinar, mantendo só os
componentes principais necessários para explicar 50/60/70/80% da variância
(`--pca-variance`, `src/models/train_embedding_to_clinical.py::apply_pca`).
**O PCA é sempre ajustado (`fit`) apenas no split de treino**, e só depois
usado para transformar (`transform`) tanto treino quanto teste — nunca ajusta
no conjunto de teste.

**Descoberta lateral relevante**: nas embeddings mean-pooled do MAE, o
**primeiro componente principal sozinho já explica ~74% da variância total**
(768 → 1 componente para 50/60/70% de variância, 2 componentes para 80%). Ou
seja, a variância dessas embeddings é dominada por uma única direção — muito
provavelmente algo genérico como brilho/contraste global da imagem, não
necessariamente informação discriminativa para os alvos clínicos.

**Delta médio (CatBoost) por combinação e nível de PCA** (`outputs/embedding_to_clinical/{combo}_pca{N}_results.csv`):

| combinação | sem PCA | PCA 50% (n_comp) | PCA 60% | PCA 70% | PCA 80% |
|---|---|---|---|---|---|
| mae + mean | **0.141** | 0.033 (1) | 0.033 (1) | 0.033 (1) | 0.111 (2) |
| mae + cls | **0.150** | 0.115 (3-4) | 0.150 (5-6) | 0.149 (8) | 0.156 (14) |
| vae + mean | **0.098** | 0.005 (1) | 0.005 (1) | 0.005 (1) | 0.005 (1) |

(o mesmo padrão se repete para o MLP — delta médio sempre pior ou igual com
PCA em toda combinação/nível, inclusive ficando negativo na maioria dos
casos.)

**Conclusão: PCA não melhora a performance em nenhuma combinação testada —
na maioria dos casos, piora, às vezes bastante**:

- **`mae+mean`**: cai de 0.141 (raw, 768-dim) para 0.033 com 1 componente
  (50–70% de variância) — perde a maior parte do sinal, confirmando que o
  componente dominante (a tal variância de ~74%) não é o que carrega a
  informação clinicamente relevante; o sinal útil está nas dimensões de
  menor variância que o PCA agressivo descarta primeiro.
- **`mae+cls`**: é o único caso onde a compressão é "quase de graça" — com
  60-80% de variância (5 a 14 componentes, vs. 768 originais) o desempenho
  fica praticamente igual ao raw (0.150 → 0.150-0.156). Não é uma melhora,
  mas mostra que o sinal do token CLS está concentrado em relativamente
  poucas direções — pode ser útil para reduzir custo computacional sem
  perder desempenho, mas não para melhorá-lo.
- **`vae+mean`**: colapsa de 0.098 para 0.005 em todos os níveis (sempre 1
  componente, já que a embedding original só tem 4 dimensões — reduzir ainda
  mais é extremamente destrutivo).

**Por que PCA não ajuda aqui**: PCA escolhe componentes por variância
explicada, não por relevância para o alvo (não é supervisionado). Quando a
variância é dominada por um fator não-discriminativo (como parece ser o caso
do brilho/contraste geral nas embeddings do MAE), manter só os componentes de
maior variância descarta justamente o sinal útil, que vive em direções de
menor variância. Isso é consistente com a mensagem geral do documento: **usar
a embedding "crua" (sem PCA) e deixar o CatBoost (que já lida bem com alta
dimensionalidade e poucas amostras) escolher o que é relevante continua
sendo a melhor estratégia** neste dataset.

## PCA por posição de patch (concatenando 196 blocos) — ideia do usuário, também não ajuda

Proposta testada: em vez de colapsar os 196 patch tokens do MAE numa única
embedding via média, ajustar um PCA **separado para cada uma das 196
posições de patch** (usando os embeddings 768-dim daquela posição através das
imagens de treino), manter só os componentes que explicam X% da variância
*naquela posição*, e concatenar os `N_i` componentes de todas as posições
num único vetor por imagem. Implementado em
`src/models/train_embedding_to_clinical_patch_pca.py::build_patch_pca_features`
— o PCA de cada posição é ajustado **apenas nos nomes de arquivo do split de
treino** daquele alvo (nunca no teste), do mesmo jeito que o PCA pós-hoc já
usado no resto do documento. Só CatBoost foi testado aqui (o MLP já havia
perdido consistentemente para o CatBoost nas dimensões menores testadas
antes, e seria ainda pior com milhares de features e ~200 amostras).

**Confirmando a preocupação do próprio usuário**: o número de features
explode rapidamente.

| variância mantida por posição | nº de features concatenadas (~) | resultado |
|---|---|---|
| 99.9% | ~30.800 (1 alvo testado: `Age`) | **processo interrompido por falta de memória do sistema** ao rodar a varredura completa nos ~40 alvos (a máquina tinha só ~9GB livres e o swap já estava saturado por outros processos) — rodou individualmente para `Age` (delta=-0.078, **pior que o baseline**) antes de travar a varredura completa |
| 95% | ~11.500 | varredura completa concluída (35 min) — delta médio **0.032** |

**Comparação direta com o pooling por média** (mesmo CatBoost, mesmos alvos,
95% de variância vs. média simples de 768-dim):

| alvo | delta (média, 768-dim) | delta (patch-PCA 95%, ~11.500-dim) |
|---|---|---|
| `Shape` | 0.292 | 0.079 |
| `Posterior_features` | 0.233 | 0.002 |
| `Verification` | 0.248 | 0.000 |
| `Symptoms_not available` | 0.294 | 0.219 |
| **média geral (39 alvos)** | **0.141** | **0.032** |

A média simples vence em 29 dos 39 alvos comparados, e por uma margem grande
na maioria dos casos que importam (`Shape`, `Posterior_features`,
`Verification`, `Symptoms_HRT/hormonal contraception` — estes últimos dois
literalmente colapsam a zero com patch-PCA). **A hipótese do usuário sobre o
risco se confirma na prática**: com ~200 amostras de treino e milhares de
features (mesmo mantendo só 95% da variância por posição), o CatBoost não
consegue extrair sinal melhor do que a média simples de 768 dimensões — e
com 99.9% de variância (~30 mil features) o problema é tão severo que nem foi
possível terminar a varredura completa na máquina disponível, além do único
resultado obtido (`Age`) já ter ficado pior que o baseline.

**Por que a média simples (mais "grosseira") vence**: apesar de descartar
toda a informação espacial entre patches, a média por dimensão já é, na
prática, uma forma de redução de variância/ruído (agrega 196 amostras
correlacionadas por dimensão) que produz um vetor denso e estável de 768
features — uma proporção amostras/features muito mais favorável (256/768 ≈
0.33) do que a concatenação por posição (256/11500 ≈ 0.02, ou 256/30800 ≈
0.008 a 99.9%). A informação espacial que a concatenação tenta preservar
existe, mas o custo em dimensionalidade supera, de longe, o benefício com
este tamanho de dataset.

## UMAP (manifold learning não-linear) — melhor que PCA, ainda pior que a embedding crua

Pergunta do usuário: t-SNE/UMAP são normalmente usados só para visualização —
dá para usar algum deles com o mesmo protocolo `fit` no treino / `transform`
no teste que usamos com PCA?

- **t-SNE não serve**: é transdutivo — otimiza um embedding fixo para o
  conjunto de dados que recebe, sem um `.transform()` de verdade para pontos
  novos. "Transformar" um caso de teste exigiria reotimizar tudo incluindo
  ele, o que vaza o teste no ajuste.
- **UMAP serve**: tem `fit`/`transform` genuínos — aprende a variedade a
  partir do grafo de vizinhos mais próximos no treino e projeta pontos novos
  nela sem reotimizar. Implementado em
  `src/models/train_embedding_to_clinical.py::apply_umap` (`--umap-components`,
  mutuamente exclusivo com `--pca-variance`) — `StandardScaler` antes (UMAP é
  baseado em distância) e `n_neighbors = min(15, n_treino-1)`, ambos
  ajustados só no split de treino de cada alvo.

Testado só para o MAE (mean e cls — o VAE já tem só 4 dimensões, não faz
sentido reduzir mais), em 10/30/50/100 componentes:

| pooling | raw (768) | UMAP 10 | UMAP 30 | UMAP 50 | UMAP 100 |
|---|---|---|---|---|---|
| mean (delta médio CatBoost) | **0.141** | 0.123 | 0.118 | 0.112 | 0.084 |
| cls (delta médio CatBoost) | **0.150** | 0.133 | 0.095 | 0.092 | 0.097 |

**UMAP é bem menos destrutivo que o PCA** (lembrando: PCA a 50-70% de
variância já derrubava o delta médio para ~0.033 no mean-pooling, um colapso
muito mais forte) — mas em nenhum nível testado o UMAP supera a embedding
crua de 768 dimensões. Curiosamente o delta piora à medida que aumentamos o
número de componentes do UMAP (de 10 para 100), o oposto do que se esperaria
de uma redução "menos agressiva" — isso é consistente com o UMAP otimizar
para preservar a topologia de vizinhança local no espaço original (tarefa
puramente não-supervisionada, sem nenhuma noção do alvo clínico), não para
maximizar variância nem informação preditiva; mais dimensões não
necessariamente aproxima melhor do "raw" para fins supervisionados, e o
resultado também tem alguma instabilidade inerente à otimização estocástica
do UMAP (temos só uma repetição por configuração, não uma média de várias
sementes).

**Conclusão**: nem PCA nem UMAP melhoram sobre a embedding crua — o UMAP
confirma que a causa da degradação com PCA não era só "linearidade vs.
não-linearidade" da técnica, e sim o mesmo problema de fundo já identificado:
qualquer redução de dimensionalidade não-supervisionada, aplicada a um
conjunto de ~200 amostras de treino, tende a descartar informação que o
CatBoost já conseguia aproveitar na embedding original de 768 dimensões.

## Conclusão geral

A hipótese do usuário se confirma: **usar a embedding da imagem para prever
características clínicas funciona muito melhor do que o caminho inverso**
testado anteriormente. Isso é consistente com a intuição de que a imagem
carrega informação visual que determina essas características (e não o
contrário). USF-MAE (domínio-específico) supera o VAE do SD genérico na
maioria dos alvos com sinal real, mas ambos mostram sinal acima do baseline
em vários alvos — o que valida o pooling de embeddings como uma direção
promissora para os próximos passos (ex.: um classificador auxiliar de
triagem por características visuais, ou uma investigação mais profunda com
mais dados/embeddings de maior dimensão para o VAE). Entre os dois modelos
testados, **CatBoost é consistentemente melhor que o MLP** neste tamanho de
dataset (256 imagens) — o MLP não teve dados suficientes para superar um
método baseado em árvore, e chegou a piorar bastante em relação ao baseline
na regressão de `Age`. Reduzir a dimensionalidade com PCA (50–80% da
variância) **não melhorou o desempenho em nenhum caso** — na maioria das
combinações, piorou bastante, porque a variância dominante das embeddings
não coincide com o sinal relevante para os alvos clínicos. Na direção
oposta — tentar preservar *mais* detalhe espacial via PCA por posição de
patch e concatenação (196 blocos) — o resultado foi ainda pior (delta médio
0.032 vs. 0.141 da média simples), exatamente pelo motivo antecipado
antes de testar: poucas centenas de amostras não sustentam milhares de
features. Testamos também UMAP (manifold learning não-linear, com
`fit`/`transform` genuíno — diferente do t-SNE, que não suporta isso) em
10-100 componentes: é bem menos destrutivo que o PCA, mas também não supera
a embedding crua em nenhum nível. **A média simples de 768 dimensões
continua sendo a melhor representação testada até agora para o MAE** — todo
método de redução de dimensionalidade tentado (PCA global, PCA por posição
de patch, UMAP) piorou o desempenho em relação a usar a embedding crua
diretamente com CatBoost.

Nota: o UMAP **não** é inútil no projeto como um todo — ele só não ajuda
*aqui*, onde a embedding é a **entrada**. Quando o UMAP é usado do outro lado,
projetando a embedding para servir de **alvo** da regressão a partir do dado
tabular, ele produz o primeiro sinal real já obtido naquela direção
(R² ≈ 0.29 contra ≈ 0 do setup por-patch original) — ver
`docs/tabular-to-umap-log.md`.
