# LOGS — diário de experimentos

Registro cronológico de alterações de código e conclusões de experimentos.
Organizado como **dia → experimento → variação**.

Os documentos em `docs/` continuam sendo a análise aprofundada por tema; este
arquivo é o índice cronológico que aponta para eles e que deixa visível o que uma
correção posterior invalidou retroativamente.

Convenção: dia mais recente no topo. Cada variação traz o resultado em uma linha,
e conclusões que mudaram depois ficam marcadas com ⚠️ e um link para a entrada que
as corrigiu (em vez de serem apagadas — o histórico do que se acreditou importa).

---

## 2026-10-21

### Experimento: embedding + pré-US para prever as variáveis de Imagem (piloto LightGBM)

**Pergunta do usuário.** Usar a embedding + informação clínica para prever as 23 variáveis do grupo
`image`. Patologia **fica de fora** por decisão do usuário: é obtida depois do exame. Sobra o grupo
Histórico = pré-US (`Age`, `Signs`, `Symptoms`), com fusão precoce e tardia.

**Código.** `src/evaluation/image_targets_with_pre_us.py` — os mesmos 8 conjuntos de features e as
mesmas comparações de `birads_analysis`, sobre todos os alvos `image`. Os folds são por alvo, como em
`acquisition_confound`, e compartilhados pelos conjuntos. `--model` escolhe CatBoost/LightGBM
(`fit_predict`/`stacked_fit_predict` de `birads_analysis` ganharam `n_classes` e `model_name`; o
padrão não muda nada no BIRADS); `--retest` refaz só os testes.

**Variação: LightGBM** (piloto, decisão do usuário: ver se compensa antes das ~5 h de CatBoost) →
`image_targets_with_pre_us_lgbm/`, 22 min, 6 workers. Um fold descartado (`Posterior_features`,
agrupada). O LightGBM fica ~0.01 abaixo do CatBoost na embedding (0.703 vs. 0.714 estrat.;
0.670 vs. 0.683 agrup.), como no sweep.

| ROC-AUC médio (23 alvos) | Aquisição | Pré-US | Aq. + pré-US | Embedding | Emb. + pré-US (precoce) | Emb. + pré-US (tardia) | Emb. + aq. + pré-US (precoce) |
|---|---|---|---|---|---|---|---|
| estratificada | 0.634 | 0.705 | 0.712 | 0.703 | **0.726** | 0.722 | 0.724 |
| agrupada `Pixel_size` | 0.573 | 0.683 | 0.688 | 0.670 | 0.694 | 0.674 | **0.698** |

- **O pré-US sozinho prevê as variáveis de imagem tão bem quanto a embedding, na média**
  (0.705 vs. 0.703; 0.683 vs. 0.670), significativo contra o acaso em 15 / 6 dos 23 alvos.
  Não é vazamento, é correlação clínica:
  - `Skin_thickening` (pré-US 0.86 / 0.89): dos 15 casos positivos, 10 têm sinal de pele no exame
    físico (cicatriz, retração de pele, *peau d'orange*, vermelhidão) — o mesmo achado visto pelo
    toque e pela inspeção;
  - composição do tecido (`homogeneous: fibroglandular` 0.87 / 0.84): idade mediana de 39 anos vs.
    57,5 nas demais — mamas jovens são fibroglandulares;
  - malignidade (`Suspicion of malignancy` 0.80 / 0.79, margens, BIRADS): idade e lesão palpável,
    como na análise do BIRADS.
- **A embedding supera o pré-US onde o achado é morfologia pura**, nos dois esquemas: `Halo`
  (+0.11 / +0.17), `Margin_spiculated` (+0.16 / +0.14), `Posterior_features` (+0.11 / +0.09),
  `Echogenicity` (+0.02 / +0.08). O pré-US vence em tecido, pele e cistos/fibroadenoma.
- **Somar as duas melhora na média, mas não em nenhum alvo isolado com q < 0.05.** Emb. + pré-US
  (precoce) − embedding: +0.023 / +0.024, positivo em 19 / 18 dos 23 alvos; − pré-US: +0.021 /
  +0.011, positivo em 16 / 13. Só `Tissue_homogeneous: fibroglandular` passa (tardia + aquisição,
  estrat., q = 0.030). A consistência entre alvos sugere ganho real pequeno, mas os alvos são
  correlacionados, então a contagem não é um teste.
- ⚠️ **Aqui a fusão precoce empata com a tardia ou vence**, ao contrário do BIRADS (onde a precoce
  diluía o pré-US). Na agrupada a tardia é pior (0.674 vs. 0.694). Não há uma fusão melhor em
  geral — depende do alvo.

⚠️ A conclusão sobre fusão precoce vs. tardia muda com o CatBoost (variação abaixo).

**Variação: CatBoost** (2026-10-22) → `image_targets_with_pre_us/`, 2h46 (a estimativa de ~5 h
estava alta), 6 workers, lançado como serviço systemd de usuário (`systemd-run --user
--unit=image-targets-catboost`, sobrevive ao fim da sessão). Sanidade: embedding 0.714 / 0.683 e
aquisição 0.634 / 0.606, idênticas a `acquisition_confound`. Mesmo fold descartado
(`Posterior_features`, agrupada).

| ROC-AUC médio (23 alvos) | Aquisição | Pré-US | Aq. + pré-US | Embedding | Emb. + pré-US (precoce) | Emb. + pré-US (tardia) | Emb. + aq. + pré-US (precoce) |
|---|---|---|---|---|---|---|---|
| estratificada | 0.634 | 0.725 | 0.727 | 0.714 | 0.728 | **0.742** | 0.735 |
| agrupada `Pixel_size` | 0.606 | 0.684 | 0.686 | 0.683 | 0.699 | **0.705** | 0.704 |

- **Confirma o piloto: pré-US ≈ embedding na média** (0.725 vs. 0.714; 0.684 vs. 0.683), com a
  mesma divisão — embedding vence na morfologia (`Halo` +0.14 / +0.23, `Margin_spiculated` +0.12 /
  +0.17, `Posterior_features` +0.11 / +0.15, `Margin_microlobulated` +0.08 agrup.); pré-US vence em
  cistos, tecido, pele e `Suspicion of malignancy`.
- **Com CatBoost, a fusão tardia é a melhor na média** (+0.017 / +0.021 sobre o pré-US; +0.028 /
  +0.022 sobre a embedding), ao contrário do LightGBM. Mas ela **perde para a embedding sozinha
  justamente nos alvos de morfologia** (agrup.: `Margin_spiculated` −0.065, `Posterior_features`
  −0.060, `Margin_microlobulated` −0.056, `Halo` −0.038): o score da embedding na fusão tardia vem
  de modelos treinados em 4/5 do fold (CV interna), mais fracos que o direto. Nesses alvos, a
  fusão precoce preserva o desempenho da embedding.
- **Significância: quase nada.** Fora `pre_us vs. acaso` (15 / 3 alvos), só 2 de 460 testes com
  q < 0.05: `Posterior_features` embedding vs. aquisição (estrat., +0.196) e
  `Complex cyst` emb. + pré-US (precoce) vs. embedding (agrup., +0.081). Nenhum ganho da fusão
  tardia sobre o melhor componente passa (menor q = 0.45).

**Leitura.** Somar o pré-US à embedding melhora a predição das variáveis de imagem em média
(~+0.02), sem ganho detectável em alvos isolados com 5 folds. As duas fontes cobrem alvos
diferentes: o pré-US (idade, sinais do exame físico) explica tecido, pele e suspeita de
malignidade; a embedding explica a morfologia da lesão (margens, halo, sombra/reforço posterior),
que é onde ela traz informação própria. Não há uma fusão melhor para todos os alvos.

### Experimento: o modelo dá score de risco de malignidade melhor que o BIRADS do médico?

**Pergunta do usuário.** Comparar o modelo com o BIRADS real (GT) quanto à malignidade
(`Classification`). ⚠️ Usar o modelo que **prevê o BIRADS ≥4c** tem um teto embutido: ele é uma
cópia com ruído do médico. Lido direto das predições fora do fold da análise abaixo: BIRADS do
médico 0.930; melhor modelo-de-BIRADS 0.842 (ordinal, fusão tardia, estrat.) / 0.810 (agrup.).
Por isso o desenho treina **direto na malignidade** e faz três perguntas.

**Código.** `src/evaluation/malignancy_vs_birads.py` (`--rewithin` refaz só a seção 3 a partir de
`oof_predictions.csv`) → `outputs/embedding_to_tabular_experiments/malignancy_vs_birads/`.
22 min, 6 workers sob `MemoryMax=10G`. Mesmos 252 casos (98 malignos) e **mesmos folds** de
`birads_analysis`; mesmo CatBoost; fusão tardia igual. `radiologist` = posto do BIRADS (0–5) usado
como score, sem modelo. Testes: t corrigido de Nadeau–Bengio por fold, BH por família × esquema.

#### 1. Modelo treinado na malignidade vs. BIRADS do médico — ROC-AUC

| Score | Estrat. | Agrup. `Pixel_size` |
|---|---|---|
| **BIRADS do médico** | **0.931** | **0.937** |
| Pré-US | 0.812 | 0.756 |
| Embedding | 0.754 | 0.736 |
| Emb. + pré-US (tardia) | 0.796 | 0.765 |
| Aquisição | 0.700 | 0.592 |

**Não.** O melhor modelo fica 0.12 (estrat., pré-US) a 0.17 (agrup., emb. + pré-US) abaixo do médico; com q < 0.05 em todos os conjuntos na
agrupada e em pré-US / aquisição + pré-US na estratificada (os demais: p = 0.045–0.099, q = 0.058–0.099). Não
surpreende: o médico vê a imagem em resolução cheia, em tempo real, com experiência; a embedding é
de 224×224 sem proporção preservada.

#### 2. Valor incremental sobre o BIRADS

| Score | Estrat. | Agrup. |
|---|---|---|
| BIRADS do médico | 0.931 | 0.937 |
| BIRADS + pré-US | 0.946 | 0.943 |
| BIRADS + embedding (tardia) | 0.944 | 0.943 |
| BIRADS + embedding + pré-US (tardia) | **0.952** | **0.944** |

O maior ganho é +0.021 (estrat., p = 0.019, **q = 0.075**); na agrupada +0.007 (p = 0.31). A
embedding sobre BIRADS + pré-US acrescenta +0.005 / +0.001: **o pouco que se ganha vem do
pré-US** (idade, palpável), não da imagem — coerente com o médico já ter extraído da imagem o que
a embedding carrega.

#### 3. Zona cinzenta: discriminação dentro de cada categoria

🐛 Corrigido antes de registrar: a 1ª versão juntava predições fora do fold de 5 modelos
diferentes dentro de uma categoria; como cada fold tem escala própria, o AUC media em parte a
diferença entre folds — o modelo `birads` (entrada constante na categoria, AUC correto 0.5) saía
com 0.21–0.43. Agora só contam pares maligno–benigno **do mesmo fold**; `birads` volta a 0.50.
IC 95% por bootstrap (2000, reamostrando dentro de cada classe).

| Zona (malignos / n) | Score | Estrat. | Agrup. |
|---|---|---|---|
| 4b (16 / 46) | Pré-US | 0.68 [0.46, 0.88] | 0.61 [0.38, 0.84] |
| | Embedding | 0.65 [0.42, 0.85] | 0.50 [0.27, 0.72] |
| | BIRADS + emb. + pré-US | 0.69 [0.46, 0.88] | 0.73 [0.51, 0.90] |
| 4c (41 / 49; 8 benignos) | Pré-US | 0.83 [0.59, 0.99] | 0.85 [0.65, 1.00] |
| | Embedding | 0.73 [0.44, 0.96] | 0.62 [0.38, 0.90] |
| 5 (40 / 46; 6 benignos) | Pré-US | 0.80 [0.55, 0.97] | 0.39 [0.07, 0.90] |
| | Embedding | 0.85 [0.61, 1.00] | **0.89 [0.68, 1.00]** |
| 4a–4c (58 / 139) | BIRADS do médico | 0.88 [0.82, 0.93] | 0.89 [0.84, 0.95] |
| | BIRADS + pré-US | 0.90 [0.84, 0.95] | 0.92 [0.86, 0.97] |
| | BIRADS + emb. + pré-US | 0.90 [0.84, 0.95] | 0.93 [0.87, 0.97] |

- **No 4b**, a categoria de maior incerteza, nenhum score separa com segurança: os ICs encostam em
  0.5. A embedding sozinha não separa na agrupada (0.50).
- **No 4c**, o pré-US separa os 8 benignos dos 41 malignos (0.83–0.85, IC acima de 0.59 nos dois
  esquemas) — provavelmente pela idade.
- **No 5**, a embedding é o único score estável entre esquemas (0.85 / 0.89, IC acima de 0.61):
  ela reconhece alguns dos 6 benignos classificados como 5. Com 6 casos, é sugestivo, não
  conclusivo. Os conjuntos com fusão oscilam muito na agrupada neste estrato (0.27–0.82).

**Leitura.** O modelo não substitui o BIRADS como score de risco (melhor modelo 0.77–0.81 vs. 0.93). Como
complemento, o ganho é pequeno e não significativo, e vem do pré-US. O único sinal de informação
própria da imagem que o médico não codificou no BIRADS está nos falsos positivos do BIRADS 5
(6 casos). ⚠️ Viés de verificação: benignos BIRADS 2–3 foram confirmados por acompanhamento, e o
próprio BIRADS decidiu quem foi biopsiado — o viés favorece o médico.

### Experimento: BIRADS a fundo — categorias, binarização, pré-ultrassom e ganho da embedding

**Motivação.** Os pesquisadores do laboratório avaliaram que as variáveis com q < 0.05 de 2026-09-30
(`Posterior_features`, `Skin_thickening`...) têm pouca utilidade clínica, mas querem explorar a
predição do **BIRADS** de outras formas. Cinco perguntas: (1) há ganho ao separar o BIRADS em
categorias? (2) faz sentido binarizar? (3) como se distribui o score dentro de cada categoria?
(4) qual a performance só com informação coletada **antes** do ultrassom? (5) somar pré-US à
embedding dá ganho, e com significância?

**Código.** `src/evaluation/birads_analysis.py` (treino + testes; `--retest` refaz só os testes a
partir de `results_folds.csv`) e `src/evaluation/birads_figures.py` (só lê os CSVs). Saídas em
`outputs/embedding_to_tabular_experiments/birads_analysis/` (`results_folds.csv`, `results.csv`,
`oof_predictions.csv`, `significance_tests.csv`, `figures/fig1–fig4`). 2h27 com 4 workers sob
`MemoryMax=10G` (1 h foi suspensão do notebook com a tampa fechada — o processo só pausa).

**Montagem.**
- 252 casos (BIRADS 1, 4 casos "normal", fora como antes). `mae_cls` bruto + CatBoost (mesmos
  hiperparâmetros), 5 folds, CV estratificada e agrupada por `Pixel_size`. Os folds são construídos
  **uma vez** a partir do BIRADS multiclasse e compartilhados por todos os alvos, métodos e conjuntos
  de features → toda comparação é pareada por fold. Nenhum fold inválido.
- Alvos: multiclasse (6 classes); **ordinal** (regressão no posto 0–5, Spearman); 6 binários
  um-contra-o-resto dedicados (`is_X`); 3 cortes binários dedicados `≥4a`, `≥4b`, `≥4c` (a base não
  tem 4d; o usuário pediu os três). Os mesmos `is_X`/`≥corte` também são **lidos** do modelo
  multiclasse (soma das probabilidades) e do ordinal, para comparar com o modelo dedicado.
- Conjuntos de features (decisão do usuário: baselines separados, embedding por cima):

| Conjunto | Features |
|---|---|
| `acquisition` | largura, altura, `Pixel_size` |
| `pre_us` | `Age` (NaN se ausente) + one-hot de `Signs` e `Symptoms` (17 colunas; `not available` como indicador próprio) |
| `acquisition+pre_us` | os dois |
| `embedding` | `mae_cls` 768-d |
| `embedding+pre_us`, `embedding+acquisition+pre_us` | **fusão precoce**: tudo concatenado |
| `emb_score+pre_us`, `emb_score+acquisition+pre_us` | **fusão tardia**: o score da embedding (fora do fold, via CV interna de 5 folds no treino) entra como feature ao lado do pré-US |

  A fusão tardia não estava no plano inicial: entrou porque o smoke test mostrou a fusão precoce
  **piorando** o pré-US (768 dimensões diluem 17 colunas nos splits do CatBoost).
- Testes: t pareado corrigido de Nadeau–Bengio (`corrected_t` de `significance_tests.py`), 5 folds,
  t com 4 g.l. BH dentro de (comparação × esquema), separando os alvos principais (multiclasse,
  ordinal, binários dedicados; 11 testes) dos scores derivados.

**Sanidade.** Multiclasse com a embedding: 0.669 estrat. / 0.635 agrup.; aquisição 0.593 / 0.595 —
idênticos a `acquisition_confound`.

#### Resultado — ROC-AUC médio (Spearman no ordinal)

| Alvo | CV | Aquisição | Pré-US | Aq. + pré-US | Embedding | Emb. + pré-US (precoce) | Emb. + pré-US (tardia) | Emb. + aq. + pré-US (tardia) |
|---|---|---|---|---|---|---|---|---|
| Multiclasse (macro) | estrat. | 0.593 | 0.665 | 0.684 | 0.669 | 0.673 | 0.689 | **0.700** |
| | agrup. | 0.595 | 0.655 | **0.680** | 0.635 | 0.650 | 0.649 | 0.667 |
| Ordinal (Spearman) | estrat. | 0.203 | 0.579 | 0.570 | 0.548 | 0.573 | 0.641 | **0.646** |
| | agrup. | 0.085 | 0.491 | 0.495 | 0.451 | 0.493 | **0.550** | 0.509 |
| `≥4a` | estrat. | 0.643 | 0.812 | 0.757 | 0.687 | 0.756 | **0.822** | 0.788 |
| | agrup. | 0.593 | **0.793** | 0.768 | 0.665 | 0.725 | 0.769 | 0.788 |
| `≥4b` | estrat. | 0.630 | 0.798 | 0.771 | 0.742 | 0.770 | **0.825** | 0.784 |
| | agrup. | 0.574 | 0.770 | 0.767 | 0.722 | 0.743 | **0.788** | 0.777 |
| `≥4c` | estrat. | 0.632 | 0.758 | 0.749 | 0.797 | 0.806 | **0.826** | 0.807 |
| | agrup. | 0.584 | 0.735 | 0.766 | 0.789 | **0.800** | 0.785 | 0.780 |

Por categoria (binário dedicado, CV agrupada): BIRADS 2 — pré-US 0.748, embedding 0.710, tardia
0.761; **3** — 0.776 / 0.620 / 0.688; **4a** — 0.591 / 0.574 / 0.593; **4b** — 0.522 / 0.611 / 0.504;
**4c** — 0.707 / 0.597 / 0.729; **5** — 0.694 / 0.784 / 0.800.

#### Respostas

1. **Separar em categorias não traz ganho.** Nenhum binário dedicado supera, com q < 0.05, o mesmo
   score lido do modelo multiclasse ou do ordinal (0 de 144 comparações em cada esquema). Na CV
   estratificada, o score ordinal empata ou vence o binário dedicado em 55 de 72 comparações. O que
   os modelos captam é um **grau de suspeita ordenado**, não a identidade de cada categoria: as
   extremas (2, 3, 5) são previsíveis (0.70–0.85), e as do meio, **4a e 4b, ficam perto do acaso**
   (0.50–0.62) com qualquer fonte de informação.
2. **Binarizar faz sentido, e o corte importa.** Os três cortes dão ROC-AUC de 0.77–0.83 com a
   melhor combinação. ⚠️ Esse valor não é comparável ao 0.67 do multiclasse macro, que mede outra
   tarefa; a comparação justa é o mesmo corte lido do multiclasse, que fica 0.01–0.06 abaixo
   do binário dedicado. **O corte define qual fonte ajuda**: em `≥4a` (indicação de biópsia) o
   pré-US sozinho é o melhor (0.81 / 0.79) e a embedding é fraca (0.69 / 0.67). Em `≥4c` (suspeita
   alta) a embedding é a melhor fonte isolada (0.80 / 0.79) e supera o pré-US (0.76 / 0.74). A
   imagem separa o 4c–5 do resto, mas não o 4a do 3.
3. **Distribuição do score** (`fig3_distribuicao_score_*.png`): pré-US e fusão tardia crescem de
   forma monotônica de 2 a 4c, com 4c ≈ 5. A embedding sozinha é **mal calibrada** em `≥4a`
   (mediana > 0.7 já no BIRADS 2 e 3) e só separa a partir do 4c em `≥4c`. As categorias 4a e 4b se
   sobrepõem nas três fontes, o que é coerente com a resposta 1.
4. **Pré-ultrassom sozinho prevê bem**, significativo contra o acaso em todos os alvos principais,
   exceto `is_4a`, `is_4b` (e `is_4c` na agrupada): multiclasse 0.665 / 0.655; `≥4a` 0.812 / 0.793
   (q ≤ 0.003). **Resiste ao agrupamento por lote** (perde ≤ 0.03). Importância no CatBoost: `Age`
   (29–43%), `Signs_no` e `Signs_palpable` (~17% cada); os indicadores `not available` somam ~9–19%
   e são o único componente suspeito de lote. Somar a aquisição ao pré-US não acrescenta nada
   (nenhum q < 0.05).
5. **Somar a embedding ao pré-US não tem ganho significativo**, em nenhum alvo, esquema ou tipo de
   fusão (menor q = 0.072).
   - Fusão precoce **piora** o pré-US nos cortes baixos (`≥4a` −0.056 / −0.068).
   - Fusão tardia, nos cortes: +0.01 a +0.07 na estratificada e −0.02 a +0.05 na agrupada. Os mais
     próximos de significância são `≥4c` estrat. (+0.067, p = 0.007, q = 0.072) e BIRADS 5 estrat.
     (+0.117, p = 0.055); nenhum dos dois se mantém na agrupada (q ≥ 0.74).
   - Sobre `aquisição + pré-US`, a embedding acrescenta no máximo +0.08 (`is_5`, q = 0.22).
   - Na direção inversa, somar pré-US à embedding (tardia) também fica sem q < 0.05: `≥4a` +0.135
     (p = 0.089) estrat. / +0.105 agrup.

**Leitura.** Para o BIRADS, a informação clínica pré-exame (sobretudo idade e lesão palpável) é a
fonte mais forte nos cortes de conduta (≥4a, ≥4b); a embedding só compete na ponta de alta suspeita
(≥4c, BIRADS 5). As duas são **parcialmente complementares** — a fusão tardia é a melhor ou empata
em quase todos os alvos —, mas com 252 casos e 5 folds o ganho não é detectável. ⚠️ O teste tem
pouco poder: com 4 g.l. e o erro corrigido, um ganho de 0.05 só passa com dp das diferenças entre folds < ~0.027
para passar. Como em 2026-09-30, "não significativo" aqui é "não detectável".

**Próximo passo** (se o BIRADS continuar o foco): CV repetida (10×5) só nas comparações de fusão
tardia × pré-US em `≥4c` e BIRADS 5, as candidatas mais fortes — custo maior pela CV aninhada
(~6× por fit); calibrar o score da embedding; testar o pré-US sem os indicadores `not available`
para isolar o componente de lote.

---

## 2026-09-30

### Experimento: teste de hipótese do ganho sobre a aquisição

**Motivação.** O critério "ganho significativo" usado até aqui (ganho de ROC-AUC sobre o baseline
de aquisição maior que 1 dp da embedding entre folds) é uma heurística, sem p-valor. Passa a se
chamar **"ganho acima da margem"**, e "significativo" fica reservado ao teste abaixo.

**Teste** (`src/evaluation/significance_tests.py` → `acquisition_confound/significance_tests.csv`;
seção 9.1 do relatório). Só para a estratégia global (`mae_cls` · raw · CatBoost), escolhida pela
média antes de olhar cada variável; a melhor-por-variável não é testada porque a seleção entre 88
combinações enviesaria o p. Usa os resultados por fold já salvos, sem retreinar.
- Estimador: média das diferenças pareadas fold a fold (embedding − aquisição, mesmas partições).
- Erro-padrão: corrigido para CV (Nadeau & Bengio, 2003) — variância × (1/k + n_teste/n_treino),
  porque os folds compartilham dados de treino.
- Referência: t de Student com k − 1 g.l. (k = folds válidos); com 5 folds, crítico 2.78 em vez de 1.96.
- Múltiplos testes: q de Benjamini–Hochberg dentro de cada grupo × esquema de CV; significativo = q < 0.05.
- Folds agrupados em que o ROC-AUC multiclasse é indefinido (classe ausente no teste) são descartados
  (`Posterior_features` agrupada: 4 folds).

**Resultado — variáveis de imagem (23), ganho positivo:**

| CV | Acima da margem | p < 0.05 | q < 0.05 |
|---|---|---|---|
| estratificada | 9 | 2 | **1** (`Posterior_features`, p = 0.0015, q = 0.035) |
| agrupada por tamanho | 11 | 2 | 0 |
| agrupada por `Pixel_size` | 8 | 4 | **0** (`Skin_thickening` p = 0.0027, q = 0.061; `Margin_spiculated`, `Posterior_features`, `Echogenicity` com p < 0.05) |

Histórico, CV estratificada: `Signs_no` e `Signs_palpable` significativos (q = 0.010); nenhum na agrupada.

⚠️ **O critério da margem é bem mais liberal que o teste formal.** As "8 variáveis com ganho robusto"
(2026-09-28) são ganho acima da margem, não ganho significativo. Com 5 folds o teste tem pouco poder:
"não significativo" aqui é "não detectável com esta amostra", não "sem ganho". As evidências mais
consistentes de conteúdo além da aquisição ficam em `Posterior_features` e `Skin_thickening`.
Próximo passo para ganhar poder sem mais dados: CV repetida com várias sementes.

### Relatório consolidado da predição de variáveis a partir das embeddings

`python -m src.evaluation.build_report` → `outputs/embedding_to_tabular_experiments/report/`
(`report.md`, 10 figuras em `figures/`, 8 tabelas em `tables/`). Só lê os CSVs de
`gbm_pca_umap_sweep/` e `acquisition_confound/`; nada é retreinado, e os números do texto saem
dos CSVs. Base para a revisão do projeto em docs.

Conteúdo: dicionário das variáveis (colunas clínicas, grupos, como cada uma foi modelada,
classes usadas e descartadas); modelos de embedding, dimensões e pooling; reduções de
dimensionalidade; modelos de inferência; protocolo de avaliação; comparações entre
embeddings, reduções e modelos (Fig. 2–5); melhor combinação por variável vs. aquisição
(Fig. 6, otimista); melhor estratégia global (`mae_cls` · raw · CatBoost) por variável vs.
aquisição, na CV estratificada e na agrupada (Fig. 7–8); fator de confusão (Fig. 9–10);
limitações. Todos os grupos de variáveis entram, separados por grupo.

Decisões com o usuário: Markdown + PNG (figuras coláveis no docs); só dados existentes — a
grade completa aparece na CV estratificada, e a CV agrupada só para a melhor estratégia.

Achados ao consolidar:
- ⚠️ A contagem de variáveis com ganho robusto na CV agrupada é **8**, não 5 (corrigido em
  2026-09-28 abaixo).
- "PCA > UMAP no mesmo k" vale nas variáveis de imagem (9 de 9 comparações); nos grupos menores
  (patologia, histórico) a ordem varia.
- A coluna `Classification` (benigno 154 / maligno 98 / normal 4) **nunca foi usada como alvo**
  — provavelmente a variável clínica mais relevante; fica como próximo passo.
- ⚠️ A grade tem **88** combinações por variável, não 91 como escrevi em 2026-09-26.

### Análise: quais estratégias aparecem mais entre as melhores por variável

`python -m src.evaluation.analyze_best_per_variable` → `report/tables/tab_estrategias_vencedoras_*.csv`;
também na seção 8.1 do relatório e na página. CV estratificada, 34 variáveis não-sentinela.

Como 5 folds deixam o vencedor mal definido, além de contar vencedores considerei as
**quase-melhores**: combinações dentro de 1 dp (entre folds) da melhor. São, na mediana, **26 das
88** por variável — o "vencedor" quase nunca se distingue das demais. **Enriquecimento** =
participação do nível entre as quase-melhores ÷ participação na grade (1 = acaso).

| Fator | Nível | Vitórias (34) | Enriquecimento |
|---|---|---|---|
| Embedding | `mae_mean` / `mae_cls` | 15 / 14 | ×1.29 / ×1.29 |
| | `vae_flat` / `vae_mean` | 4 / 1 | ×0.48 / ×0.60 |
| Redução | raw | 23 | ×1.50 |
| | PCA | 10 (5 com k = 30) | ×1.04 |
| | UMAP | 1 | ×0.74 |
| Modelo | CatBoost | 14 | ×1.37 |
| | XGBoost / LightGBM | 9 / 9 | ×1.15 / ×1.12 |
| | MLP | 2 | ×0.36 |

**Escolher por variável compensa pouco** (separado por k, em `tab_estrategias_vencedoras_quase_melhores.csv`, fator "redução (com k)":
`raw` ×1.50, PCA ×0.96–1.15, UMAP ×0.68–0.80): a estratégia global (`mae_cls` · raw · CatBoost) fica entre
as quase-melhores em 32 de 38 variáveis, e o vencedor individual a supera além do desvio em só 6 —
três delas diagnósticos raros (`DCIS`, `Diagnosis_Fibroadenoma`, `Invasive lobular carcinoma`;
13–30 positivos), onde o máximo de 88 tentativas é o mais otimista. Vantagem média: +0.034
nas variáveis de imagem.

---

## 2026-09-28

### Experimento: separar aquisição de conteúdo (`acquisition_confound`)

Continuação dos próximos passos (a) e (b) do sweep de 2026-09-26; o (c) (embedding só da
lesão, recortada pela máscara) fica para depois. Script: `src/evaluation/acquisition_confound.py`.
Saídas: `outputs/embedding_to_tabular_experiments/acquisition_confound/` (`results_folds.csv`,
`results.csv`, `summary_by_group.csv`, `history_from_annotations.csv`).

**Pergunta.** O sweep mostrou que só as variáveis de aquisição da imagem (largura, altura,
`Pixel_size`) já preveem boa parte dos alvos, e que alvos de histórico eram previstos melhor
que os de imagem. Quanto do desempenho da embedding é **reconhecimento do lote de aquisição**
e quanto é **conteúdo da imagem**?

**Montagem.** `mae_cls` bruto + CatBoost (a melhor configuração do sweep), 38 alvos de
classificação, 5 folds. Três conjuntos de features avaliados nos **mesmos folds** dentro de
cada esquema de CV, mais o Dummy:

| Conjunto | Features |
|---|---|
| aquisição | largura, altura, `Pixel_size` |
| embedding | `mae_cls` (768-d) |
| embedding + aquisição | as duas juntas (771-d) — passo **(a)** |

| Esquema de CV | Como os folds são formados — passo **(b)** |
|---|---|
| `stratified` | `StratifiedKFold(5)`, igual ao sweep (reproduz o sweep exatamente: `Halo` 0.7912 nos dois) |
| `group_size` | `StratifiedGroupKFold(5)`, grupo = tamanho da imagem (52 grupos): nenhum tamanho aparece em treino e teste ao mesmo tempo |
| `group_pixel_size` | idem, grupo = `Pixel_size` (28 grupos) — **mais rigoroso**: um mesmo `Pixel_size` cobre vários tamanhos (ex.: 0.0078125 = 485×537 e 831×538, 82 casos), provavelmente o mesmo aparelho/profundidade |

Os lotes são muito desiguais (o maior tem 66 casos por tamanho, 82 por `Pixel_size`; 27
tamanhos têm um único caso), então os folds agrupados têm tamanhos de 37 a 82. Um fold é
descartado se o treino não tem todas as classes ou o teste tem uma só: **3 de 570** (em
`Symptoms_HRT`, `family history` e `DCIS`, todos em `group_pixel_size`). CatBoost 1 thread,
4 workers sob `MemoryMax=10G`, 57 min.

#### Resultado por grupo (ROC-AUC médio)

| Grupo | CV | Aquisição | Embedding | Embedding + aquisição | Ganho (emb+aq − aq) |
|---|---|---|---|---|---|
| `image` (23) | estratificada | 0.634 | 0.714 | 0.710 | +0.076 |
| | agrupada por tamanho | 0.565 | 0.672 | 0.677 | +0.112 |
| | agrupada por `Pixel_size` | 0.606 | 0.683 | 0.694 | +0.089 |
| `history` (5) | estratificada | 0.718 | 0.825 | 0.829 | +0.111 |
| | agrupada por tamanho | 0.668 | 0.766 | 0.758 | +0.090 |
| | agrupada por `Pixel_size` | 0.653 | 0.775 | 0.779 | +0.126 |
| `pathology` (6) | estratificada | 0.655 | 0.636 | 0.640 | −0.015 |
| | agrupada por tamanho | 0.562 | 0.605 | 0.601 | +0.040 |
| | agrupada por `Pixel_size` | 0.593 | 0.591 | 0.582 | −0.012 |

- **(a) Juntar a aquisição à embedding não acrescenta nada** (imagem: 0.710 vs. 0.714; o
  mesmo em todos os esquemas e grupos). A embedding **já contém** a informação de aquisição.
- **(b) Separar os lotes custa pouco à embedding**: nos alvos de imagem, 0.714 → 0.672–0.683
  (−0.03 a −0.04). Essa é a parcela que vinha de reconhecer o lote. A aquisição sozinha cai
  mais (0.634 → 0.565–0.606), mas não chega a 0.5: os parâmetros de aquisição têm alguma
  relação com os alvos que se generaliza entre lotes. Hipótese plausível: `Pixel_size`
  acompanha a profundidade de varredura, que acompanha o tamanho da mama/lesão — isso seria
  correlação clínica real, não artefato.

#### Resultado por variável — CV agrupada por `Pixel_size` (o esquema mais rigoroso)

Ganho = ROC-AUC embedding − ROC-AUC aquisição, com margem igual ao desvio entre folds da
embedding (aquisição com erro 0). Em negrito, ganhos cujo módulo supera a margem. As colunas
"estratificada" repetem o sweep, para referência.

| Alvo | Aquisição (estrat.) | Embedding (estrat.) | Aquisição (agrup.) | Embedding (agrup.) | Emb + aquisição (agrup.) | Ganho (agrup.) |
|---|---|---|---|---|---|---|
| `Skin_thickening` | 0.444 | 0.790 | 0.533 | 0.779 ± 0.078 | 0.775 | **+0.246 ± 0.078** |
| `Posterior_features` (4 cl.) | 0.602 | 0.798 | 0.554 | 0.775 ± 0.063 | 0.756 | **+0.221 ± 0.063** |
| `Margin_spiculated` | 0.688 | 0.815 | 0.687 | 0.835 ± 0.130 | 0.845 | **+0.148 ± 0.130** |
| `Echogenicity` (6 cl.) | 0.541 | 0.637 | 0.540 | 0.657 ± 0.094 | 0.650 | **+0.117 ± 0.094** |
| `Margin_circumscribed` | 0.682 | 0.765 | 0.633 | 0.740 ± 0.057 | 0.751 | **+0.107 ± 0.057** |
| `Margin_microlobulated` | 0.636 | 0.659 | 0.563 | 0.717 ± 0.220 | 0.757 | +0.153 ± 0.220 |
| `Halo` | 0.748 | 0.791 | 0.689 | 0.783 ± 0.126 | 0.763 | +0.094 ± 0.126 |
| `Mammary duct ectasia` | 0.498 | 0.761 | 0.673 | 0.757 ± 0.175 | 0.747 | +0.083 ± 0.175 |
| `Margin_indistinct` | 0.657 | 0.743 | 0.661 | 0.736 ± 0.094 | 0.706 | +0.076 ± 0.094 |
| `Margin_angular` | 0.647 | 0.700 | 0.571 | 0.630 ± 0.163 | 0.603 | +0.059 ± 0.163 |
| `BIRADS` (6 cl.) | 0.593 | 0.669 | 0.595 | 0.635 ± 0.069 | 0.649 | +0.040 ± 0.069 |
| `Shape` (3 cl.) | 0.674 | 0.686 | 0.622 | 0.611 ± 0.061 | 0.691 | −0.011 ± 0.061 |
| `Intraductal papilloma` | 0.817 | 0.832 | 0.714 | 0.626 ± 0.265 | 0.690 | −0.088 ± 0.265 |
| `Diagnosis_Invasive carcinoma NST` | 0.608 | 0.729 | 0.663 | 0.693 ± 0.090 | 0.711 | +0.030 ± 0.090 |
| `Diagnosis_Invasive lobular carcinoma` | 0.769 | 0.731 | 0.543 | 0.586 ± 0.241 | 0.606 | +0.043 ± 0.241 |
| `Diagnosis_Fibroadenoma` | 0.639 | 0.448 | 0.539 | 0.554 ± 0.134 | 0.476 | +0.015 ± 0.134 |
| `Diagnosis_DCIS` | 0.683 | 0.513 | 0.674 | 0.386 ± 0.151 | 0.333 | **−0.287 ± 0.151** |

- **Sobrevivem ao agrupamento com ganho acima da margem (5)**: `Skin_thickening`,
  `Posterior_features`, `Margin_spiculated`, `Echogenicity`, `Margin_circumscribed` — a
  evidência mais forte, até aqui, de informação visual real na embedding.
  ⚠️ Corrigido em 2026-09-30: contei só as variáveis desta tabela, que é uma seleção. Sobre as
  23 variáveis de imagem são **8**; faltaram `Calcifications` (+0.210 ± 0.134),
  `Tissue_homogeneous: fibroglandular` (+0.121 ± 0.119) e
  `Tissue_heterogeneous: predominantly fat` (+0.060 ± 0.027). Lista completa no relatório
  consolidado (seção 9). `Margin_spiculated`
  e `Echogenicity` estavam dentro da margem na CV estratificada e passam a superá-la agora.
- **Perdem o sinal ao separar lotes**: `Intraductal papilloma` (0.832 → 0.626), `Shape`
  (0.686 → 0.611, abaixo da aquisição), `Invasive lobular carcinoma` (0.731 → 0.586),
  `BIRADS` (ganho +0.076 → +0.040, dentro da margem). Nestes, parte relevante do desempenho
  do sweep era reconhecimento de lote.
- `DCIS` fica **abaixo do acaso** (0.386) — com 14 positivos, provavelmente concentrados em
  poucos lotes; um dos 5 folds foi descartado.
- `Margin_indistinct`, `BIRADS` e `Invasive carcinoma NST` passavam da margem na CV
  estratificada (ver 2026-09-26) e deixam de passar com os lotes separados.

#### ⚠️ Correção: `Signs_*` não são controles negativos

Em 2026-09-26 tratei o grupo `history` como controle negativo ("não visível na imagem") e
concluí que ele "falhou". Teste direto (`src/evaluation/history_from_annotations.py`):
CatBoost treinado **só com as anotações de imagem do especialista** (BIRADS, Shape,
Echogenicity, Posterior_features, Halo, Calcifications, Skin_thickening, Margin_*,
Interpretation_*, Tissue_*; 65 features one-hot), mesmos folds:

| Alvo | Anotações de imagem | Embedding (estrat.) | Embedding (agrup. `Pixel_size`) |
|---|---|---|---|
| `Signs_palpable` | **0.873** ± 0.053 | 0.844 | 0.799 |
| `Signs_no` | **0.854** ± 0.066 | 0.849 | 0.811 |
| `Symptoms_no` | 0.743 ± 0.049 | 0.732 | 0.692 |
| `Symptoms_HRT/hormonal contraception` | 0.738 ± 0.150 | 0.883 | 0.764 |
| `Symptoms_family history` | 0.652 ± 0.144 | 0.818 | **0.806** |

- **`Signs_*` e `Symptoms_no` são previstos pelas anotações tanto quanto pela embedding.**
  Lesão palpável é lesão maior e mais suspeita — algo visível. Logo esses alvos estão ligados a
  características visíveis e **não servem de controle negativo**; a embedding prevê-los não é
  evidência de fator de confusão.
- `Symptoms_HRT` excede as anotações na CV estratificada (0.883 vs. 0.738), mas com os lotes
  separados cai para 0.764, dentro da margem das anotações. O excesso vinha do lote.
- **Só `Symptoms_family history` mantém um excesso inexplicado** (0.806 com os lotes
  separados vs. 0.652 das anotações), com 22 positivos e um fold descartado. É o único resíduo
  que ainda aponta para algo além da lesão e da aquisição.

**Leitura revisada.** O fator de confusão de aquisição existe, mas é **menor** do que o
registrado em 2026-09-26: vale ~0.03–0.04 de ROC-AUC médio nos alvos de imagem, e está
concentrado em alguns alvos (`Intraductal papilloma`, `Shape`, `Invasive lobular carcinoma`).
Com os lotes separados, a embedding do USF-MAE ainda prevê os alvos de imagem com ROC-AUC
médio de 0.68, e cinco variáveis mantêm ganho claro sobre a aquisição (⚠️ são 8 sobre as 23
variáveis de imagem — ver correção de 2026-09-30).

**Próximo passo**: (c) embedding só da lesão, recortada pela máscara (`Mask_tumor_filename`),
avaliada com CV agrupada por `Pixel_size`; investigar o resíduo de `Symptoms_family history`;
escolher controles negativos que de fato não se relacionem com a lesão.

---

## 2026-09-26

### 🧭 Mudança de direção do projeto: da reconstrução para o estudo da informação clínica nas embeddings

Nenhuma rota de reconstrução (tabular→embedding→decoder, UMAP inverso, perturbações —
ver 2026-09-15) produziu imagem com anatomia. O projeto passa a perguntar **quanta
informação sobre as variáveis que o especialista anota olhando a imagem** (BIRADS,
forma, margem, ecogenicidade, diagnóstico...) as embeddings carregam, medida pelo
desempenho de classificadores sobre a embedding. Três comparações:

1. USF-MAE vs. SD-VAE;
2. PCA vs. UMAP (vs. embedding bruta), no mesmo nº de componentes;
3. CatBoost vs. XGBoost vs. LightGBM (+ MLP), com ROC-AUC, PR-AUC, precision, recall,
   accuracy.

Resultados organizados em `outputs/embedding_to_tabular_experiments/{folder_name}/`.
Doc do tema: `docs/embedding-to-clinical-gbm-log.md` (continua
`docs/embedding-to-clinical-log.md`, que segue válido para a rodada anterior).

### Experimento: sweep embedding → clínica com CV (`train_embedding_to_clinical_gbm.py`)

**O que muda em relação à rodada anterior** (`train_embedding_to_clinical.py`):
- Split único 80/20 → **StratifiedKFold(5)** / KFold(5) para Age, partições idênticas
  entre embeddings/redutores/modelos (comparação pareada por fold).
- CatBoost + MLP → **CatBoost, XGBoost, LightGBM, MLP**.
- **PR-AUC** adicionado (`safe_pr_auc`); PCA com nº fixo de componentes (10/30/50),
  igual ao UMAP; UMAP com 3 seeds por fold.
- Nova embedding **`vae_flat`** (latente 4×32×32 achatado, 4096-d). O `vae_mean`
  (4-d) descartava toda a informação espacial, o que tornava a comparação MAE vs. VAE
  da rodada anterior injusta com o VAE.
- Colunas single-label com 2 classes após o filtro (Halo, Calcifications,
  Skin_thickening, Verification) viram alvos **binários**, não multiclasse-macro.
- Alvos-sentinela (`*_not available`, `*_not applicable`) ficam de fora das médias do
  resumo (medem preenchimento do laudo, não a lesão).

**Código.** Novo `src/models/embedding_to_clinical_shared.py` (loaders, PCA/UMAP
ajustados só no treino, `safe_roc_auc`, `safe_pr_auc`), usado pelo script antigo e
pelo novo. Verificado: o script antigo gera CSV **idêntico** (`assert_frame_equal`)
antes e depois do refactor (`--model mae --pooling mean --umap-components 10`).
`embedding_pooling.py` ganhou `load_vae_flat`.

#### 🐛 Incidente: o primeiro smoke test travou a máquina (OOM)

O VSCode travou durante a primeira execução. Causa: importar o script custava
**1,3 GB e 15 s** porque `embedding_pooling → serve.inference` carrega torch/torchvision,
e cada um dos 8 workers do joblib reimportava isso — ~12 GB antes de treinar qualquer
modelo, numa máquina com 15 GB e **sem swap**. Correção: import tardio dos loaders em
`load_embeddings` (só o processo principal carrega embeddings) → 437 MB por worker;
`--n-jobs` padrão 4. Execuções longas passam a rodar com teto rígido de memória:
`systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0 python -m ...`.

Medição de custo (1 fold, 1 thread): o fit sobre a embedding **bruta** domina — CatBoost
em `vae_flat` BIRADS leva 106 s, em `mae_mean` 21 s; depois de PCA/UMAP, < 1,5 s.
Estimativa do sweep completo: ~7 h de CPU.

#### Variação: sweep completo — `gbm_pca_umap_sweep`

Comando: `python -m src.models.train_embedding_to_clinical_gbm --folder-name gbm_pca_umap_sweep`
(2h25, 4 workers, sob `MemoryMax=10G`). Análise estatística:
`python -m src.evaluation.analyze_embedding_to_clinical_sweep --folder-name gbm_pca_umap_sweep`.
Saídas em `outputs/embedding_to_tabular_experiments/gbm_pca_umap_sweep/`; tabelas completas
em `docs/embedding-to-clinical-gbm-log.md`. (Contexto dos testes detalhado em 2026-09-28, a
partir de dúvidas levantadas na revisão dos resultados.)

##### Contexto dos testes

**Grade.** 4 embeddings × 7 redutores × 4 modelos × 39 alvos, cada combinação avaliada em
5 folds:

| Fator | Níveis |
|---|---|
| Embedding | `mae_mean` (média dos 196 patches, 768-d), `mae_cls` (token CLS, 768-d), `vae_mean` (média espacial do latente, 4-d; só `raw`), `vae_flat` (latente 4×32×32 achatado, 4096-d) |
| Redução | `raw`; PCA com 10/30/50 componentes; UMAP com 10/30/50 (3 seeds por fold, média) — ajustados só no fold de treino |
| Modelo | CatBoost, XGBoost, LightGBM (150 árvores, prof. 4, lr 0.08, classes balanceadas), MLP `(64,)`; baseline `DummyClassifier(most_frequent)` |
| Validação | `StratifiedKFold(5)` (`KFold(5)` para Age), `random_state=42`: as **mesmas partições** para todas as embeddings, redutores e modelos de um alvo, logo tudo é comparável pareado por fold |

**Como cada tipo de variável foi modelado** — depende de a coluna ter um ou vários
rótulos por caso:

- **Coluna de rótulo único → um único modelo multiclasse com todas as classes de uma
  vez** (não um modelo por dummy). As classes viram 0..K-1 e cada biblioteca usa seu modo
  multiclasse nativo (softmax), devolvendo uma probabilidade por classe. ROC-AUC e PR-AUC
  são calculados um-contra-o-resto com média macro a partir dessas probabilidades;
  precision/recall com média macro. Classes com < 5 casos são descartadas.

  | Variável | Classes usadas |
  |---|---|
  | `BIRADS` | 6 (2, 3, 4a, 4b, 4c, 5) — a classe `1` (4 casos) foi descartada |
  | `Shape` | 3 (irregular, oval, round) |
  | `Echogenicity` | 6 (hypoechoic, heterogeneous, anechoic, isoechoic, complex cystic/solid, hyperechoic) |
  | `Posterior_features` | 4 (no, shadowing, enhancement, combined) |

  Colunas de rótulo único que ficam com **2 classes** após o filtro viram **binárias**
  (classe positiva = minoritária): `Halo`, `Skin_thickening`, `Verification` e
  `Calcifications` (`no` vs. `in a mass`; `intraductal` e `indefinable` têm 2 casos cada).

- **Coluna multirrótulo → um modelo binário por rótulo (dummy)**, porque um caso pode ter
  vários rótulos ao mesmo tempo (separados por `&`). Entram os rótulos com 5–95% de
  positivos. Casos com mais de um rótulo, por coluna:

  | Coluna | Casos com > 1 rótulo | Exemplo |
  |---|---|---|
  | `Interpretation` | 149 | `Dysplasia&Fibroadenoma` |
  | `Margin` | 67 | `spiculated&angular&indistinct` |
  | `Diagnosis` | 23 | `Invasive carcinoma NST&DCIS` |
  | `Signs` | 13 | `nipple retraction&palpable` |
  | `Tissue_composition` | 6 | `lactating&homogeneous: fibroglandular` |
  | `Symptoms` | 5 | `family history&HRT` |

  ⚠️ Em aberto: `Tissue_composition`, `Symptoms` e `Signs` são quase de rótulo único e
  poderiam ser modeladas como multiclasse (mais comparável a `BIRADS`/`Shape`). Nesta
  rodada seguiram como um binário por rótulo, herdado de `clinical_targets.py`.

**Grupos de alvo** (`target_group` em `train_embedding_to_clinical_gbm.py`) — separam o que
o especialista vê na imagem do que não está nela:

| Grupo | Alvos | Papel |
|---|---|---|
| `image` (23) | `BIRADS`, `Shape`, `Echogenicity`, `Posterior_features`, `Halo`, `Calcifications`, `Skin_thickening`; `Margin_` angular, circumscribed, indistinct, microlobulated, spiculated; `Interpretation_` complex cyst, cyst filled with thick fluid, dysplasia, fibroadenoma, intraductal papilloma, mammary duct ectasia, suspicion of malignancy; `Tissue_` heterogeneous predominantly fat, heterogeneous predominantly fibroglandular, homogeneous fat, homogeneous fibroglandular | **pergunta principal** — anotados pelo especialista a partir da imagem |
| `pathology` (6) | `Verification`; `Diagnosis_` benign mammary dysplasia, DCIS, fibroadenoma, invasive carcinoma NST, invasive lobular carcinoma | diagnóstico confirmado (biópsia/acompanhamento) |
| `history` (5 + Age) | `Signs_no`, `Signs_palpable`, `Symptoms_no`, `Symptoms_HRT/hormonal contraception`, `Symptoms_family history`; `Age` (regressão, fora das médias de ROC-AUC) | **controle negativo** — não visíveis na imagem; se forem bem previstos, há fator de confusão |
| `sentinel` (4) | `Signs_not available`, `Symptoms_not available`, `Tissue_not available`, `Diagnosis_not applicable` | fora dos resumos — medem se o laudo foi preenchido, não a lesão |

Decisões de fronteira: `Interpretation` entrou em `image` por ser a leitura do
radiologista sobre a imagem; `Tissue_composition` também, porque a composição do tecido de
fundo aparece no ultrassom.

**Por que as dimensões da imagem aparecem na análise.** Elas **não são features** dos
modelos sobre as embeddings — todas as comparações de embedding, redutor e modelo usam só
a embedding. Entram apenas num **controle separado**
(`src/evaluation/metadata_only_baseline.py`): CatBoost treinado com os mesmos folds sobre
três conjuntos de variáveis da imagem — só tamanho (largura, altura), só `Pixel_size`, e os
três juntos ("aquisição"). O motivo:
- o dataset tem imagens de **52 tamanhos** diferentes e `Pixel_size` de 0.0043 a 0.0139, o
  que reflete o aparelho e a forma de exportação, não a lesão;
- antes de virar embedding, toda imagem é redimensionada para 224×224 (MAE) ou 256×256
  (VAE) **sem preservar a proporção**, então o aparelho fica "gravado" na embedding —
  pela distorção, pela escala do pixel, pela textura do equipamento e por bordas e texto
  sobre a imagem;
- o controle responde *quanto dá para acertar sem olhar a lesão*. Se três números de
  aquisição já preveem um alvo, a embedding prever esse alvo não prova conteúdo clínico.

##### Resultados — grupo `image` (23 alvos), Wilcoxon pareado por alvo

~80 testes no total: tratar 0.01 < p < 0.05 como sugestivo, não conclusivo. Quando o fator
comparado não é o modelo, os 4 modelos são promediados por alvo antes de parear.

| Pergunta | Resultado |
|---|---|
| MAE vs. VAE | **MAE**: ROC-AUC 0.70 vs. 0.59 (VAE no melhor redutor, PCA 10), melhor em 22–23 de 23 alvos (p < 0.001); PR-AUC 0.49 vs. 0.39. `mae_mean` ≈ `mae_cls` (p = 0.39). `vae_flat` bruto ≈ `vae_mean` (0.562 vs. 0.568): achatar o latente não recupera sinal. |
| PCA vs. UMAP | No MAE **nenhum ajuda**: bruto 0.700 > PCA 0.635–0.660 > UMAP 0.618–0.623 (bruto vence em 20–23 de 23, p < 0.001). Com o mesmo k, **PCA > UMAP** nas 9 comparações (p entre 0.002 e 0.08, sempre na mesma direção). No VAE, PCA 10 ajuda um pouco (+0.03, p = 0.005). |
| Modelos | CatBoost 0.714 ≳ XGBoost 0.707 ≈ LightGBM 0.703 (diferenças não significativas, p 0.05–0.23) > MLP 0.674 (p ≤ 0.033). No VAE, empate (Friedman p = 0.51). LightGBM ~8× mais rápido que o CatBoost. |

ROC-AUC por redutor (média dos 4 modelos):

| | bruto | PCA 10 | PCA 30 | PCA 50 | UMAP 10 | UMAP 30 | UMAP 50 |
|---|---|---|---|---|---|---|---|
| `mae_cls` | **0.700** | 0.635 | 0.660 | 0.643 | 0.618 | 0.623 | 0.621 |
| `mae_mean` | **0.693** | 0.633 | 0.652 | 0.646 | 0.613 | 0.607 | 0.617 |
| `vae_flat` | 0.562 | **0.592** | 0.588 | 0.583 | 0.568 | 0.566 | 0.565 |

Variáveis do escopo: embedding na melhor configuração (`mae_cls` bruto + CatBoost, média ±
desvio entre folds) ao lado de dois baselines com os mesmos folds e o mesmo CatBoost:
- **Dummy** — chuta a classe majoritária: ROC-AUC = 0.5 sempre, PR-AUC = prevalência da
  classe positiva (1/K no multiclasse macro);
- **Aquisição** — só as variáveis da imagem: largura, altura e `Pixel_size`.

**Ganho ROC-AUC** = ROC-AUC embedding − ROC-AUC aquisição. A margem é o desvio entre folds da
embedding (a aquisição é tratada como erro 0). Em negrito, ganhos cujo módulo supera a margem.

| Alvo | ROC-AUC embedding | ROC-AUC aquisição | Ganho ROC-AUC | PR-AUC embedding | PR-AUC Dummy | PR-AUC aquisição | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| `Margin_spiculated` | 0.815 ± 0.164 | 0.688 | +0.127 ± 0.164 | 0.504 | 0.129 | 0.231 | 0.500 | 0.305 |
| `Posterior_features` (4 cl.) | 0.798 ± 0.069 | 0.602 | **+0.196 ± 0.069** | 0.560 | 0.250 | 0.339 | 0.489 | 0.457 |
| `Halo` | 0.791 ± 0.046 | 0.748 | +0.043 ± 0.046 | 0.625 | 0.218 | 0.451 | 0.643 | 0.527 |
| `Margin_circumscribed` | 0.765 ± 0.043 | 0.682 | **+0.084 ± 0.043** | 0.717 | 0.449 | 0.647 | 0.660 | 0.661 |
| `Margin_indistinct` | 0.743 ± 0.041 | 0.657 | **+0.085 ± 0.041** | 0.732 | 0.449 | 0.644 | 0.662 | 0.652 |
| `Margin_angular` | 0.700 ± 0.072 | 0.647 | +0.053 ± 0.072 | 0.419 | 0.164 | 0.255 | 0.435 | 0.281 |
| `Shape` (3 cl.) | 0.686 ± 0.054 | 0.674 | +0.013 ± 0.054 | 0.546 | 0.333 | 0.505 | 0.405 | 0.410 |
| `BIRADS` (6 cl.) | 0.669 ± 0.053 | 0.593 | **+0.076 ± 0.053** | 0.375 | 0.167 | 0.264 | 0.314 | 0.323 |
| `Margin_microlobulated` | 0.659 ± 0.107 | 0.636 | +0.023 ± 0.107 | 0.343 | 0.141 | 0.222 | 0.227 | 0.132 |
| `Echogenicity` (6 cl.) | 0.637 ± 0.110 | 0.541 | +0.096 ± 0.110 | 0.370 | 0.167 | 0.243 | 0.282 | 0.255 |
| `Diagnosis_Invasive carcinoma NST` | 0.729 ± 0.106 | 0.608 | **+0.121 ± 0.106** | 0.521 | 0.258 | 0.352 | 0.628 | 0.409 |
| `Diagnosis_Invasive lobular carcinoma` | 0.731 ± 0.149 | 0.769 | −0.037 ± 0.149 | 0.167 | 0.051 | 0.186 | 0.000 | 0.000 |
| `Diagnosis_DCIS` | 0.513 ± 0.089 | 0.683 | **−0.170 ± 0.089** | 0.097 | 0.055 | 0.134 | 0.000 | 0.000 |
| `Diagnosis_Fibroadenoma` | 0.448 ± 0.080 | 0.639 | **−0.191 ± 0.080** | 0.133 | 0.117 | 0.212 | 0.000 | 0.000 |

Ganho acima da margem em 5 variáveis: `Posterior_features`, `Margin_circumscribed`,
`Margin_indistinct`, `BIRADS` e `Invasive carcinoma NST`. Em `DCIS` e `Fibroadenoma`, a
aquisição prevê **melhor** que a embedding, além da margem. Nas demais, o ganho fica dentro da
margem. Tabela completa, com os 38 alvos de classificação: `metadata_only_baseline.csv`.

Leitura das métricas: diagnósticos raros (13–30 positivos) têm recall 0 no limiar padrão
(0.5) — o modelo ordena, mas nunca prevê a classe; para eles, ROC-AUC e PR-AUC são as
métricas informativas. `accuracy` engana aqui: o baseline que chuta a classe majoritária já
tem 0.759 em média. `Age`: R² = 0.126 ± 0.187 (`mae_mean` bruto + CatBoost).

##### ⚠️ Fator de confusão de aquisição — muda a leitura dos valores absolutos

Média por grupo, com os mesmos baselines da tabela acima (`mae_cls` bruto + CatBoost):

| Grupo | ROC Dummy | ROC tamanho | ROC pixel | ROC aquisição | ROC embedding | Ganho sobre aquisição |
|---|---|---|---|---|---|---|
| `history` (não visível) | 0.500 | 0.719 | 0.712 | 0.718 | **0.825** | +0.107 |
| `image` | 0.500 | 0.635 | 0.625 | 0.634 | **0.714** | +0.080 |
| `pathology` | 0.500 | **0.657** | 0.651 | 0.655 | 0.636 | −0.019 |

| Grupo | PR Dummy (prevalência) | PR tamanho | PR pixel | PR aquisição | PR embedding |
|---|---|---|---|---|---|
| `history` | 0.279 | 0.426 | 0.414 | 0.432 | **0.620** |
| `image` (sem `Calcifications`) | 0.232 | 0.378 | 0.367 | 0.381 | **0.486** |
| `pathology` | 0.134 | 0.245 | 0.237 | 0.253 | **0.294** |

`Calcifications` fica fora da média de PR-AUC: o valor do Dummy e da embedding vem do
sweep, que tem a classe positiva invertida, e o dos baselines de aquisição já usa a
classe corrigida. O ROC-AUC não é afetado pela inversão.

- **Tamanho sozinho, `Pixel_size` sozinho e os três juntos dão praticamente o mesmo
  resultado** (imagem: 0.635 / 0.625 / 0.634). Cada variável isolada já identifica o lote
  de aquisição; juntá-las não acrescenta informação.
- **O controle negativo falhou**: alvos de histórico (`Symptoms_HRT` 0.883, `Signs_palpable`
  0.844, `family history` 0.818) são previstos *melhor* que os de imagem.
  ⚠️ Corrigido em 2026-09-28: `Signs_*` são previstos igualmente bem só pelas anotações de
  imagem do especialista, então não são controles negativos; `Symptoms_HRT` perde o excesso
  quando os lotes são separados. Só `family history` mantém um resíduo inexplicado.
- **Três números de aquisição já preveem muito**: os 28 casos com imagem 433×469 px têm
  **todos** `Signs = not available` — o tamanho identifica um lote de coleta, e cada lote
  tem seu padrão de preenchimento e sua população.
- O ganho da embedding sobre os metadados é **maior** no histórico que na imagem, então nem
  o ganho residual pode ser atribuído à lesão. **O desenho mede o que a embedding permite
  prever, não a informação clínica visual que ela carrega.** As comparações relativas
  (MAE vs. VAE, PCA vs. UMAP, modelos) seguem válidas; os valores absolutos são um teto
  contaminado.
  ⚠️ Revisto em 2026-09-28: com a CV agrupada por lote, a embedding perde só ~0.03–0.04 nos
  alvos de imagem (0.714 → 0.68), e oito variáveis mantêm ganho claro sobre a aquisição. O
  argumento se apoiava no controle negativo, que não era válido.
- Candidatos mais fortes a sinal real (ganho sobre a aquisição acima do desvio da
  embedding): `Skin_thickening` (+0.346 ± 0.179), `Mammary duct ectasia` (+0.264 ± 0.087),
  `Posterior_features` (+0.196 ± 0.069), `Calcifications` (+0.173 ± 0.153). `Margin_spiculated`
  tem ganho alto (+0.127), mas dentro da margem (± 0.164). Explicáveis por aquisição:
  `Shape` (+0.013), `Intraductal papilloma` (+0.015), `Interpretation_Fibroadenoma` (−0.037).

⚠️ **Corrige** `docs/embedding-to-clinical-log.md`: lá o UMAP parecia "menos destrutivo"
que o PCA, mas o PCA tinha poucos componentes (fração de variância); com o mesmo k, o
PCA preserva mais.

🐛 `Calcifications` saiu com a classe positiva invertida (`no` = 1): PR-AUC/precision/recall
desse alvo nesta rodada se referem a "sem calcificação". Corrigido no código (positiva =
minoritária) para as próximas rodadas; ROC-AUC e as comparações não são afetados.

**Ressalvas**: `best_by_target.csv` escolhe o máximo de 91 combinações por alvo (⚠️ são 88:
3 embeddings × 7 reduções × 4 modelos + `vae_mean` só raw × 4 — corrigido em 2026-09-30) (otimista,
ex.: `Invasive lobular carcinoma` 0.903 com 13 positivos); 5 folds sobre 256 imagens dão
desvio de 0.04–0.18 de ROC-AUC por alvo; precision/recall sem ajuste de limiar.

**Próximo passo**: separar aquisição de conteúdo — (a) metadados + embedding vs. só
metadados; (b) CV agrupada por lote de aquisição; (c) embedding só da lesão, recortada
pela máscara (`Mask_tumor_filename`), verificando se o controle `history` cai para ~0.5;
(d) decidir se `Tissue_composition`/`Symptoms`/`Signs` viram multiclasse.

---

## 2026-09-15

### 🐛 Correção: MAE embaralha patch tokens por imagem (`ids_restore`)

**O que é.** O `random_masking` do MAE embaralha a sequência de patches via
`argsort(rand)` **mesmo com `mask_ratio=0.0`**, e a permutação é sorteada **por
imagem**. Logo o índice de um token em `embeddings_per_patch.json` corresponde a um
patch espacial diferente em cada imagem. O `ids_restore` salvo no JSON desfaz isso.

**Como apareceu.** Ao gerar reconstruções do MAE, o decode da embedding verdadeira
saiu como mosaico sem anatomia, e eu concluí que o decoder do MAE não reconstruía.
O usuário refutou apontando `outputs/perturbation_v0/case001_original_vs_reconstruction.png`,
que mostra reconstrução fiel. A diferença era que `serve/perturbation_test.py` passa
`ids_restore` e meu script não.

**Verificação.** Decodificar a mesma embedding com e sem `ids_restore` produz mosaico
vs. reconstrução fiel. Decodificar tokens já em ordem espacial com `ids_restore=None`
é **byte-idêntico** a decodificar os embaralhados com o `ids_restore` real.

**Correção aplicada.**
- Novo `src/preprocessing/mae_tokens.py::restore_spatial_order()` — módulo sem
  dependências, para evitar o ciclo de import
  (`build_tabular_dataset` → `embedding_pooling` → `build_tabular_dataset_vae` → `build_tabular_dataset`).
- `src/preprocessing/embedding_pooling.py::load_mae_patch_tensor()` — passa a reordenar.
- `src/preprocessing/build_tabular_dataset.py` — reordena antes de atribuir `x_pos`/`y_pos`.
- `src/models/umap_inferred_reconstruction_mae.py` — reordena antes de achatar para o UMAP.
- `dataset_breast/processed_dataset_embeddings.csv` regenerado: **98.9% das células de
  embedding mudaram** (shape idêntico, `x_pos`/`y_pos` idênticos por construção).

**Alcance.**

| Afetado (índice de token tratado como posição) | Imune |
|---|---|
| `build_tabular_dataset.py::token_positions` → `x_pos`/`y_pos` | Pooling por **média** (invariante a permutação, verificado) |
| `load_mae_patch_tensor` → PCA por posição de patch | Pooling por **CLS** (anexado *depois* do mascaramento, sempre índice 0) |
| Experimento original tabular→embedding por-patch | Todo o VAE (não usa MAE) |

**Impacto medido no experimento original** (`src/models/train_catboost.py`). Mesmo
código, mesmo split por paciente, mesmos hiperparâmetros — só muda a ordenação dos
tokens:

| Dataset | MAE | RMSE | **R²** |
|---|---|---|---|
| Antigo (tokens embaralhados) | 0.133 | 0.168 | **0.015** |
| Corrigido (ordem espacial) | 0.113 | 0.145 | **0.185** |

⚠️ **Isto invalida o diagnóstico de causa raiz registrado em `ONBOARDING.md`** — "a
posição de um patch não carrega o mesmo conteúdo visual entre pacientes diferentes". A
afirmação foi formulada sobre dados em que as posições *literalmente* eram patches
diferentes em cada imagem. O regressor de fato não conseguia aprender, mas não pela
razão registrada.

O que isso **não** significa: R²=0.185 continua longe de viabilizar reconstrução, e não
contradiz as medições de UMAP desta mesma data (todas com embeddings *pooled*, imunes
ao bug). Muda a explicação de *por que* a direção tabular→embedding falhava, e mostra
que ela falhava menos do que se acreditava.

#### Variação: reconstrução visual com as predições corrigidas

`serve/reconstruct_and_compile.py --cases case007.png case080.png case207.png case229.png`
(flag `--cases` adicionada; os 4 são do split de **teste** — `train_catboost.py` escreve
predições para todos os 256 pacientes, inclusive os 204 de treino, então escolher casos
arbitrários inflaria o resultado). Saída: `outputs/reconstructions/compiled/`.

- Painel `decoder` (embedding **verdadeira**): **fiel**. No `case080` a lesão hipoecoica
  aparece nítida com sombra acústica posterior; no `case229` as camadas fasciais estão
  bem definidas. Confirma visualmente que o decoder do MAE sempre funcionou.
- Painel `infered` (embedding **predita do tabular**): gradiente cinza sem estrutura.
  Nenhuma lesão, nenhuma anatomia — colapso para a média, igual a antes da correção.

⚠️ **R² subiu 12× e a reconstrução não melhorou em nada.** O ganho de 0.015 → 0.185 vem
provavelmente de estrutura global de baixa frequência (gradiente de brilho, geometria da
sonda), que só passou a ficar consistentemente posicionada após a correção — nada disso
é conteúdo diagnóstico. Lição metodológica: **R² agregado sobre 768 dimensões não mede
capacidade de reconstrução**.

Isso divide a conclusão antiga em duas partes com destinos diferentes:
- **Se mantém**: prever a embedding a partir do tabular não serve para reconstrução
  (agora com teto íntegro para comparar).
- **Estava errada**: a explicação. Não é "a posição do patch não carrega o mesmo conteúdo
  entre pacientes" — isso descrevia o bug. O motivo real é que o tabular só determina
  estatísticas globais da imagem, não o conteúdo local da lesão.

Reconcilia os experimentos desta data: tabular→pooled (R²≈0.08) e tabular→por-patch
(R²=0.185) medem a mesma coisa modesta, e nenhum sobrevive ao teste visual.

**`ONBOARDING.md` revisado** com essa distinção (itens 1, 2 e 4 das Conclusões, mais a
ressalva em `compare_token_variance.py`).

**Ainda pendente por causa desta correção:**
- Reexecutar o experimento de **PCA por posição de patch** (`docs/embedding-to-clinical-log.md`),
  que usava `load_mae_patch_tensor` e portanto agrupava regiões diferentes na mesma "posição".
- Reexecutar `src/models/train_catboost_per_position.py` e os scripts de
  `src/evaluation/` que leem `processed_dataset_embeddings.csv` — em especial
  `compare_token_variance.py`, cuja medição de "~0.3% da variância explicada pela
  posição" sustentava a antiga causa raiz e está invalidada.

---

### Experimento: reconstrução via UMAP → UMAP inverso

Pergunta: dá para voltar do espaço UMAP para a embedding, e o que acontece com as
imagens? Doc: `docs/tabular-to-umap-log.md`.

Resposta curta: `umap-learn` tem `inverse_transform`, mas ele não viabiliza
reconstrução — e o gargalo não é o regressor, é a própria projeção.

#### Variação: round-trip puro no latente do VAE (sem regressão)

`src/models/umap_roundtrip_reconstruction.py` — latente `(1,4,32,32)` = 4096 dims,
decodificável pelo SD-VAE.

| Componentes | R² no espaço latente | SSIM vs. reconstrução do VAE |
|---|---|---|
| 2 | −0.764 | 0.139 |
| 5 | −0.751 | 0.153 |

R² negativo = pior que chutar o latente médio do treino. Visualmente: sobra só a
silhueta grosseira (envelope da sonda), some toda a textura interna, e **aparece cor**
numa modalidade grayscale — sinal de que o latente invertido cai fora da variedade de
latentes reais.

#### Variação: limite estrutural do `inverse_transform`

| Componentes | Resultado |
|---|---|
| 2, 5 | roda |
| 10 | morre por falta de memória (Delaunay em 10-D) |
| 50 | `QhullError: negative request size (-75562456). Did int overflow due to high-D?` |

O próprio `umap-learn` avisa disso e recomenda **autoencoder** para quem precisa de
espaço latente de dimensão alta *com* inversa.

#### Variação: cadeia completa inferida (tabular → imagem), VAE

`src/models/umap_inferred_reconstruction.py` — strips de 4 painéis em
`outputs/umap_inferred_reconstruction/`.

| Componentes | SSIM roundtrip | SSIM inferido |
|---|---|---|
| 2 | 0.139 | 0.119 |
| 5 | 0.153 | 0.119 |

Acrescentar dimensões melhora o roundtrip e **não** melhora o inferido — o limite não
está no regressor.

#### Variação: cadeia completa inferida, MAE (após correção do `ids_restore`)

`src/models/umap_inferred_reconstruction_mae.py` — tokens `(1,197,768)` = 151.296 dims.
Strips em `outputs/umap_inferred_reconstruction_mae/`.

| Componentes | SSIM roundtrip | SSIM inferido |
|---|---|---|
| 2 | 0.349 | 0.239 |
| 5 | 0.286 | 0.249 |

O MAE preserva bem mais que o VAE no round-trip (mantém estrutura interna, não só a
silhueta) — provavelmente porque seus tokens são representação semântica de alto nível,
mais fácil de interpolar, enquanto o latente do VAE é espacial e de baixo nível. Mas o
painel inferido volta a ser textura genérica: perde lesão e margens.

⚠️ A primeira execução deste experimento (antes da correção) deu SSIM ~0.23 com o teto
também degradado, e eu concluí erradamente que o decoder do MAE não reconstruía.

---

### 🐛 Descoberta: `umap.inverse_transform` é dependente do lote

Encontrado ao investigar uma contradição na busca gulosa de flips: o mesmo vetor de
features dava `dist=1.1076` quando avaliado em lote e `dist=0.0000` avaliado sozinho.
E o controle random-k superava o guloso, o que é impossível por construção.

**O fato.** A mesma coordenada UMAP, invertida sozinha ou dentro de um lote, produz
vetores com **diferença relativa > 1.0** — completamente diferentes, não arredondamento.

| Tamanho do lote | Diferença relativa vs. inversão solo |
|---|---|
| 1, 2, 5, 10, 20 | ~1e-4 (desprezível) |
| **52, 60** | **1.21** |

Determinístico para um lote fixo (duas chamadas idênticas dão diferença 0.0).

**Mecanismo** (`umap_.py::inverse_transform`, linha ~136): o cronograma de otimização vem
de `make_epochs_per_sample(graph.data, n_epochs)`, normalizado pelo **peso máximo de todo
o lote**, e `optimize_layout_inverse` consome um único `rng_state` sequencialmente entre
os pontos. Quantos epochs e quais amostras negativas cada ponto recebe dependem de quais
outros pontos foram invertidos junto.

**Consequência conceitual — e isso fortalece a conclusão anterior.** Eu vinha dizendo que
a inversa do UMAP não serve para reconstrução porque perde informação. O motivo real é
mais grave: ela **não é função bem definida de um ponto**. Reconstruir a imagem de um
paciente daria resultado diferente conforme quais outros pacientes estivessem no mesmo
lote. Isso descarta o caminho sozinho, independentemente de qualquer métrica.

**Alcance:**

| Resultado | Situação |
|---|---|
| Busca gulosa de flips (1ª versão) | **Inválido** — busca em lote, verificação solo |
| Controle "entre pacientes" do experimento de perturbação (SSIM 0.334) | **Inválido como comparação** — lote de 12 vs. perturbações solo |
| Comparações entre perturbações (flips e gaussiano entre si) | **Válidas** — todas solo |
| `train_tabular_to_umap_inverse.py` (teto R²=0.080) | **Suspeito** — lote de 52 nos dois lados; internamente consistente, valor absoluto arbitrário |
| `umap_roundtrip_reconstruction.py` (R² latente −0.75) | Mesma ressalva |
| `umap_inferred_reconstruction{,_mae}.py` (strips de 4 painéis) | Mesma ressalva — lote de 52 |
| Tudo que **não** usa `inverse_transform` (R² no espaço UMAP, embedding→clínica, correção do `ids_restore`) | **Intacto** |

**Protocolo obrigatório daqui em diante**: inverter **um ponto por vez**. Custo: ~10s por
inversão (Delaunay em 5-D sobre 204 pontos de treino), o que torna busca exaustiva com
inversões inviável — por isso `umap_flip_search.py` pontua a busca no espaço de
coordenadas UMAP (barato e batch-independente, pois o CatBoost prediz linha a linha) e
inverte solo apenas a sequência escolhida e os controles, imprimindo a correlação entre
os dois espaços para auditar o proxy.

---

### Experimento: sensibilidade da reconstrução inferida a perturbações no tabular

`src/models/umap_perturbation_reconstruction.py --case case080.png --n-components 5`
Saída: `outputs/umap_perturbation_reconstruction/`.

Pergunta: perturbando as features clínicas de um paciente, como muda a imagem que a
cadeia `tabular → CatBoost → UMAP → inversa → decoder MAE` produz?

**Controle obrigatório**: SSIM entre a predição deste paciente e a de 12 outros pacientes
reais do teste = **0.334**. Sem essa referência, "as imagens ficaram parecidas" não
conclui nada — pode significar apenas que a perturbação foi pequena demais.

#### Variação: ruído gaussiano (fração do desvio-padrão de cada feature)

| Ruído | SSIM vs. base |
|---|---|
| 0.05× – 0.5× sd | **1.0000 (idêntico)** |
| 1.0× sd | 0.883 ± 0.109 |
| 2.0× sd | 0.713 ± 0.282 |

⚠️ O SSIM **exatamente 1.0** (não 0.999) revela o mecanismo: CatBoost é ensemble de
árvores, logo **constante por partes**. Se a perturbação não cruza nenhum limiar de
split, a predição é bit a bit idêntica. Isso é propriedade da classe de modelo, não
ausência de sinal — **"perturbação pequena" não é um teste informativo neste pipeline**.

#### Variação: flip de uma feature clínica one-hot por vez

| Feature invertida | SSIM vs. base |
|---|---|
| `Margin_spiculated` | **0.709** |
| `Margin_microlobulated` | 0.730 |
| `BIRADS_4c` | 0.737 |
| `Diagnosis_Usual ductal hyperplasia` | 0.748 |
| `Shape_oval` | 0.951 |

**Achado positivo**: as features mais influentes são margem espiculada, margem
microlobulada e BI-RADS 4c — exatamente os descritores associados a malignidade. O
modelo reage mais forte às features com maior carga diagnóstica, o que é sinal real e
não ruído.

**Mas**: a perturbação mais agressiva chega a 0.709, longe do controle de 0.334. Mexer
nas features move a imagem numa fração do que separa dois pacientes reais.

**Ressalva sobre o controle**: trocar de paciente muda todas as features de uma vez,
enquanto um flip muda uma só — a comparação não é perfeitamente justa. O ruído a 2× sd
muda todas simultaneamente e chega a 0.713, mas ruído gaussiano sobre colunas one-hot
gera linhas fora da variedade de dados reais (nenhum paciente tem `Shape_oval = 0.37`), e
árvores respondem mal a isso. Ler 0.334 como ordem de grandeza, não como limite preciso.

Visualmente, todas as variantes seguem textura sem anatomia — muda o padrão global de
claro/escuro, não a presença de uma lesão. Consistente com as reconstruções acima.

---

### Experimento: quantos flips maximizam a mudança na reconstrução?

`src/models/umap_flip_search.py` — busca gulosa: partindo da linha real do paciente,
adiciona repetidamente a mudança que mais desloca a predição. Saída:
`outputs/umap_flip_search/`. Caso: `case080.png` (split de teste).

Protocolo: toda inversão é feita **um ponto por vez** (ver a descoberta de dependência
de lote acima). A busca pontua no espaço de coordenadas UMAP (barato, batch-independente);
só a sequência escolhida e os controles são invertidos solo e decodificados. Auditoria do
proxy: correlação z-dist vs. token-dist = **0.88** (livre) e **0.99** (coerente).

#### Controle: distribuição de distâncias entre pacientes reais

| p25 | p50 | p75 | p90 | **p100 (máx)** | média |
|---|---|---|---|---|---|
| 0.274 | 0.445 | 0.611 | 0.645 | **0.777** | 0.422 |

⚠️ p0 = **0.0000**: existe outro paciente cuja predição é **idêntica** à do case080 —
mesma assinatura de constância por partes do CatBoost vista no ruído gaussiano.

#### Variação: flips livres de colunas one-hot

| k | dist. token | random-k | SSIM | mad/255 |
|---|---|---|---|---|
| 1 | 0.297 | 0.098 | 0.865 | 10.9 |
| 2 | 0.511 | 0.115 | 0.657 | 18.9 |
| 3 | 0.651 | 0.123 | 0.504 | 25.3 |
| **4** | **0.835** | 0.220 | **0.175** | **33.3** |
| 5–8 | 0.72–0.79 | 0.11–0.21 | 0.18–0.27 | 29–35 |

**Satura em 4 flips.** Além disso só flutua.

#### Variação: grupos categóricos coerentes (cada linha é um paciente possível)

Satura em **6 grupos**, em `dist = 0.824`. Os grupos 7 e 8 (`Halo`, `Calcifications`) não
mudam nada — o nível que o guloso escolheria já era o ativo. Sequência escolhida:
`Apocrine carcinoma` → `shadowing` → `irregular` → `BIRADS 4b` → `margem angular` →
`heterogênea` (um perfil maligno coerente).

#### Conclusões

**1. Satura cedo**: 4 flips livres ou 6 grupos coerentes. Não é efeito cumulativo.

**2. Há estrutura real**: o guloso bate o random-k por fator de 4 a 7× (0.824 vs.
0.12–0.21). **Qual** descritor se muda importa muito mais que **quantos** — este era o
controle que decidia se havia estrutura ou só "mais perturbação, mais mudança".

**3. ⚠️ Minha previsão estava errada.** Eu esperava que o modo coerente ficasse bem abaixo
do livre, atribuindo o resultado do modo livre a extrapolação com pacientes impossíveis.
Os dois deram praticamente o mesmo valor (0.824 vs. 0.835) — a explicação por
extrapolação **está descartada**.

**4. ⚠️ Correção de enquadramento**: reportei inicialmente "208% do controle" comparando o
máximo do guloso contra a **média** entre pacientes. Contra o **máximo** real (0.777) o
valor é **106%**. A leitura correta: seis descritores clínicos coerentes deslocam a
reconstrução aproximadamente tanto quanto a distância entre os dois pacientes mais
diferentes do conjunto, ultrapassando-a em ~6%.

**5. O resultado é quase tautológico.** O modelo prevê `z` exclusivamente a partir das
features clínicas, logo toda a variação de saída entre pacientes é, por construção,
causada por variação de features. "Mudar features move tanto quanto trocar de paciente"
não é descoberta; a única questão real era se uma combinação escolhida a dedo
ultrapassaria a faixa observada — e ultrapassa por pouco, com um perfil extremo que
nenhum paciente real possui.

**Ressalva que permanece**: nenhuma variante mostra anatomia. Muda o padrão global de
textura, não aparece lesão. Consistente com todo o resto desta data.

---

### Experimento: tabular → embedding projetada no manifold (UMAP)

Doc: `docs/tabular-to-umap-log.md`. Script: `src/models/train_tabular_to_umap.py`.
Embeddings *pooled* — **imunes ao bug do `ids_restore`**.

Ideia: em vez de prever a embedding completa (que falhou historicamente com R²≈0),
prever sua projeção UMAP de baixa dimensão, com UMAP ajustado só no treino.

#### Variação: R² medido no espaço UMAP (mean-pooling)

| Fonte | Alvo | R² médio | desvio (3 seeds) |
|---|---|---|---|
| embedding *(teto)* | umap5 | 0.955 | 0.005 |
| **tabular** | **umap5** | **0.293** | 0.062 |
| tabular | umap30 | 0.264 | 0.029 |
| tabular *(controle)* | raw 768 | 0.082 | — |

Estável de 5 a 30 componentes, então não é artefato de dimensionalidade baixa *dentro
do espaço UMAP*.

#### Variação: R² medido no espaço original de 768 dims (via `inverse_transform`)

`src/models/train_tabular_to_umap_inverse.py`. Esta é a medição que torna as rotas
comparáveis — e que **inverteu a conclusão**.

| Rota | R² (768 dims) |
|---|---|
| tabular → 768d direto | +0.082 |
| **teto**: embedding verdadeira → UMAP → inversa | **+0.080** |
| tabular → UMAP → inversa | −0.158 |
| baseline `DummyRegressor` | −0.040 |

⚠️ **Correção**: o R²=0.293 é real como "prever posição no manifold", mas **não se
traduz** em prever a embedding. Medido no mesmo espaço, passar pelo UMAP é pior que ir
direto e pior que o baseline. O checkpoint que eu havia sugerido não se sustenta.

**O que sobrevive**: sair do espaço por-patch para o pooled vale R²≈0 → 0.082 sozinho.

**Ressalva em aberto**: o vetor tabular inclui `BIRADS`, `Shape`, `Echogenicity`,
`Margin`, `Diagnosis` — o laudo do radiologista, derivado de olhar a mesma imagem.
Separar features de laudo vs. de paciente é o próximo passo natural.

---

## Anterior a 2026-09-15

Experimentos anteriores não foram registrados neste formato cronológico. Seus
resultados estão em:

- `ONBOARDING.md` — MAE, direção tabular→embedding por-patch (R²≈0). ⚠️ Diagnóstico de
  causa raiz sob reverificação após a correção do `ids_restore`.
- `docs/vae-experiment-log.md` — SD-VAE, fases 1–3 (R²=0.054 na regressão tabular).
- `docs/embedding-to-clinical-log.md` — direção inversa (embedding→features clínicas),
  que **funciona**; inclui PCA, PCA por posição de patch e UMAP como redutores de
  entrada. Resultados com pooling mean/CLS são imunes ao bug do `ids_restore`; o de
  **PCA por posição de patch** não é, e precisa ser reexecutado.
