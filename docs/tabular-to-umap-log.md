# Log de experimentos — prever a embedding **projetada no manifold (UMAP)** a partir dos dados tabulares

Este log cobre a direção **direta** (tabular clínico → embedding da imagem), que é
a direção original do projeto e que havia **falhado** nas duas tentativas anteriores:

- `ONBOARDING.md`: MAE por-patch, **R² ≈ 0** em XGBoost, LightGBM, CatBoost e TabPFN.
- `docs/vae-experiment-log.md`: SD-VAE por-posição, **R² = 0.054**, reconstrução final
  um borrão sem estrutura.

A causa raiz diagnosticada nas duas foi a mesma: **a posição de um patch (ou de um
pixel latente) não carrega o mesmo conteúdo visual entre pacientes diferentes**, então
o regressor não tem alvo consistente para aprender e colapsa para a média.

(O log da direção **inversa** — embedding → features clínicas, que funciona — está em
`docs/embedding-to-clinical-log.md`.)

## A ideia testada

Em vez de prever a embedding completa, prever a embedding **projetada num manifold
UMAP de baixa dimensão**, com o UMAP ajustado apenas no conjunto de treino.

Reconstruir a imagem a partir dessa projeção **não é possível**, e isso foi aceito desde
o início. O valor do experimento é outro: se a regressão funcionar, significa que o dado
clínico tabular determina **em que região do manifold de imagens** a lesão se encontra,
mesmo sem determinar a embedding exata — o que seria um resultado interessante por si só
e um possível ponto de partida.

> **Leia a seção "⚠️ Correção" no fim deste documento antes de usar os números das
> tabelas.** Medidas em espaço UMAP e em espaço de embedding não são comparáveis
> diretamente; quando tudo é medido no mesmo espaço, a conclusão muda.

## Duas decisões de desenho

### 1. Nível de imagem, não por patch

O setup por-patch é exatamente o que foi diagnosticado como estruturalmente quebrado.
Rodar UMAP sobre os tokens de patch apenas **reparametrizaria o mesmo espaço
desalinhado** — não corrigiria a causa raiz. Aqui cada imagem contribui com **um único
vetor** (média dos 196 tokens, ou o token CLS), então não sobra nenhuma "posição" para
colapsar. São 256 imagens = 256 linhas, split 80/20 (seed 42) → 204 treino / 52 teste.

### 2. Dois controles, sem os quais os números não são interpretáveis

| Controle | O que é | Para que serve |
|---|---|---|
| **Inferior** (`raw_embedding`) | tabular → embedding pooled **crua** de 768 dims, sem UMAP | Se o UMAP "funciona" mas a embedding pooled crua também, o ganho veio do *pooling*, não do UMAP |
| **Superior** (`source=embedding`) | a própria embedding pooled → suas **próprias** coordenadas UMAP | Se nem a embedding original prevê suas coordenadas UMAP no teste, então um R² baixo do lado tabular **não diz nada sobre o tabular** — diz que as coordenadas são ruído fora da amostra |

O controle superior é o mais importante: é ele que separa "o dado tabular é fraco" de
"o alvo é impossível". **Ele passou** (R² ≈ 0.88–0.96), então o alvo é legítimo.

O controle inferior (tabular → pooled cru) **nunca havia sido medido** — todos os
resultados anteriores de `tabular → embedding` eram por-patch/por-posição. Ele é, por
si só, um resultado novo.

### Protocolo anti-vazamento

Todo `StandardScaler` e todo `UMAP` são ajustados **apenas no split de treino** e
aplicados ao teste, exatamente como em
`src/models/train_embedding_to_clinical.py::apply_umap`. Cada configuração é repetida
com **3 seeds de UMAP**, porque com ~204 amostras de treino uma única projeção pode
parecer previsível por acaso — o desvio entre seeds está reportado.

Modelo: `CatBoostRegressor(loss_function='MultiRMSE', iterations=150, depth=6,
learning_rate=0.08, l2_leaf_reg=3.0)` — os **mesmos hiperparâmetros** de
`src/models/train_catboost.py`, para que a comparação contra o fracasso original seja
apples-to-apples. Baseline: `DummyRegressor(strategy='mean')`.

## Resultados

Script: `src/models/train_tabular_to_umap.py`
CSVs: `outputs/tabular_to_umap/mae_{mean,cls}_tabular_to_umap.csv`

### Pooling por média

| Fonte | Alvo | dims | R² médio | desvio (3 seeds) | Pearson médio |
|---|---|---|---|---|---|
| embedding *(teto)* | umap2 | 2 | 0.961 | 0.013 | 0.986 |
| embedding *(teto)* | umap5 | 5 | 0.955 | 0.005 | 0.983 |
| embedding *(teto)* | umap10 | 10 | 0.945 | 0.006 | 0.978 |
| embedding *(teto)* | umap30 | 30 | 0.927 | 0.016 | 0.971 |
| **tabular** | **umap2** | 2 | **0.211** | 0.070 | 0.496 |
| **tabular** | **umap5** | 5 | **0.293** | 0.062 | 0.564 |
| **tabular** | **umap10** | 10 | **0.271** | 0.045 | 0.538 |
| **tabular** | **umap30** | 30 | **0.264** | 0.029 | 0.531 |
| tabular *(controle inferior)* | raw 768 | 768 | 0.082 | — | 0.353 |

### Pooling por CLS

| Fonte | Alvo | dims | R² médio | desvio (3 seeds) | Pearson médio |
|---|---|---|---|---|---|
| embedding *(teto)* | umap2 | 2 | 0.927 | 0.002 | 0.975 |
| embedding *(teto)* | umap5 | 5 | 0.911 | 0.009 | 0.965 |
| embedding *(teto)* | umap10 | 10 | 0.890 | 0.008 | 0.959 |
| embedding *(teto)* | umap30 | 30 | 0.885 | 0.015 | 0.958 |
| **tabular** | **umap2** | 2 | **0.223** | 0.072 | 0.529 |
| **tabular** | **umap5** | 5 | **0.250** | 0.035 | 0.525 |
| **tabular** | **umap10** | 10 | **0.226** | 0.020 | 0.520 |
| **tabular** | **umap30** | 30 | **0.177** | 0.020 | 0.459 |
| tabular *(controle inferior)* | raw 768 | 768 | 0.072 | — | 0.344 |

## Interpretação

**1. O resultado é real e é uma melhora grande sobre o histórico.** R² ≈ 0.26–0.29
(mean-pooled) contra **R² ≈ 0 no MAE por-patch** e **0.054 no VAE por-posição**. Pela
primeira vez nesta direção há um sinal claramente acima do baseline.

**2. O ganho não é artefato de dimensionalidade baixa.** Esta era a objeção óbvia:
R² médio sobre 2 dimensões é mais fácil de subir do que sobre 768. Se fosse só isso,
o R² cairia fortemente de 2 → 30 componentes. **Não cai** — no mean-pooling ele é
0.211 (2 dims), 0.293 (5), 0.271 (10), 0.264 (30), ou seja, praticamente **plano de 5 a
30 dimensões**, e o Pearson médio (que é insensível a escala) acompanha: 0.50 → 0.56 →
0.54 → 0.53. Isso é a evidência mais forte do experimento.

**3. O ganho se divide em duas partes distintas**, e é importante não confundi-las:
- **Sair do espaço por-patch** já vale muito: R² ≈ 0 → **0.082** só trocando por-patch
  por pooled cru. Isso confirma diretamente o diagnóstico original (o problema era o
  desalinhamento posicional, não a ausência de informação no dado tabular).
- **O UMAP acrescenta um salto por cima disso**: 0.082 → **0.293**. Comparando pelo
  Pearson, que é menos sensível à contagem de dimensões: 0.353 → 0.564.

**4. Média bate CLS**, consistentemente e em todos os níveis — mesmo padrão já
observado na direção inversa em `docs/embedding-to-clinical-log.md`.

**5. Há um teto claro e o tabular está longe dele.** A própria imagem prevê suas
coordenadas de manifold com R² ≈ 0.95; o tabular chega a ≈ 0.29. Em termos grosseiros,
**o dado clínico tabular explica cerca de um terço do que a própria imagem explica**
sobre a posição no manifold. É sinal real, mas está longe de ser uma substituição da
imagem.

## ⚠️ Ressalvas

**O vetor tabular contém o laudo do radiologista, não só demografia.**
`build_clinical_features` inclui `BIRADS`, `Shape`, `Echogenicity`, `Margin`,
`Posterior_features`, `Halo`, `Calcifications`, `Diagnosis` etc. — todas derivadas de
**olhar a mesma imagem**. Então "o tabular prevê a posição no manifold" significa, em
boa parte, "a descrição textual que o radiologista fez da imagem prevê a embedding
daquela imagem" — o que é bem menos surpreendente do que "dados demográficos do
paciente preveem a imagem". Isso **não invalida** a comparação (é exatamente o mesmo
conjunto de features do experimento original que falhou), mas muda o que o resultado
significa. Separar features de laudo de features de paciente seria o próximo passo
natural para saber quanto do sinal é qual.

**A reconstrução continua impossível por este caminho.** O UMAP não tem inversa exata,
então não dá para voltar de uma coordenada prevista até uma embedding de 768 dims e daí
ao decoder. Isso era sabido e aceito desde a formulação do experimento.

**52 amostras de teste.** Os desvios entre seeds (até 0.070 em umap2) mostram que o
número em 2 dimensões é instável. Os níveis de 5–30 componentes são bem mais estáveis
(desvios 0.020–0.062) e é neles que a conclusão deve se apoiar.

## ⚠️ Correção: medindo tudo no mesmo espaço, o UMAP **não** é um ganho

As seções acima medem R² **no espaço UMAP** (0.29) e comparam com um controle medido
**no espaço da embedding** (0.082). São espaços diferentes, então essa comparação é
sugestiva mas **não é válida**: R² médio sobre 5 coordenadas UMAP e R² médio sobre 768
dimensões de embedding não são a mesma grandeza.

O `umap-learn` tem `inverse_transform`, o que permite trazer tudo de volta para o espaço
original de 768 dims e medir as três rotas com **a mesma métrica no mesmo espaço**.
Script: `src/models/train_tabular_to_umap_inverse.py`.

| Rota (R² no espaço original de 768 dims, mean-pooled, 5 comp.) | R² |
|---|---|
| tabular → 768d **direto** (sem manifold) | **+0.082** |
| **Teto**: embedding verdadeira → UMAP → inversa, *sem regressão* | **+0.080** |
| tabular → UMAP → inversa | **−0.158** |
| baseline `DummyRegressor` | −0.040 |

**O teto é o número decisivo.** Pegar a embedding *verdadeira*, projetar e inverter
imediatamente — sem nenhuma regressão no meio — recupera apenas R² = 0.080. Ou seja,
**a projeção UMAP sozinha já destrói quase toda a variância da embedding**. Nenhuma
regressão, por melhor que seja, compensa isso: o gargalo não está na regressão, está na
projeção.

Consequência para a leitura das seções anteriores: o R² ≈ 0.29 **é real como medida de
"prever a posição no manifold"**, e sua estabilidade de 5 a 30 componentes também é
real. Mas prever bem as coordenadas do manifold **não se traduz** em prever a embedding,
porque as coordenadas são um resumo lossy demais. Medido no espaço comum, passar pelo
UMAP é pior que ir direto e pior que o baseline.

## Sobre reconstrução: o UMAP nunca foi o gargalo

Mesmo que a inversa do UMAP fosse perfeita, **a reconstrução da imagem continuaria
impossível por este caminho** — e por um motivo anterior ao UMAP. O passo irreversível
é o **pooling**: a embedding pooled é a média dos 196 tokens de patch, e nenhuma inversa
recupera 196 vetores a partir da média deles. Uma inversa perfeita entregaria um único
vetor de 768 dims, enquanto o decoder do MAE precisa dos 196 tokens. O mesmo vale para o
VAE (média espacial do latente).

## Conclusão

A ideia foi testada de forma justa e **não se sustenta como caminho para recuperar a
embedding**. O que sobrevive é mais estreito do que parecia à primeira medição:

- **Sobrevive**: o dado tabular prevê parcialmente *a posição da imagem no manifold*
  (R² ≈ 0.26–0.29, estável entre 5 e 30 componentes, contra um teto de 0.95 dado pela
  própria imagem). É o primeiro sinal não-trivial já obtido nesta direção.
- **Sobrevive**: sair do espaço por-patch para o pooled, sozinho, leva R² ≈ 0 → 0.082 no
  espaço da embedding, confirmando diretamente o diagnóstico original de que o problema
  era desalinhamento posicional.
- **Não sobrevive**: a ideia de usar isso como checkpoint rumo à embedding ou à
  reconstrução. Medido no espaço da embedding, o UMAP perde para a rota direta.

O caminho mais informativo a seguir seria separar as features tabulares em "laudo" vs.
"paciente", para descobrir se o sinal de posição no manifold é diagnóstico ou apenas a
redundância esperada entre a descrição que o radiologista fez da imagem e a imagem.

## O que acontece com as imagens fazendo só UMAP → UMAP inverso

Pergunta do usuário: e se tirarmos a regressão da jogada e fizermos apenas o
round-trip — projetar a embedding verdadeira e invertê-la de volta — como ficam as
imagens?

A embedding pooled do MAE não serve para isso (é a média de 196 tokens, não
decodificável). O **latente do VAE** serve: cada imagem é um `(1, 4, 32, 32)` = 4096
dims que o decoder do SD-VAE reconstrói diretamente. Script:
`src/models/umap_roundtrip_reconstruction.py`.

Pipeline, **sem nenhuma regressão no meio**:

```
latente verdadeiro (4096d) → StandardScaler → UMAP(n) → inverse_transform
                           → desfaz escala → reshape (1,4,32,32) → decoder VAE → imagem
```

O UMAP é ajustado **apenas no split de treino** e as imagens mostradas são do split de
**teste**, então o que se vê é comportamento genuíno fora da amostra. Cada strip tem 3
painéis: `original`, `VAE roundtrip` (decode do latente verdadeiro — o **teto**, já que
toda rota passa pelo mesmo decoder) e `UMAP-n roundtrip`. **A comparação que isola a
perda do UMAP é o painel 3 contra o painel 2**, não contra o original.

### Resultado numérico

| Componentes | R² no espaço latente (52 imgs de teste) | SSIM vs. reconstrução do VAE | PSNR vs. VAE |
|---|---|---|---|
| 2 | **−0.764** | 0.139 | 14.40 |
| 5 | **−0.751** | 0.153 | 14.88 |

R² **negativo** significa pior do que simplesmente chutar o latente médio do treino
para toda imagem. O round-trip não é uma compressão com perda — é destruição.

### Resultado visual

As imagens ficam **irreconhecíveis como ultrassom**. O que sobrevive é apenas a
silhueta global mais grosseira: em imagens com o envelope em leque da sonda
(ex. `case207`), o formato do leque é preservado, mas o interior vira textura sem
estrutura. Em imagens retangulares (ex. `case007`) não sobra nada identificável.

Dois sintomas revelam o que está acontecendo:
- **Toda a textura interna desaparece** — e é exatamente nela que mora o conteúdo
  diagnóstico do ultrassom (a lesão, suas margens, sombra acústica posterior).
- **Surge cor** numa modalidade que é grayscale. O latente invertido cai **fora da
  variedade de latentes reais**, numa região que o decoder nunca viu, e responde com
  artefatos coloridos. É o sinal mais claro de que `inverse_transform` não devolve um
  latente plausível, apenas um ponto qualquer que interpola pontos de treino.

### Limite estrutural: acima de ~5 dimensões a inversa nem roda

O próprio `umap-learn` avisa (`umap_.py:3210`) que `inverse_transform` funciona melhor
em baixa dimensão e "pode falhar completamente". Confirmado nesta máquina:

| Componentes | O que acontece |
|---|---|
| 2, 5 | roda (resultados acima) |
| 10 | **morre por falta de memória** (exit 137) — a triangulação de Delaunay em 10-D explode |
| 50 | **`QhullError`**: `negative request size (-75562456). Did int overflow due to high-D?` |

Ou seja, não adianta apostar em "mais dimensões resolvem": a inversa do UMAP é
utilizável apenas na faixa de 2–5 dimensões, justamente a faixa que mais destrói
informação. O próprio aviso da biblioteca recomenda um **autoencoder** para quem precisa
de espaço latente de dimensão alta *com* operação inversa — que é precisamente a
descrição do problema aqui.

### Conclusão desta parte

O round-trip UMAP→inverso **não é um caminho viável para reconstrução**, e o motivo é
independente da qualidade de qualquer regressor: a perda já acontece antes, na projeção.
Isto é a contraparte visual do R² ≈ 0.08 medido no espaço do MAE — os dois experimentos,
em modelos e espaços diferentes, dizem a mesma coisa. Se o objetivo for reconstrução a
partir de um espaço reduzido, o instrumento certo é um autoencoder treinado com perda de
reconstrução (que otimiza exatamente isso), não um método de manifold learning cujo
objetivo é preservar vizinhança para visualização.
