# Boston Housing K-Means

Clusterização de regiões censitárias de Boston com K-Means, com treinamento reproduzível e inferência de novos registros por terminal ou JSON.

## Objetivo

Identificar perfis de regiões a partir de características estruturais, ambientais, geográficas e socioeconômicas, sem usar o valor dos imóveis (`MEDV`). O projeto tem duas etapas independentes:

1. **Treinamento** (`python -m src.train`): analisa os dados, escolhe K, treina o modelo e salva o artefato.
2. **Inferência** (`python -m src.inference`): carrega o artefato e classifica uma nova observação, sem treinar novamente.

O K-Means divide os registros em K grupos, alternando entre atribuir cada registro ao centroide mais próximo e recalcular cada centroide como a média de seu grupo. O processo minimiza a inércia, que é a soma dos quadrados das distâncias euclidianas entre os registros e seus centroides.

> **Finalidade exploratória e demonstrativa.** Os agrupamentos não são classificações objetivas da qualidade de uma região, e o modelo não deve ser usado isoladamente em decisões reais de crédito, moradia, investimento ou avaliação imobiliária. Veja [Limitações](#limitações-da-base).

### Registros são regiões, não residências

Cada linha da base descreve uma **região censitária**, e não uma residência isolada. Por isso, a interface usa a expressão "nova residência", mas a inferência representa tecnicamente um novo perfil de região com as mesmas 13 características.

## Tecnologias

| Biblioteca | Uso |
|---|---|
| scikit-learn | `Pipeline`, `ColumnTransformer`, imputação, padronização, One-Hot, `KMeans` e métricas |
| pandas e NumPy | Leitura e manipulação dos dados |
| matplotlib | Gráficos |
| joblib | Persistência do modelo |
| pytest | Testes automatizados |

Ambiente validado: Python 3.14.3 no Windows 11, com as versões fixadas em `requirements.txt`.

## Estrutura

```text
boston-housing-kmeans/
├── data/                  Base local (fora do Git) e .gitkeep
├── models/                kmeans_model.joblib: pipeline completo e metadados
├── reports/               Tabelas e gráficos gerados pelo treinamento
├── experiments/           Treinamentos com K experimental (gerado sob demanda, fora do Git)
├── src/
│   ├── config.py          Caminhos, variáveis, hiperparâmetros e K final
│   ├── preprocessing.py   Leitura, validação, pré-processador e desnormalização
│   ├── eda.py             Análise exploratória
│   ├── train.py           Seleção de K, treinamento, relatórios e persistência
│   └── inference.py       Validação da entrada, classificação e apresentação
├── tests/test_pipeline.py Testes com dados sintéticos e com a base real, se presente
├── examples/nova_residencia.json  Entrada ilustrativa
├── requirements.txt
├── .gitignore
└── README.md
```

Em relação à estrutura planejada, foram acrescentados `src/eda.py`, para isolar a análise exploratória, e os relatórios `feature_boxplots.png`, `correlation_matrix.csv`, `silhouette_scores.png`, `k_selection.json`, `cluster_analysis.csv` e `feature_influence.csv`.

## Base de dados

| Item | Resultado |
|---|---|
| Registros e colunas | 506 e 14 |
| Valores ausentes | 120 (1,69% das células) |
| Registros com ao menos um ausente | 112 (22,13%): 104 com um e 8 com dois |
| Registros completos | 394 |
| Registros duplicados | 0, também considerando só as 13 características |
| Registros com outlier IQR em ao menos uma quantitativa | 204 |

Nenhum registro foi removido. Duplicados não existem nesta versão, e os incompletos são tratados pelos imputadores.

### Variáveis

| Variável | Significado | Papel | Ausentes |
|---|---|---|---|
| CRIM | Taxa de criminalidade per capita | Quantitativa | 20 |
| ZN | % de terrenos residenciais para lotes acima de 25.000 pés² | Quantitativa | 20 |
| INDUS | % de área comercial não varejista | Quantitativa | 20 |
| CHAS | Fronteira com o rio Charles: 0 = não, 1 = sim | Categórica binária | 20 |
| NOX | Óxidos nítricos, em partes por 10 milhões | Quantitativa | 0 |
| RM | Número médio de cômodos por residência | Quantitativa | 0 |
| AGE | % de unidades ocupadas pelos proprietários anteriores a 1940 | Quantitativa | 20 |
| DIS | Distância ponderada a cinco centros de emprego | Quantitativa | 0 |
| RAD | Índice de acessibilidade a rodovias radiais | Quantitativa discreta | 0 |
| TAX | Imposto sobre propriedade por US$ 10.000 | Quantitativa | 0 |
| PTRATIO | Razão aluno-professor | Quantitativa | 0 |
| B | 1000(Bk − 0,63)², em que Bk é a proporção de pessoas negras | Quantitativa | 0 |
| LSTAT | % da população de menor status socioeconômico | Quantitativa | 20 |
| MEDV | Valor mediano das residências, em milhares de dólares | **Fora do modelo** | 0 |

Cada uma das seis colunas incompletas tem exatamente 20 ausentes. Esse padrão uniforme sugere remoção proposital em relação à versão clássica da base, mas o arquivo não documenta sua origem.

### Principais observações

- **Assimetria forte:** CRIM (5,21), B (−2,89) e ZN (2,26). A mediana de CRIM é 0,25, e o máximo é 88,98.
- **Concentração em um valor:** ZN é 0 em 74,07% dos registros válidos, B está no teto de 396,9 em 23,91% e MEDV está no teto de 50 em 3,16%.
- **Bloco RAD/TAX:** RAD = 24 e TAX = 666 representam 26,09% dos registros cada. A correlação entre elas é 0,91 em Pearson e 0,70 em Spearman, o que indica que o bloco infla a medida linear.
- **Variáveis redundantes:** NOX e DIS (−0,77 em Pearson, −0,88 em Spearman), AGE e DIS (−0,74, −0,80) e INDUS e NOX (0,76, 0,79) medem um eixo comum de urbanização. O K-Means não desconta essa redundância, de modo que o eixo pesa mais na distância.
- **CHAS:** 452 registros com 0, 34 com 1 e 20 ausentes. É analisada por frequência e fica fora do IQR e da matriz de correlação.

### Outliers pelo critério de 1,5 × IQR

| Variável | Abaixo | Acima | Total | % |
|---|---:|---:|---:|---:|
| B | 77 | 0 | 77 | 15,22 |
| CRIM | 0 | 65 | 65 | 13,37 |
| ZN | 0 | 63 | 63 | 12,96 |
| RM | 8 | 22 | 30 | 5,93 |
| PTRATIO | 15 | 0 | 15 | 2,96 |
| LSTAT | 0 | 7 | 7 | 1,44 |
| DIS | 0 | 5 | 5 | 0,99 |
| INDUS, NOX, AGE, RAD, TAX | 0 | 0 | 0 | 0,00 |

Em CRIM, ZN e B, boa parte dos pontos marcados reflete a forma assimétrica da distribuição, e não erro de medição. Por isso, os outliers foram mantidos e documentados.

![Distribuição das variáveis](reports/feature_distributions.png)

### Limitações da base

- A base é histórica, de 1978 (Harrison & Rubinfeld, *Hedonic prices and the demand for clean air*), e não representa Boston atualmente.
- A variável `B` foi construída sobre a composição racial da população e sobre a hipótese de que a segregação afetava os preços. O scikit-learn removeu `load_boston` por esse problema ético, citando a análise [Racist data destruction?](https://medium.com/@docintangible/racist-data-destruction-113e3eff54a8).
- `LSTAT` é uma medida socioeconômica sensível, cuja definição original também merece cautela.
- `B` e `LSTAT` foram mantidas por fazerem parte da definição da base e da metodologia adotada, e não por serem neutras.

### Redistribuição e como obter a base

Não foi encontrada licença que confirme a redistribuição deste CSV. A origem é o trabalho de 1978, disseminado pelo arquivo StatLib, e os espelhos públicos trazem informações conflitantes. Por isso, `.gitignore` mantém o arquivo fora do repositório e `data/.gitkeep` preserva a pasta.

Para reproduzir o treinamento, coloque o arquivo em `data/HousingData.csv` ou informe outro caminho com `--data`. A versão usada aqui tem este SHA-256:

```text
7a1eefeaed0bcc17daa972f9e2131b802d0fde5347c5f2ae80af852088dd5f43
```

## Pré-processamento

Tudo ocorre em um `Pipeline` com `ColumnTransformer`, ajustado apenas no treinamento:

| Grupo | Variáveis | Etapas |
|---|---|---|
| Quantitativas | 12: CRIM, ZN, INDUS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT | `SimpleImputer(median)` e `StandardScaler` |
| Categórica | CHAS | `SimpleImputer(most_frequent)` e `OneHotEncoder(categories=[[0, 1]], handle_unknown="ignore")` |

As 13 características de entrada geram 14 colunas transformadas: 12 numéricas e 2 de CHAS.

- **Exclusão de MEDV:** o objetivo é agrupar regiões sem depender do valor de mercado, e a inferência não pode exigir esse valor. MEDV fica fora do ajuste, da normalização, dos centroides e da seleção de K. Ela aparece apenas na análise exploratória e em uma descrição a posteriori dos clusters.
- **Mediana:** é robusta à assimetria das variáveis com ausentes. Como limitação, ignora o contexto: um INDUS ausente em uma região com RAD = 24 recebe a mediana geral (9,69), embora o valor típico desse bloco seja 18,1.
- **Moda em CHAS:** preserva uma categoria válida; o valor aprendido é 0.
- **StandardScaler:** o K-Means usa distâncias euclidianas e seria dominado pelas maiores escalas. Sem padronização, TAX responderia por 74,11% da variância total, e B, por 21,74%.
- **OneHotEncoder sem padronização posterior:** a média de cada coluna em um cluster é diretamente a proporção da categoria. Duas regiões com CHAS diferente somam 2 à distância quadrática.
- **RAD como índice numérico:** é um índice discreto de acessibilidade e foi mantido como característica numérica, conforme a estratégia do projeto, sem One-Hot. A limitação é que o K-Means assume uma relação ordenada e usa as distâncias numéricas entre índices, então o salto de 8 para 24 pesa como se fosse uma escala contínua.
- **Outliers e balanceamento:** não há remoção de outliers nem balanceamento. O K-Means é sensível a valores extremos, pois eles deslocam as médias. Esse efeito é medido em [Análise crítica](#análise-crítica).

## Treinamento

### Configuração

`KMeans(n_clusters=K, random_state=42, n_init=20, max_iter=300, algorithm="lloyd")`, usada em toda a análise. Duas execuções completas geraram os 18 arquivos de `reports/` e `models/` idênticos byte a byte.

Não há divisão treino/teste, pois o objetivo é descrever a estrutura da base completa. Inércia e Silhouette são calculados nos próprios registros de treinamento: avaliam os agrupamentos encontrados, mas não medem, por si sós, a generalização para novos dados. A estabilidade por reamostragem foi incluída para reduzir essa lacuna.

### Procedimento de escolha de K

1. **Cotovelo geométrico:** K e inércia são normalizados para [0, 1], traça-se a reta entre os pontos extremos da curva e escolhe-se o ponto de maior distância perpendicular a ela.
2. **Silhouette (K de 2 a 10):** compara a coesão interna com a separação entre grupos. K = 1 não tem Silhouette.
3. **Distribuição:** nenhum cluster pode ficar abaixo de 5% dos registros.
4. **Estabilidade:** ARI médio e percentil 5 entre a partição de referência e 50 modelos treinados em reamostragens.
5. **Interpretabilidade:** avaliada na análise abaixo, por não ser calculável.

A regra automática de `select_k` adota o cotovelo geométrico se ele respeitar o tamanho mínimo; caso contrário, o maior Silhouette entre os K que o respeitam. Nesta base, ela indica K = 3. A decisão final do responsável pelo projeto é **K = 2**, definida em `FINAL_K` (`src/config.py`). O arquivo `reports/k_selection.json` registra as duas informações.

![Método do cotovelo](reports/elbow_method.png)

| K | Inércia | Redução vs. K anterior (%) | Inércia explicada (%) | Distância ao segmento | Silhouette | Tamanhos | ARI médio | ARI p5 |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | 6135,43 | – | 0,00 | 0,0000 | – | 506 | – | – |
| **2** | **3804,50** | **37,99** | **37,99** | **0,2797** | **0,3714** | **330, 176** | **0,966** | **0,937** |
| 3 | 3140,37 | 17,46 | 48,82 | 0,3031 | 0,2815 | 258, 167, 81 | 0,778 | 0,405 |
| 4 | 2768,72 | 11,83 | 54,87 | 0,2817 | 0,2711 | 237, 132, 79, 58 | 0,682 | 0,497 |
| 5 | 2447,41 | 11,61 | 60,11 | 0,2525 | 0,2538 | 184, 130, 100, 55, 37 | 0,613 | 0,436 |
| 6 | 2173,97 | 11,17 | 64,57 | 0,2160 | 0,2610 | 163, 110, 100, 55, 41, 37 | 0,738 | 0,495 |
| 7 | 1968,69 | 9,44 | 67,91 | 0,1689 | 0,2767 | 124, 110, 100, 56, 53, 37, 26 | 0,795 | 0,556 |
| 8 | 1776,45 | 9,77 | 71,05 | 0,1199 | 0,2848 | 121, 113, 98, 57, 53, 32, 25, 7 | 0,854 | 0,744 |
| 9 | 1647,27 | 7,27 | 73,15 | 0,0612 | 0,2846 | 110, 107, 93, 54, 53, 30, 25, 25, 9 | 0,881 | 0,721 |
| 10 | 1534,29 | 6,86 | 74,99 | 0,0000 | 0,2795 | 114, 109, 74, 53, 47, 31, 26, 25, 22, 5 | 0,868 | 0,695 |

## Resultados

### K selecionado: 2

| Métrica do modelo definitivo | Valor |
|---|---|
| Inércia | 3804,4977 |
| Silhouette | 0,3714 |
| Inércia explicada | 37,99% |
| Iterações até a convergência | 8 |

### Comparação entre K = 2 e K = 3

K = 3 foi o candidato indicado pelo método geométrico na faixa de 1 a 10. K = 2 foi escolhido considerando a ambiguidade do cotovelo, o Silhouette Score e a estabilidade dos agrupamentos. Os valores de K = 3 vêm da mesma análise e foram mantidos sem alteração em `reports/elbow_results.csv`.

| Critério | K = 2 (adotado) | K = 3 (candidato geométrico) |
|---|---:|---:|
| Distância ao segmento | 0,2797 | 0,3031 |
| Inércia | 3804,50 | 3140,37 |
| Inércia explicada | 37,99% | 48,82% |
| Silhouette | 0,3714 | 0,2815 |
| ARI médio | 0,966 | 0,778 |
| ARI p5 | 0,937 | 0,405 |
| Tamanhos | 330, 176 | 258, 167, 81 |
| Menor cluster | 34,78% | 16,01% |

**Por que K = 2:**

- **Cotovelo ambíguo:** a distância em K = 3 (0,3031) é pouco maior que em K = 4 (0,2817) e K = 2 (0,2797). O candidato também muda com a faixa avaliada: é 2 nas faixas 1–6 e 1–7, 3 nas faixas 1–8, 1–9 e 1–10, e 5 na faixa 2–10. Além disso, a maior queda de inércia ocorre de K = 1 para K = 2 (37,99%).
- **Silhouette superior:** 0,3714 é o maior valor entre todos os K avaliados, contra 0,2815 em K = 3.
- **Estabilidade superior:** o ARI médio é 0,966 em K = 2, contra 0,778 em K = 3. O percentil 5 cai de 0,937 para 0,405, o que mostra instabilidade relevante de K = 3. A solução de K = 3 é a mesma em 15 sementes na base completa, então não se trata de mínimo local. Em verificação pontual, porém, parte das reamostragens produziu outra partição, com grupos próximos de 205, 165 e 134 registros.

**O que se perde com K = 2:**

- O modelo explica menos inércia: 37,99% contra 48,82%.
- O perfil de baixa densidade e alto zoneamento residencial, que formava um grupo de 81 registros em K = 3, fica misturado no Cluster 0. Por isso, o centroide desse cluster representa um grupo heterogêneo.

Para examinar a alternativa, execute `python -m src.train --k 3`. Os resultados vão para `experiments/k3/`, sem alterar o modelo e os relatórios oficiais de K = 2. Veja [Treinamentos experimentais](#treinamentos-experimentais).

### Distribuição

| Cluster | Registros | % |
|---:|---:|---:|
| 0 | 330 | 65,22 |
| 1 | 176 | 34,78 |

![Distribuição dos clusters](reports/cluster_distribution.png)

Os grupos têm tamanhos diferentes, sem balanceamento artificial, e nenhum é muito pequeno.

### Centroides na escala original

Os valores são médias dos registros de cada cluster, incluindo os imputados. São perfis médios encontrados pelo algoritmo, e não regiões reais.

| Variável | Cluster 0 | Cluster 1 |
|---|---:|---:|
| CRIM | 0,2633 | 9,5089 |
| ZN | 16,5121 | 0,0000 |
| INDUS | 7,0008 | 18,5815 |
| NOX | 0,4873 | 0,6811 |
| RM | 6,4536 | 5,9678 |
| AGE | 57,2967 | 90,5006 |
| DIS | 4,7483 | 2,0076 |
| RAD (média do índice) | 4,4636 | 19,0852 |
| TAX | 301,5727 | 608,2330 |
| PTRATIO | 17,8412 | 19,6074 |
| B | 386,3990 | 300,9398 |
| LSTAT | 9,6696 | 18,2803 |
| CHAS: categoria 0 | 93,03% | 93,75% |
| CHAS: categoria 1 | 6,97% | 6,25% |
| CHAS: predominante | 0 | 0 |
| RAD mais frequente nos registros (complementar) | 5 (28,48%) | 24 (75,00%) |

### Características dos grupos

Os números dos clusters são apenas identificadores: Cluster 0 não é melhor nem pior que Cluster 1.

- **Cluster 0, menor densidade:** concentra as regiões com zoneamento residencial, com menores CRIM, INDUS, NOX, TAX e LSTAT e maior DIS. É heterogêneo, pois reúne perfis intermediários e de baixa densidade.
- **Cluster 1, alta densidade e bloco RAD = 24:** maiores CRIM, INDUS, NOX, AGE, TAX e LSTAT, menor DIS e nenhum zoneamento para lotes grandes. Reúne os 132 registros da base com RAD = 24 e outros 44 registros.

### Análise crítica

| Cluster | Distância média | Distância p95 | Distância máxima | Silhouette médio | Silhouette negativos |
|---:|---:|---:|---:|---:|---:|
| 0 | 2,4029 | 4,2102 | 5,0427 | 0,4064 | 0 |
| 1 | 2,7996 | 4,4960 | 9,4822 | 0,3056 | 8 |

- **Coesão:** o Silhouette geral de 0,3714 é o melhor entre os K avaliados, mas ainda é moderado. O Cluster 0 é mais coeso que o Cluster 1, que concentra os 8 registros com Silhouette negativo, mais próximos do outro grupo.
- **Valores extremos:** o Cluster 1 é o mais disperso, e sua distância máxima de 9,48 pertence ao registro de maior CRIM (88,98). A padronização reduz o domínio das escalas, mas não elimina a influência de extremos.
- **Peso das variáveis:** TAX (16,34%), INDUS (14,72%), RAD (13,91%) e NOX (13,80%) concentram a separação entre grupos em `reports/feature_influence.csv`. ZN responde por apenas 2,54%, o que confirma que o zoneamento residencial pouco distingue os dois grupos.
- **Dependência do bloco RAD/TAX:** a divisão se apoia fortemente nesse bloco. Como RAD é tratada numericamente, o salto de 8 para 24 reforça a separação, de modo que parte dela decorre dessa decisão de modelagem.
- **CHAS quase não contribui:** responde por 0,0005% da separação, e os dois clusters têm composição praticamente igual. A proximidade com o rio não caracteriza nenhum grupo.
- **MEDV a posteriori:** as médias foram 25,72 no Cluster 0 e 16,55 no Cluster 1. Os grupos diferem em valor sem que MEDV tenha participado do treinamento, mas essa comparação é descritiva e ocorreu depois da escolha de K.

## Inferência

1. A entrada é validada: exige as 13 características e aceita `identificacao` como opcional.
2. Os valores viram um `DataFrame`, na ordem canônica das colunas.
3. O pipeline salvo aplica imputadores, scaler e encoder com os parâmetros do treinamento, sem nenhum novo ajuste.
4. `predict()` atribui o cluster do centroide mais próximo.
5. A distância euclidiana ao centroide atribuído é calculada no espaço transformado de 14 colunas. Ela não está nas unidades originais das variáveis.
6. O resultado mostra o centroide na escala original.

### Reconstrução dos centroides

O `ColumnTransformer` não oferece `inverse_transform`, então `denormalize_centroids` reconstrói cada bloco de forma explícita:

- **Quantitativas:** usa `StandardScaler.inverse_transform` nas colunas numéricas do centroide. O resultado é a média do cluster. Por isso, RAD pode ser decimal, e essa média não é arredondada nem tratada como categoria.
- **CHAS:** as colunas One-Hot do centroide já são proporções das categorias no cluster, considerando a imputação. As categorias vêm do `OneHotEncoder` ajustado, e a predominante é a de maior proporção. CHAS não é interpretada como variável contínua.

### Regras de validação

| Situação | Comportamento |
|---|---|
| Campo ausente, característica inesperada ou `MEDV` informada | Erro |
| Texto, booleano, lista, `NaN`, `Infinity` ou número não finito | Erro |
| CHAS diferente de 0 ou 1 | Erro |
| Valor negativo, ou percentual (ZN, INDUS, AGE, LSTAT) acima de 100 | Erro, por contrariar a definição da variável |
| JSON inválido ou com campos duplicados | Erro |
| Modelo inexistente, corrompido ou com estrutura inesperada | Erro, com orientação |
| Campo presente com `null` | Aceito: imputado com o valor do treinamento e informado em aviso |
| Valor fora do intervalo do treinamento, ou RAD não observado | Aceito, com aviso informativo |
| Versão do scikit-learn diferente da usada no treinamento | Aceito, com aviso |

## Execução

### Instalação

```bash
python -m venv .venv
```

Ativação no Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

Ativação no Linux/macOS:

```bash
source .venv/bin/activate
```

Instalação das dependências:

```bash
pip install -r requirements.txt
```

As versões fixadas foram validadas com Python 3.14. Versões mais antigas do Python podem não atender a elas.

### Treinamento

Requer a base em `data/HousingData.csv` e leva cerca de um minuto e meio, em razão da análise de estabilidade.

```bash
python -m src.train
```

Com outra base:

```bash
python -m src.train --data caminho/para/HousingData.csv
```

K = 2 é o padrão, definido em `FINAL_K` (`src/config.py`). Este comando é equivalente a `python -m src.train` e gera os mesmos arquivos:

```bash
python -m src.train --k 2
```

`--k` é validado antes de qualquer processamento: aceita inteiros de 2 a 10 (`K_MAX`), que é a faixa em que o Silhouette é avaliado. Fora dela, como em `--k 1`, o comando termina imediatamente com código 2, sem análise nem gravação de arquivos.

#### Treinamentos experimentais

Um K diferente de `FINAL_K` é tratado como experimento e nunca grava em `models/` ou `reports/`:

```bash
python -m src.train --k 3
```

| Execução | Modelo | Relatórios |
|---|---|---|
| `python -m src.train` ou `--k 2` (oficial) | `models/kmeans_model.joblib` | `reports/` |
| `--k 3` (experimental) | `experiments/k3/kmeans_model.joblib` | `experiments/k3/reports/` |

A pasta `experiments/` é criada sob demanda, pode ser apagada a qualquer momento e fica fora do Git. Para classificar uma observação com um modelo experimental, use `--model`:

```bash
python -m src.inference --json examples/nova_residencia.json --model experiments/k3/kmeans_model.joblib
```

Somente a análise exploratória:

```bash
python -m src.eda
```

### Inferência interativa

```bash
python -m src.inference
```

O sistema solicita a identificação e as 13 características. Decimais aceitam ponto ou vírgula, e `null` indica valor desconhecido.

### Inferência por JSON

```bash
python -m src.inference --json examples/nova_residencia.json
```

Os valores de `examples/nova_residencia.json` são ilustrativos e não representam uma observação real. Saída obtida com o modelo treinado:

```text
==============================================
           RESULTADO DA INFERÊNCIA            
==============================================

Identificação: Regiao demonstrativa

Cluster atribuído: 0
(O número identifica o grupo; não indica qualidade)

Distância até o centroide: 1.2785
(Espaço normalizado e codificado)
Referência do cluster no treinamento: média 2.4029, p95 4.2102

----------------------------------------------
CENTROIDE NA ESCALA ORIGINAL
----------------------------------------------
(Perfil médio do cluster, não uma região real)

CRIM:       0.2633
ZN:         16.5121
INDUS:      7.0008
NOX:        0.4873
RM:         6.4536
AGE:        57.2967
DIS:        4.7483
RAD:        4.4636
TAX:        301.5727
PTRATIO:    17.8412
B:          386.3990
LSTAT:      9.6696

CHAS:

Categoria 0: 93.03%
Categoria 1: 6.97%

Categoria predominante: 0 (93.03%)

----------------------------------------------
ESTATÍSTICA COMPLEMENTAR (NÃO FAZ PARTE DO CENTROIDE)
----------------------------------------------
RAD mais frequente nos registros reais do cluster: 5 (28.48%)

==============================================
```

### Testes

```bash
pytest -v
```

Os testes usam dados sintéticos e não dependem da base real. Os dois testes que a utilizam são ignorados se o CSV não estiver em `data/`.

## Artefato do modelo

`models/kmeans_model.joblib` contém um dicionário com:

- `pipeline`: `Pipeline` com as etapas `preprocessor` e `kmeans`, incluindo imputadores, scaler, encoder e centroides ajustados;
- `metadata`: variáveis, configuração, K, métricas, intervalos do treinamento, organização dos centroides e versões de Python e bibliotecas.

O artefato guarda estatísticas agregadas e o rótulo de cluster de cada registro, mas não os valores da base. Ele deve ser usado com as versões de `requirements.txt`. Como Joblib usa pickle, carregue apenas arquivos de origem confiável.

## Publicação

O diretório está pronto para virar um repositório independente, mas nenhuma operação de Git foi executada. A publicação é manual e deve ocorrer após a revisão, dentro da pasta do projeto:

```bash
git init -b main
```

```bash
git add .
```

```bash
git status
```

Confira que `data/HousingData.csv` e `.venv/` não aparecem na lista.

```bash
git commit -m "Projeto de clusterização K-Means da base Boston Housing"
```

Crie um repositório vazio chamado `boston-housing-kmeans` no GitHub e substitua `<usuario>` abaixo:

```bash
git remote add origin https://github.com/<usuario>/boston-housing-kmeans.git
```

```bash
git push -u origin main
```
