# Dados do ISIC: o que baixar (Entrega 2)

Fonte consultada: <https://challenge.isic-archive.com/data/> e <https://challenge.isic-archive.com/landing/2017/> 

## 1. Edição recomendada: ISIC 2017 (ISBI 2017)

Por que a de 2017 e não as outras:

| Edição | Prós para o projeto | Contras para o projeto |
|---|---|---|
| **2017 (recomendada)** | Tem as três coisas que a proposta usa: **máscaras de segmentação feitas por especialistas** (Parte 1), para medir a segmentação por limiar com IoU/Dice; **rótulo de melanoma** (Parte 3); e divisão oficial **treino/validação/teste**. Imagens dermatoscópicas com pelos, vinheta e variação de cor, que justificam cada etapa do pré-processamento. Licença CC-0. | Download grande (~12 GB no total) e classes desbalanceadas (poucos melanomas). |
| 2016 | Pequena (900 treino + 379 teste, ~600 MB + ~230 MB), máscaras e rótulo benigno/maligno, CC-0. Boa alternativa se faltar disco ou tempo. | Menos imagens, sem conjunto de validação. |
| 2018 | Tarefa 3 = HAM10000 (10.015 imagens, 7 classes). | As máscaras (Tarefa 1) cobrem só 2.594 imagens, que não são as mesmas da Tarefa 3. Licença CC-BY-NC na Tarefa 3. 10,4 GB só a entrada da Tarefa 1. |
| 2019 | 25.331 imagens, 8 classes, metadados. | Sem máscaras de segmentação; 9,1 GB; CC-BY-NC. |
| 2020 | 33.126 imagens, rótulo binário de melanoma, metadados do paciente. | Sem máscaras; 23 GB (JPEG); melanoma em proporção muito pequena; CC-BY-NC. |
| 2024 | 401.059 recortes de fotografia 3D de corpo inteiro (SLICE-3D). | Não são imagens dermatoscópicas: fogem do problema da proposta. |

## 2. Arquivos para baixar (ISIC 2017)

| # | Arquivo (nome no site) | Tamanho | Conteúdo | Necessário? |
|---|---|---|---|---|
| 1 | `ISIC-2017_Training_Data.zip` | 5,8 GB | 2.000 imagens JPEG de treino **+ arquivos `*_superpixels.png`** (da Parte 2, ignorados pelos scripts) | **Sim** |
| 2 | `ISIC-2017_Training_Part1_GroundTruth.zip` | 9 MB | 2.000 máscaras binárias PNG (`ISIC_xxxxxxx_segmentation.png`) | **Sim** |
| 3 | `ISIC-2017_Training_Part3_GroundTruth.csv` | 43 KB | Rótulos: `image_id, melanoma, seborrheic_keratosis` | **Sim** |
| 4 | `ISIC-2017_Validation_Data.zip` | 878 MB | ~150 imagens de validação | Sim (ajuste de parâmetros) |
| 5 | `ISIC-2017_Validation_Part1_GroundTruth.zip` | 559 KB | Máscaras da validação | Sim |
| 6 | `ISIC-2017_Validation_Part3_GroundTruth.csv` | 3 KB | Rótulos da validação | Sim |
| 7 | `ISIC-2017_Test_v2_Data.zip` | 5,4 GB | ~600 imagens de teste | Sim, para o resultado final |
| 8 | `ISIC-2017_Test_v2_Part1_GroundTruth.zip` | 18 MB | Máscaras do teste | Sim |
| 9 | `ISIC-2017_Test_v2_Part3_GroundTruth.csv` | 13 KB | Rótulos do teste | Sim |

Total aproximado: **12,1 GB** compactados. O `01_inventario.py` também lê, se existir, um CSV de metadados (idade e sexo) dentro da pasta de imagens (`*metadata*.csv`). Não consegui confirmar pelo site se esse arquivo vem dentro do zip de 2017. Se não vier, o pipeline funciona normalmente sem ele.

## 3. Onde colocar os arquivos

Extraia tudo em `dados/isic2017/`. O `01_inventario.py` encontra as pastas mesmo que o zip crie um nível a mais:

```
ProjetoMelanoma/
└── dados/
    └── isic2017/
        ├── ISIC-2017_Training_Data/                 (ISIC_0000000.jpg, ...)
        ├── ISIC-2017_Training_Part1_GroundTruth/    (ISIC_0000000_segmentation.png, ...)
        ├── ISIC-2017_Training_Part3_GroundTruth.csv
        ├── ISIC-2017_Validation_Data/
        ├── ISIC-2017_Validation_Part1_GroundTruth/
        ├── ISIC-2017_Validation_Part3_GroundTruth.csv
        ├── ISIC-2017_Test_v2_Data/
        ├── ISIC-2017_Test_v2_Part1_GroundTruth/
        └── ISIC-2017_Test_v2_Part3_GroundTruth.csv
```
