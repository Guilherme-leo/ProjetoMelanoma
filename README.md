# Pré-processamento de imagens dermatoscópicas para apoio à triagem de melanoma

Disciplina: Processamento de Imagens (UFRPE). Autores: Guilherme Leonardo Guimarães, Guilherme Salgueiro Silveira e Ronaldo de Araújo Ribeiro Filho.

> **Ferramenta de apoio, não de diagnóstico.** O sistema só prepara as imagens e extrai medidas objetivas para ajudar o dermatologista. Ele **não** dá diagnóstico e **não** substitui o exame clínico nem a biópsia.

O pipeline tem 11 etapas. São scripts separados que rodam em sequência: cada etapa lê a saída da anterior e grava a sua própria saída em uma pasta numerada.

## 1. Estrutura de pastas

```
ProjetoMelanoma/
├── README.md                  ← este arquivo
├── DADOS_ISIC.md              ← o que baixar do ISIC, licença e citação
├── requirements.txt
├── artigo/                    ← artigo parcial (.docx)
├── dados/
│   └── isic2017/              ← coloque aqui os arquivos baixados (ver DADOS_ISIC.md, seção 6)
├── scripts/
│   ├── comum.py               ← utilitários (leitura/escrita, log, manifesto, exemplos)
│   ├── metricas.py            ← métricas de classificação (usadas na etapa 11)
│   ├── 01_inventario.py       ← valida a base e gera o manifesto
│   ├── 02_redimensionamento.py
│   ├── 03_remocao_bordas.py   ← vinheta/bordas pretas + campo de visão (FOV)
│   ├── 04_remocao_pelos.py    ← black-hat linear + inpainting (e reflexos)
│   ├── 05_constancia_cor.py   ← Shades of Gray (p = 6)
│   ├── 06_realce_contraste.py ← CLAHE no canal L* (CIELAB)  → imagem para VISUALIZAÇÃO
│   ├── 07_filtragem_fourier.py← passa-baixa gaussiano na DFT  → imagem para SEGMENTAÇÃO
│   ├── 08_padronizacao.py     ← 256×256 + média/desvio do treino
│   ├── 09_segmentacao.py      ← Otsu (do zero) + morfologia; IoU/Dice contra o ISIC
│   ├── 10_descritores.py      ← ABCD, textura, momentos de Hu, descritores de Fourier
│   ├── 11_classificacao.py    ← SVM, Random Forest, Regressão Logística (scikit-learn)
│   └── juncao.py              ← roda tudo (ou um intervalo) em ordem
└── saidas/                    ← criada na execução: 01_inventario/, 02_redimensionamento/, ...
```

Toda pasta de saída de etapa (`saidas/NN_nome/`) contém:

| Arquivo | Para que serve |
|---|---|
| `manifesto.csv` | Uma linha por imagem que deu certo: caminhos (relativos), rótulo, conjunto e medidas da etapa. **É a entrada da etapa seguinte.** |
| `imagens/` | Imagens processadas (PNG, sem perdas). Algumas etapas também gravam `mascaras/`, `fov/`, `pelos/` e `mascaras_pred/`. |
| `erros.csv` | Imagens que falharam e o motivo. O lote **não** para por causa delas. |
| `etapa.log` | Log detalhado, com parâmetros, cada imagem processada e avisos. |
| `resumo.json` | Contagens, parâmetros usados, tempo total e tempo médio por imagem. |
| `exemplos/` | Painéis "antes / depois" (só com `--exemplos N`), prontos para as figuras do artigo. |

A etapa 11 trabalha com tabelas, não com imagens. Ela grava `metricas.csv`, `predicoes.csv`, curvas ROC, matrizes de confusão e os modelos treinados (veja a seção 5).

## 2. Instalação

Requer Python 3.9 ou superior. No PowerShell, a partir da pasta `ProjetoMelanoma`:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
```

Para conferir a instalação, sem processar nada:

```bash
python scripts/juncao.py --simular
```

## 3. Dados

Baixe os arquivos listados em [DADOS_ISIC.md](DADOS_ISIC.md) e extraia-os em `dados/isic2017/`. Para o primeiro teste bastam o zip de treino, as máscaras de treino e o CSV de rótulos de treino. Para a classificação ter números de validação e teste, os conjuntos de validação e teste também são necessários.

## 4. Ordem de execução

### Opção A: uma etapa por vez (recomendado na primeira execução)

```bash
python scripts/01_inventario.py --entrada dados/isic2017 --saida saidas/01_inventario --amostra 30 --exemplos 3
```

```bash
python scripts/02_redimensionamento.py --entrada saidas/01_inventario --saida saidas/02_redimensionamento --exemplos 6
```

```bash
python scripts/03_remocao_bordas.py --entrada saidas/02_redimensionamento --saida saidas/03_remocao_bordas --exemplos 6
```

```bash
python scripts/04_remocao_pelos.py --entrada saidas/03_remocao_bordas --saida saidas/04_remocao_pelos --exemplos 6
```

```bash
python scripts/05_constancia_cor.py --entrada saidas/04_remocao_pelos --saida saidas/05_constancia_cor --exemplos 6
```

```bash
python scripts/06_realce_contraste.py --entrada saidas/05_constancia_cor --saida saidas/06_realce_contraste --exemplos 6
```

```bash
python scripts/07_filtragem_fourier.py --entrada saidas/06_realce_contraste --saida saidas/07_filtragem_fourier --exemplos 6
```

```bash
python scripts/08_padronizacao.py --entrada saidas/07_filtragem_fourier --saida saidas/08_padronizacao --exemplos 6
```

```bash
python scripts/09_segmentacao.py --entrada saidas/08_padronizacao --saida saidas/09_segmentacao --exemplos 6
```

```bash
python scripts/10_descritores.py --entrada saidas/09_segmentacao --saida saidas/10_descritores --exemplos 6
```

```bash
python scripts/11_classificacao.py --entrada saidas/10_descritores --saida saidas/11_classificacao
```

Com a mesma `--semente` (padrão 42), todas as etapas escolhem **as mesmas imagens** de exemplo, então dá para acompanhar uma lesão do começo ao fim. Para ver todos os parâmetros de uma etapa:

```bash
python scripts/09_segmentacao.py --help
```

### Opção B: tudo de uma vez com `juncao.py`

Etapas 01 a 11:

```bash
python scripts/juncao.py --dados dados/isic2017 --saidas saidas --exemplos 6
```

Teste rápido (30 imagens por conjunto, numa pasta separada):

```bash
python scripts/juncao.py --dados dados/isic2017 --saidas saidas_teste --amostra 30 --exemplos 3
```

Só um intervalo de etapas (as saídas anteriores precisam existir):

```bash
python scripts/juncao.py --saidas saidas --de 9 --ate 11
```

Parâmetros específicos de uma etapa:

```bash
python scripts/juncao.py --dados dados/isic2017 --extra "4:--limiar 10 --pelos-claros" --extra "11:--modelos svm rf"
```

O `juncao.py` para na primeira etapa que falhar e mostra o comando para retomar a partir dela (`--de N`). As saídas das etapas anteriores são preservadas.

**Disco:** cada etapa de imagem grava PNGs de ~512 px (256 px a partir da 08). Para o ISIC 2017 completo (~2.750 imagens), reserve algo da ordem de 1 a 2 GB por etapa (estimativa, a confirmar no primeiro uso). `--formato jpg` reduz o espaço, mas tem perda de qualidade.

### Experimento extra para o artigo

**Tabela 6: segmentação com e sem pré-processamento.** Rode a etapa 09 sobre a saída da 02 (imagem só redimensionada) e compare o `metricas_segmentacao.csv` das duas pastas:

```bash
python scripts/09_segmentacao.py --entrada saidas/02_redimensionamento --saida saidas/09_segmentacao_sem_preproc --exemplos 6
```

## 5. Como validar cada etapa

Depois de cada etapa, abra `resumo.json` e `erros.csv` e olhe a pasta `exemplos/`.

| Etapa | O que conferir |
|---|---|
| 01 inventário | Em `resumo.json`, `contagem_por_conjunto_e_classe` deve somar ~2.000 (treino), ~150 (validação) e ~600 (teste). Confira também `imagens_com_mascara` (deve ser igual ao total) e `duplicatas_exatas` (a própria base ISIC 2017 tem 6 pares de imagens idênticas, 4 deles entre conjuntos diferentes). `erros.csv` deve estar vazio ou ter poucos casos. `distribuicao_classes.png` e `resolucoes.png` vão para a Seção 4 do artigo. |
| 02 redimensionamento | Nenhuma imagem com lado maior diferente de 512 (colunas `largura`/`altura` do manifesto). Nos exemplos, a lesão não pode aparecer distorcida. |
| 03 bordas | Nos exemplos, o contorno amarelo do FOV deve acompanhar a vinheta preta. Filtre `fov_suspeito = 1` no manifesto e olhe essas imagens uma a uma. `borda_detectada` informa quantas imagens tinham vinheta. |
| 04 pelos | A máscara do painel do meio deve marcar os pelos e **não** os pontos/glóbulos da lesão. Ordene o manifesto por `fracao_pelos` e olhe as maiores. Se sobrar pelo, diminua `--limiar`. Se estiver apagando estrutura da lesão, aumente `--alongamento-min`. |
| 05 cor | Imagens muito amareladas ou azuladas devem ficar com aspecto mais neutro. As colunas `iluminante_b/g/r` mostram o quanto cada imagem foi corrigida. |
| 06 contraste | `contraste_rms_depois` deve ser, em geral, maior que `contraste_rms_antes`. A cor da lesão não pode mudar de matiz. |
| 07 Fourier | No `*_espectro.png`, as altas frequências (bordas do espectro) ficam atenuadas. A imagem fica suave, mas a borda da lesão continua visível. `fourier_energia_removida` deve ficar pequena (poucos %). |
| 08 padronização | Todas as imagens com 256 × 256. `normalizacao.json` traz média e desvio do **treino**. `treino.csv`, `validacao.csv` e `teste.csv` têm as contagens da etapa 01. |
| 09 segmentação | `metricas_segmentacao.csv` traz IoU e Dice por conjunto. Nos exemplos, o contorno verde (previsto) deve ficar próximo do vermelho (ISIC). Olhe as linhas com `segmentacao_vazia = 1` ou `segmentacao_suspeita = 1` e as de menor `iou`. Se a lesão vier "vazada", aumente `--fechamento`; se vier com pele grudada, teste `--canal azul`. |
| 10 descritores | `descritores.csv` deve ter uma linha por imagem da etapa 09, menos as que estão em `erros.csv` (segmentações vazias). Confira o `dicionario_descritores.csv`. Valores plausíveis: `compacidade ≥ 1`, `solidez` entre 0 e 1, `assimetria_*` entre 0 e 1. |
| 11 classificação | `metricas.csv`, linhas com `limiar_tipo = escolhido`: a sensibilidade na validação/teste deve ficar perto da `--sensibilidade-alvo` (padrão 0,85). Todo modelo deve superar a linha "Referência trivial (dummy)", que tem AUC perto de 0,5. `roc_teste.png`, `matriz_confusao_*_teste.png` e `importancia_atributos_*.png` são as figuras da Seção 6.3. |

Para ajustar um parâmetro, rode de novo só aquela etapa e as seguintes, por exemplo `juncao.py --de 9`.

**Importante:** só olhe os números do **teste** depois de fixar todos os parâmetros com base no treino e na validação. Ajustar parâmetros olhando o teste invalida o resultado.

## 6. Ligação com o artigo

| Script | Seção do artigo | Conteúdo da disciplina |
|---|---|---|
| 01 | 4 Base de Dados | Fundamentos (aquisição e representação) |
| 02 | 5.1.1 | Fundamentos; Interpolação |
| 03 | 5.1.2 | Segmentação por limiar; Morfologia; Processamento em cores |
| 04 | 5.1.3 | Morfologia Matemática; Melhoria (mediana, inpainting); Cores (HSV) |
| 05 | 5.1.4 | Processamento de Imagens em Cores |
| 06 | 5.1.5 | Melhoria de Imagens (realce); Cores (CIELAB) |
| 07 | 5.1.6 | Transformada de Fourier (filtragem na frequência) |
| 08 | 5.1.7 | Transformações geométricas; Interpolação; Fundamentos (normalização) |
| 09 | 5.2 | Segmentação de Imagens (Otsu do zero); Morfologia (pós-processamento) |
| 10 | 5.3 | Descrição e Representação (ABCD, momentos de Hu); Fourier (descritores de contorno) |
| 11 | 5.4 e 6.3 | Classificação de Padrões |

## 7. Artigo Completo

Para mais detalhes sobre a fundamentação teórica, metodologia e resultados obtidos ao longo do desenvolvimento deste pipeline, consulte o documento oficial:
**[Pré-processamento de imagens dermatoscópicas para apoio à triagem de melanoma](https://docs.google.com/document/d/1TJdZjIw91cIfJIoU8lKhTwOxuV-7_KwQ0w1qkouw6Js/edit?usp=sharing)**
