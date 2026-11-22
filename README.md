# Pré-processamento de imagens dermatoscópicas para apoio à triagem de melanoma

Disciplina: Processamento de Imagens (UFRPE). Autores: Guilherme Leonardo Guimarães, Guilherme Salgueiro Silveira e Ronaldo de Araújo Ribeiro Filho.

> **Ferramenta de apoio, não de diagnóstico.** O sistema só prepara as imagens e extrai medidas objetivas para ajudar o dermatologista. Ele **não** dá diagnóstico e **não** substitui o exame clínico nem a biópsia.

Este repositório contém a fase atual do projeto: o **pipeline de pré-processamento**. Ele tem 8 etapas implementadas, que rodam em sequência. Cada etapa lê a saída da anterior e grava a sua própria saída em uma pasta numerada. As fases de segmentação, extração de descritores (ABCD) e classificação ainda serão implementadas.

## 1. Estrutura de pastas

```
ProjetoMelanoma/
├── README.md                  ← este arquivo
├── DADOS_ISIC.md              ← o que baixar do ISIC, licença e citação
├── requirements.txt
├── dados/
│   └── isic2017/              ← coloque aqui os arquivos baixados (ver DADOS_ISIC.md, seção 3)
├── scripts/
│   ├── comum.py               ← utilitários (leitura/escrita, log, manifesto, exemplos)
│   ├── 01_inventario.py       ← valida a base e gera o manifesto
│   ├── 02_redimensionamento.py
│   ├── 03_remocao_bordas.py   ← vinheta/bordas pretas + campo de visão (FOV)
│   ├── 04_remocao_pelos.py    ← black-hat linear + inpainting (e reflexos)
│   ├── 05_constancia_cor.py   ← Shades of Gray no FOV
│   ├── 06_realce_contraste.py ← CLAHE no canal L*
│   ├── 07_filtragem_fourier.py ← filtro passa-baixa
│   ├── 08_padronizacao.py     ← tamanho fixo e estatísticas do treino
│   └── juncao.py              ← executa o pipeline em sequência
└── saidas/                    ← criada na execução: 01_inventario/, 02_redimensionamento/, ...
```

Toda pasta de saída de etapa (`saidas/NN_nome/`) contém:

| Arquivo | Para que serve |
|---|---|
| `manifesto.csv` | Uma linha por imagem que deu certo: caminhos (relativos), rótulo, conjunto e medidas da etapa. **É a entrada da etapa seguinte.** |
| `imagens/` | Imagens processadas (PNG, sem perdas). Algumas etapas também gravam `mascaras/`, `fov/` e `pelos/`. |
| `erros.csv` | Imagens que falharam e o motivo. O lote **não** para por causa delas. |
| `etapa.log` | Log detalhado, com parâmetros, cada imagem processada e avisos. |
| `resumo.json` | Contagens, parâmetros usados, tempo total e tempo médio por imagem. |
| `exemplos/` | Painéis "antes / depois" (só com `--exemplos N`), prontos para as figuras do artigo. |

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

## 3. Dados

Baixe os arquivos listados em [DADOS_ISIC.md](DADOS_ISIC.md) e extraia-os em `dados/isic2017/`. Para o primeiro teste bastam o zip de treino, as máscaras de treino e o CSV de rótulos de treino.

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
python scripts/06_realce_contraste.py --entrada saidas/05_constancia_cor --saida saidas/06_realce_contraste --exemplos 6
python scripts/07_filtragem_fourier.py --entrada saidas/06_realce_contraste --saida saidas/07_filtragem_fourier --exemplos 6
python scripts/08_padronizacao.py --entrada saidas/07_filtragem_fourier --saida saidas/08_padronizacao --exemplos 6
```

### Opção B: pipeline completo

```bash
python scripts/juncao.py --dados dados/isic2017 --saidas saidas --exemplos 6
```

Use `--de 5 --ate 8` para retomar a partir da etapa 05 quando a saída da 04 já existir. Consulte `python scripts/juncao.py --help` para outras opções.

Com a mesma `--semente` (padrão 42) e os mesmos IDs de entrada, as etapas 02 a 08 escolhem as mesmas imagens de exemplo. O inventário escolhe exemplos por classe. Para ver todos os parâmetros de uma etapa:

```bash
python scripts/04_remocao_pelos.py --help
```

**Disco:** cada etapa grava PNGs de ~512 px. Para o ISIC 2017 completo (~2.750 imagens), reserve algo da ordem de 1 a 2 GB por etapa (estimativa, a confirmar no primeiro uso). `--formato jpg` reduz o espaço, mas tem perda de qualidade.

## 5. Como validar cada etapa

Depois de cada etapa, abra `resumo.json` e `erros.csv` e olhe a pasta `exemplos/`.

| Etapa | O que conferir |
|---|---|
| 01 inventário | Em `resumo.json`, `contagem_por_conjunto_e_classe` deve somar ~2.000 (treino), ~150 (validação) e ~600 (teste). Confira também `imagens_com_mascara` (deve ser igual ao total) e `duplicatas_exatas` (o esperado é uma lista vazia). `erros.csv` deve estar vazio ou ter poucos casos. `distribuicao_classes.png` e `resolucoes.png` vão para a Seção 4 do artigo. |
| 02 redimensionamento | Nenhuma imagem com lado maior diferente de 512 (colunas `largura`/`altura` do manifesto). Nos exemplos, a lesão não pode aparecer distorcida. |
| 03 bordas | Nos exemplos, o contorno amarelo do FOV deve acompanhar a vinheta preta. Filtre `fov_suspeito = 1` no manifesto e olhe essas imagens uma a uma. `borda_detectada` informa quantas imagens tinham vinheta (dado para o artigo). |
| 04 pelos | A máscara do painel do meio deve marcar os pelos e **não** os pontos/glóbulos da lesão. Ordene o manifesto por `fracao_pelos` e olhe as maiores. Se sobrar pelo, diminua `--limiar`. Se estiver apagando estrutura da lesão, aumente `--alongamento-min`. |
| 05 cor | Compare os painéis antes/depois, procurando correções excessivas e canais saturados. Ordene por `fracao_cor_valida_fov` e examine as imagens com `cor_poucos_pixels = 1`; sem pixels válidos, a imagem vai para `erros.csv`. Os campos `iluminante_b/g/r` registram a estimativa normalizada. A correção altera as cores e não garante fidelidade clínica. |
| 06 contraste | Compare os painéis e `contraste_rms_antes/depois`; aumento do contraste não demonstra melhora clínica. |
| 07 Fourier | Inspecione a suavização e os espectros; confira se bordas e estruturas relevantes continuam visíveis. |
| 08 padronização | Confira tamanho e alinhamento das máscaras; `normalizacao.json` deve conter estatísticas calculadas apenas sobre pixels do treino. |

Para executar os testes automatizados da etapa 05, com as dependências instaladas:

```bash
python -m unittest discover -s tests -v
```

## 6. Ligação com o artigo

| Script | Seção do artigo | Tópico da disciplina |
|---|---|---|
| 01 | 4 Base de Dados | Fundamentos (aquisição e representação) |
| 02 | 5.1.1 | Fundamentos / Interpolação |
| 03 | 5.1.2 | Segmentação por limiar + Morfologia + Componentes conexos |
| 04 | 5.1.3 | Morfologia Matemática + Melhoria (inpainting) |
| 05 | 5.1.4 | Processamento de Imagens em Cores |
| 06 | 5.1.5 | Melhoria de Imagens (realce) + Cores (CIELAB) |
| 07 | 5.1.6 | Transformada de Fourier (filtragem na frequência) |
| 08 | 5.1.7 | Transformações geométricas + Fundamentos (normalização) |

## 7. Artigo Completo

Para mais detalhes sobre a fundamentação teórica, metodologia e resultados obtidos ao longo do desenvolvimento deste pipeline, consulte o documento oficial:
**[Pré-processamento de imagens dermatoscópicas para apoio à triagem de melanoma](https://docs.google.com/document/d/1kNc_jokgjRy45pVe7nd7nU__ZfcJ1CHY3Quaag9uElo/edit?usp=sharing)**
