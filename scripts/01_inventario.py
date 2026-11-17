# -*- coding: utf-8 -*-
"""
ETAPA 01 - INVENTARIO E VALIDACAO DA BASE (ISIC 2017)       [Artigo: Secao 4 e 5.1]

O que faz
  1. Localiza, dentro de --entrada, as pastas/arquivos oficiais do ISIC 2017
     (Training, Validation e Test_v2): imagens, mascaras de segmentacao (Parte 1)
     e rotulos de diagnostico (Parte 3).
  2. Ignora os arquivos *_superpixels.png (usados so na Parte 2 do desafio).
  3. Abre cada imagem para confirmar que nao esta corrompida e registra a resolucao.
  4. Cruza imagem x rotulo x mascara e calcula um hash MD5 para achar duplicatas.
  5. Grava o manifesto.csv que todas as etapas seguintes usam.

Por que existe
  Antes de qualquer processamento e preciso saber exatamente QUANTAS imagens ha,
  de QUE classes, em QUE resolucoes e se algo esta faltando ou corrompido. Esses
  numeros alimentam a Secao 4 (Base de Dados) do artigo. Nenhuma imagem e alterada
  ou copiada aqui: o manifesto apenas aponta para os arquivos originais.

Rotulos
  O CSV da Parte 3 tem as colunas image_id, melanoma, seborrheic_keratosis.
  rotulo = "melanoma" | "ceratose_seborreica" | "nevo" (nenhuma das duas)
  alvo_melanoma = 1 para melanoma e 0 para as lesoes benignas (tarefa binaria).

Uso
  python scripts/01_inventario.py --entrada dados/isic2017 --saida saidas/01_inventario
  python scripts/01_inventario.py --entrada dados/isic2017 --saida saidas/01_inventario --amostra 200 --exemplos 4
"""
from __future__ import annotations

import csv
import hashlib
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum  # noqa: E402

# Prefixos oficiais dos arquivos do ISIC 2017 (nome do zip sem a extensao).
PREFIXOS_PADRAO = {
    "treino": "ISIC-2017_Training",
    "validacao": "ISIC-2017_Validation",
    "teste": "ISIC-2017_Test_v2",
}


def localizar(raiz: Path, nome: str, pasta: bool) -> Optional[Path]:
    """Procura 'nome' em qualquer nivel abaixo de 'raiz' (zips costumam criar pastas aninhadas)."""
    candidatos = [raiz / nome] + sorted(raiz.rglob(nome))
    for c in candidatos:
        if (pasta and c.is_dir()) or (not pasta and c.is_file()):
            return c
    return None


def ler_rotulos(csv_path: Path) -> Dict[str, Dict[str, float]]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        leitor = csv.DictReader(f)
        obrig = {"image_id", "melanoma", "seborrheic_keratosis"}
        faltando = obrig - set(leitor.fieldnames or [])
        if faltando:
            raise SystemExit(f"[ERRO] {csv_path.name} nao tem as colunas {sorted(faltando)}.")
        return {l["image_id"].strip(): {"melanoma": float(l["melanoma"]),
                                        "seborrheic_keratosis": float(l["seborrheic_keratosis"])}
                for l in leitor}


def ler_metadados(pasta_dados: Path) -> Dict[str, Dict[str, str]]:
    """Le, se existir, um CSV de metadados (idade/sexo) dentro da pasta de imagens."""
    meta: Dict[str, Dict[str, str]] = {}
    for arq in sorted(pasta_dados.glob("*metadata*.csv")):
        with open(arq, "r", encoding="utf-8-sig", newline="") as f:
            for l in csv.DictReader(f):
                iid = (l.pop("image_id", "") or "").strip()
                if iid:
                    meta.setdefault(iid, {}).update({f"meta_{k}": v for k, v in l.items()})
    return meta


def rotulo_texto(r: Dict[str, float]) -> str:
    if r["melanoma"] >= 0.5:
        return "melanoma"
    if r["seborrheic_keratosis"] >= 0.5:
        return "ceratose_seborreica"
    return "nevo"


def md5(arq: Path) -> str:
    h = hashlib.md5()
    with open(arq, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def amostrar_estratificado(linhas: List[Dict], n: int, semente: int) -> List[Dict]:
    """Amostra n imagens POR CONJUNTO mantendo a proporcao de classes (teste rapido)."""
    rng = random.Random(semente)
    saida: List[Dict] = []
    por_split = defaultdict(list)
    for l in linhas:
        por_split[l["split"]].append(l)
    for split, itens in por_split.items():
        if n >= len(itens):
            saida.extend(itens)
            continue
        por_classe = defaultdict(list)
        for l in itens:
            por_classe[l["rotulo"]].append(l)
        for classe, grupo in por_classe.items():
            k = max(1, round(n * len(grupo) / len(itens)))
            rng.shuffle(grupo)
            saida.extend(grupo[:k])
    return sorted(saida, key=lambda l: (l["split"], l["image_id"]))


def salvar_graficos(saida: Path, linhas: List[Dict], logger) -> None:
    """Graficos de distribuicao de classes e de resolucao (figuras da Secao 4)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib nao instalado: graficos da base nao foram gerados.")
        return
    splits = ["treino", "validacao", "teste"]
    classes = ["melanoma", "ceratose_seborreica", "nevo"]
    cont = Counter((l["split"], l["rotulo"]) for l in linhas)
    presentes = [s for s in splits if any(l["split"] == s for l in linhas)]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(presentes))
    for i, c in enumerate(classes):
        vals = [cont[(s, c)] for s in presentes]
        barras = ax.bar(x + (i - 1) * 0.27, vals, width=0.27, label=c.replace("_", " "))
        ax.bar_label(barras, fontsize=8)
    ax.set_xticks(x, presentes)
    ax.set_ylabel("Numero de imagens")
    ax.set_title("ISIC 2017 - distribuicao de classes por conjunto")
    ax.legend()
    fig.tight_layout()
    fig.savefig(saida / "distribuicao_classes.png", dpi=200)
    plt.close(fig)

    larg = [int(l["largura_original"]) for l in linhas]
    alt = [int(l["altura_original"]) for l in linhas]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(larg, alt, s=6, alpha=0.4)
    ax.set_xlabel("Largura (px)")
    ax.set_ylabel("Altura (px)")
    ax.set_title("Resolucao das imagens originais")
    fig.tight_layout()
    fig.savefig(saida / "resolucoes.png", dpi=200)
    plt.close(fig)


def main() -> int:
    p = comum.criar_parser("Etapa 01 - inventario e validacao da base ISIC 2017.")
    p.add_argument("--amostra", type=int, default=0,
                   help="Se > 0, mantem so N imagens por conjunto (estratificado por classe).")
    p.add_argument("--sem-hash", action="store_true", help="Nao calcula MD5 (mais rapido).")
    for split, pref in PREFIXOS_PADRAO.items():
        p.add_argument(f"--prefixo-{split}", default=pref, help=f"Prefixo dos arquivos do conjunto {split}.")
    args = p.parse_args()

    raiz, saida = Path(args.entrada), Path(args.saida)
    logger = comum.configurar_log(saida, "01_inventario", args.verboso)
    if not raiz.is_dir():
        logger.error(f"Pasta de dados nao encontrada: {raiz.resolve()}")
        return 1

    linhas: List[Dict] = []
    erros: List[Dict] = []
    t0 = time.perf_counter()
    for split in PREFIXOS_PADRAO:
        pref = getattr(args, f"prefixo_{split}")
        pasta_img = localizar(raiz, f"{pref}_Data", pasta=True)
        pasta_msk = localizar(raiz, f"{pref}_Part1_GroundTruth", pasta=True)
        csv_rot = localizar(raiz, f"{pref}_Part3_GroundTruth.csv", pasta=False)
        if pasta_img is None:
            logger.warning(f"[{split}] pasta '{pref}_Data' nao encontrada - conjunto ignorado.")
            continue
        if csv_rot is None:
            logger.warning(f"[{split}] '{pref}_Part3_GroundTruth.csv' nao encontrado - conjunto ignorado.")
            continue
        if pasta_msk is None:
            logger.warning(f"[{split}] mascaras '{pref}_Part1_GroundTruth' nao encontradas "
                           f"(segmentacao nao podera ser avaliada neste conjunto).")
        rotulos = ler_rotulos(csv_rot)
        meta = ler_metadados(pasta_img)
        arquivos = sorted(a for a in pasta_img.glob("ISIC_*.jpg") if "superpixels" not in a.name)
        logger.info(f"[{split}] {len(arquivos)} imagens | {len(rotulos)} rotulos | "
                     f"metadados: {'sim' if meta else 'nao'}")
        # (--amostra e --limite sao aplicados depois, quando os rotulos ja estao associados)
        for arq in comum.tqdm(arquivos, desc=f"01_inventario[{split}]", unit="img"):
            iid = arq.stem
            try:
                if iid not in rotulos:
                    raise KeyError("imagem sem rotulo no CSV da Parte 3")
                img = comum.ler_imagem(arq)
                h, w = img.shape[:2]
                msk = (pasta_msk / f"{iid}_segmentation.png") if pasta_msk else None
                if msk is not None and not msk.is_file():
                    logger.debug(f"{iid}: mascara ausente")
                    msk = None
                r = rotulos[iid]
                linha = {
                    "image_id": iid, "split": split,
                    "rotulo": rotulo_texto(r),
                    "alvo_melanoma": int(r["melanoma"] >= 0.5),
                    "melanoma": r["melanoma"], "seborrheic_keratosis": r["seborrheic_keratosis"],
                    "imagem": comum.caminho_relativo(arq, saida),
                    "mascara": comum.caminho_relativo(msk, saida) if msk else "",
                    "largura_original": w, "altura_original": h,
                    "tamanho_arquivo_kb": round(arq.stat().st_size / 1024, 1),
                    "md5": "" if args.sem_hash else md5(arq),
                }
                linha.update(meta.get(iid, {}))
                linhas.append(linha)
            except Exception as exc:
                erros.append({"image_id": iid, "etapa": "01_inventario", "erro": f"{type(exc).__name__}: {exc}"})
                logger.warning(f"ERRO {iid}: {exc}")
        faltam = set(rotulos) - {a.stem for a in arquivos}
        for iid in sorted(faltam):
            erros.append({"image_id": iid, "etapa": "01_inventario", "erro": "rotulo sem imagem correspondente"})

    if not linhas:
        logger.error("Nenhuma imagem valida encontrada. Confira a estrutura de pastas no README.")
        comum.escrever_csv(saida / "erros.csv", erros, ["image_id", "etapa", "erro"])
        return 1

    # Duplicatas exatas (mesmo conteudo binario) - importante para nao vazar treino no teste.
    duplicatas = []
    if not args.sem_hash:
        por_hash = defaultdict(list)
        for l in linhas:
            por_hash[l["md5"]].append(f"{l['split']}:{l['image_id']}")
        duplicatas = [v for v in por_hash.values() if len(v) > 1]
        for grupo in duplicatas:
            logger.warning(f"Duplicata exata: {grupo}")

    if args.amostra > 0:
        linhas = amostrar_estratificado(linhas, args.amostra, args.semente)
        logger.info(f"--amostra {args.amostra}: manifesto reduzido para {len(linhas)} imagens.")
    if args.limite > 0:
        linhas = linhas[: args.limite]

    comum.escrever_csv(saida / comum.NOME_MANIFESTO, linhas)
    comum.escrever_csv(saida / "erros.csv", erros, ["image_id", "etapa", "erro"])

    # Estatisticas para a Secao 4 do artigo.
    cont = Counter((l["split"], l["rotulo"]) for l in linhas)
    por_split = {s: {c: cont[(s, c)] for c in ("melanoma", "ceratose_seborreica", "nevo")}
                 for s in sorted({l["split"] for l in linhas})}
    larg = np.array([int(l["largura_original"]) for l in linhas])
    alt = np.array([int(l["altura_original"]) for l in linhas])
    resumo = {
        "etapa": "01_inventario",
        "entrada": str(raiz.resolve()),
        "imagens_ok": len(linhas),
        "imagens_com_erro": len(erros),
        "contagem_por_conjunto_e_classe": por_split,
        "imagens_com_mascara": sum(1 for l in linhas if l["mascara"]),
        "resolucao": {
            "largura_min": int(larg.min()), "largura_mediana": float(np.median(larg)), "largura_max": int(larg.max()),
            "altura_min": int(alt.min()), "altura_mediana": float(np.median(alt)), "altura_max": int(alt.max()),
            "resolucoes_distintas": len(set(zip(larg.tolist(), alt.tolist()))),
        },
        "duplicatas_exatas": duplicatas,
        "tempo_total_s": round(time.perf_counter() - t0, 2),
    }
    comum.salvar_json(saida / "resumo.json", resumo)
    salvar_graficos(saida, linhas, logger)

    # Exemplos: algumas imagens originais de cada classe (figura "exemplos da base").
    if args.exemplos > 0:
        for classe in ("melanoma", "ceratose_seborreica", "nevo"):
            ids = comum.escolher_exemplos([l["image_id"] for l in linhas if l["rotulo"] == classe],
                                          args.exemplos, args.semente)
            paineis = []
            for l in linhas:
                if l["image_id"] in ids:
                    caminho = comum.resolver_caminho(l["imagem"], saida)
                    paineis.append((l["image_id"], comum.ler_imagem(caminho)))
            if paineis:
                comum.salvar_imagem(saida / "exemplos" / f"exemplos_{classe}.png", comum.montar_painel(paineis, 260))

    logger.info(f"Inventario: {len(linhas)} imagens validas, {len(erros)} problemas. Por conjunto: {por_split}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
