# -*- coding: utf-8 -*-
"""
ETAPA 03 - REMOCAO DE BORDAS ESCURAS / VINHETA DO DERMATOSCOPIO   [Artigo: Secao 5.1.2]

O que faz
  1. Marca como "escuro" todo pixel cujo MAIOR canal (B, G ou R) fica abaixo de
     --limiar-escuro (pretos quase puros, tipicos da vinheta circular e das
     faixas pretas da lente do dermatoscopio).
  2. Mantem apenas as regioes escuras CONECTADAS A BORDA da imagem e com area
     minima (--area-min). Uma lesao escura no centro nao toca a borda e, por
     isso, nao e removida.
  3. O complemento dessas regioes e o campo de visao (FOV, "field of view").
     A imagem, a mascara da lesao e o FOV sao recortados no retangulo envolvente
     do FOV; os cantos que sobram fora do FOV sao preenchidos com a mediana da
     cor da pele vizinha (ou por inpainting, com --preenchimento inpaint).
  4. O FOV e salvo em fov/ para que as etapas seguintes (constancia de cor,
     estatisticas) ignorem pixels que nao sao pele.

Justificativa (topicos: Segmentacao por limiarizacao + Morfologia + Componentes conexos)
  A vinheta preta e o artefato mais comum em imagens dermatoscopicas. Se nao for
  tratada, ela (a) distorce a estimativa do iluminante na etapa 05, (b) e
  confundida com lesao na limiarizacao global (Otsu) e (c) altera estatisticas
  de cor. O teste "toca a borda" evita apagar lesoes escuras centrais.

Limitacoes conhecidas (citar no artigo)
  * Reguas/escalas impressas e marcacoes de caneta NAO sao removidas aqui; tracos
    finos e escuros sao parcialmente atenuados pela etapa 04 (black-hat).
  * Uma lesao muito escura que ocupa a borda da foto pode ser confundida com
    vinheta. Se a area do FOV ficar abaixo de --fov-min, a imagem e mantida sem
    recorte e marcada com fov_suspeito=1 para inspecao manual.

Uso
  python scripts/03_remocao_bordas.py --entrada saidas/02_redimensionamento --saida saidas/03_remocao_bordas --exemplos 6
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum  # noqa: E402


def detectar_fov(img: np.ndarray, limiar: int, area_min: float) -> Tuple[np.ndarray, int]:
    """Devolve (fov uint8 0/255, numero de regioes de borda removidas)."""
    h, w = img.shape[:2]
    escuro = (img.max(axis=2) < limiar).astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    escuro = cv2.morphologyEx(escuro, cv2.MORPH_OPEN, k3)          # tira pontinhos isolados
    escuro = cv2.morphologyEx(escuro, cv2.MORPH_CLOSE, k3, iterations=2)

    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(escuro, connectivity=8)
    manter = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        toca_borda = x == 0 or y == 0 or (x + ww) >= w or (y + hh) >= h
        if toca_borda and area >= area_min * h * w:
            manter.append(i)
    if not manter:
        return np.full((h, w), 255, np.uint8), 0

    borda = np.isin(rotulos, manter).astype(np.uint8) * 255
    fov = cv2.bitwise_not(borda)
    # fica so com o maior componente do FOV (descarta "ilhas" claras dentro da vinheta)
    n2, rot2, st2, _ = cv2.connectedComponentsWithStats(fov, connectivity=8)
    if n2 > 1:
        maior = 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))
        fov = (rot2 == maior).astype(np.uint8) * 255
    # suaviza o contorno do FOV
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fov = cv2.morphologyEx(fov, cv2.MORPH_CLOSE, k7)
    return fov, len(manter)


def preencher_fora_do_fov(img: np.ndarray, fov: np.ndarray, modo: str) -> np.ndarray:
    fora = fov == 0
    if not fora.any():
        return img
    if modo == "inpaint":
        return cv2.inpaint(img, (fora.astype(np.uint8) * 255), 5, cv2.INPAINT_TELEA)
    # mediana da cor de um anel de pele logo dentro do limite do FOV
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    anel = (fov > 0) & (cv2.erode(fov, k) == 0)
    amostra = img[anel] if anel.sum() > 50 else img[fov > 0]
    cor = np.median(amostra, axis=0).astype(np.uint8)
    saida = img.copy()
    saida[fora] = cor
    return saida


def processar(linha: Dict, ctx: comum.Contexto) -> Dict:
    p = ctx.params
    iid = linha["image_id"]
    img = comum.ler_imagem(linha["imagem"])
    h, w = img.shape[:2]
    fov, n_regioes = detectar_fov(img, p["limiar_escuro"], p["area_min"])
    frac = float((fov > 0).mean())

    suspeito = 0
    if frac < p["fov_min"]:
        ctx.logger.warning(f"{iid}: FOV cobre so {frac:.0%} da imagem - recorte ignorado (fov_suspeito=1).")
        fov, n_regioes, suspeito = np.full((h, w), 255, np.uint8), 0, 1

    x, y, ww, hh = cv2.boundingRect(fov) if n_regioes else (0, 0, w, h)
    recorte = img[y:y + hh, x:x + ww]
    fov_r = fov[y:y + hh, x:x + ww]
    saida = preencher_fora_do_fov(recorte, fov_r, p["preenchimento"])

    d_img = ctx.destino("imagens", iid)
    comum.salvar_imagem(d_img, saida)
    d_fov = ctx.destino("fov", f"{iid}_fov", "png")
    comum.salvar_imagem(d_fov, fov_r)

    if linha.get("mascara"):
        m = comum.ler_mascara(linha["mascara"])[y:y + hh, x:x + ww]
        d_m = ctx.destino("mascaras", f"{iid}_segmentation", "png")
        comum.salvar_imagem(d_m, m)
        linha["mascara"] = d_m

    linha.update({
        "imagem": d_img, "fov": d_fov,
        "borda_detectada": int(n_regioes > 0), "regioes_borda": n_regioes,
        "fracao_fov": round(float((fov_r > 0).mean()), 4),
        "recorte_xywh": f"{x},{y},{ww},{hh}", "fov_suspeito": suspeito,
        "largura": saida.shape[1], "altura": saida.shape[0],
    })

    if iid in ctx.exemplos:
        comum.salvar_exemplo(ctx.saida, iid, [
            ("antes", img),
            ("FOV detectado", comum.sobrepor_contorno(img, fov, (0, 255, 255))),
            ("depois", saida),
        ])
    return linha


def main() -> int:
    p = comum.criar_parser("Etapa 03 - remocao de vinheta/bordas escuras e calculo do FOV.")
    p.add_argument("--limiar-escuro", type=int, default=30,
                   help="Pixel e 'escuro' se max(B,G,R) < limiar (0-255).")
    p.add_argument("--area-min", type=float, default=0.002,
                   help="Area minima (fracao da imagem) de uma regiao escura de borda.")
    p.add_argument("--fov-min", type=float, default=0.35,
                   help="Se o FOV ficar menor que esta fracao, a deteccao e descartada.")
    p.add_argument("--preenchimento", choices=["mediana", "inpaint"], default="mediana",
                   help="Como preencher os cantos fora do FOV apos o recorte.")
    args = p.parse_args()
    params = {"limiar_escuro": args.limiar_escuro, "area_min": args.area_min,
              "fov_min": args.fov_min, "preenchimento": args.preenchimento}
    return comum.executar_etapa("03_remocao_bordas", args, processar, params,
                                "Vinheta/bordas pretas: limiarizacao + componentes conexos tocando a borda.")


if __name__ == "__main__":
    sys.exit(main())
