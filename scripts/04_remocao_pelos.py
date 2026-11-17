# -*- coding: utf-8 -*-
"""
ETAPA 04 - REMOCAO DE PELOS E DE REFLEXOS ESPECULARES          [Artigo: Secao 5.1.3]

O que faz (inspirado no DullRazor, Lee et al., 1997)
  PELOS
  1. Converte para tons de cinza e aplica um filtro de mediana 3x3 (ruido).
  2. Aplica a transformacao BLACK-HAT (fechamento - imagem) com elementos
     estruturantes LINEARES em --orientacoes direcoes (padrao 12, de 0 a 165 graus)
     e guarda a resposta maxima. O black-hat realca estruturas escuras e mais
     finas que o elemento estruturante: exatamente o caso dos pelos.
  3. Limiariza a resposta (--limiar) e filtra os componentes conexos: so ficam
     os ALONGADOS (razao comprimento/largura >= --alongamento-min) e com area
     minima. Isso protege pontos e glóbulos da lesao, que sao estruturas
     dermatoscopicas relevantes para o diagnostico.
  4. Dilata levemente a mascara (cobre o "halo" do pelo).
  REFLEXOS (pode desligar com --sem-reflexos)
  5. Pixels muito claros e pouco saturados (V alto e S baixo em HSV), em
     manchas pequenas, sao tratados como reflexo de luz/bolha de gel.
  RECONSTRUCAO
  6. Pelos + reflexos sao preenchidos por inpainting (Telea, 2004), que propaga
     a textura vizinha para dentro da regiao mascarada.

Justificativa (topicos: Morfologia Matematica + Melhoria de imagens)
  Pelos sobre a lesao criam bordas falsas que atrapalham a segmentacao e as
  medidas de forma (perimetro, irregularidade). O elemento estruturante linear
  em varias orientacoes responde a estruturas finas e compridas e nao a pontos,
  o que reduz a remocao indevida de estruturas da propria lesao.

Saidas extras
  pelos/<id>_pelos.png   mascara do que foi removido (figura do artigo)
  colunas fracao_pelos e fracao_reflexos no manifesto (permite separar imagens
  "com muitos pelos" na analise de resultados).

Uso
  python scripts/04_remocao_pelos.py --entrada saidas/03_remocao_bordas --saida saidas/04_remocao_pelos --exemplos 6
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum  # noqa: E402


def kernels_lineares(comprimento: int, orientacoes: int) -> List[np.ndarray]:
    """Elementos estruturantes em forma de segmento de reta, centrados, em varias direcoes."""
    comprimento = comprimento if comprimento % 2 == 1 else comprimento + 1
    c = comprimento // 2
    kernels = []
    for i in range(orientacoes):
        ang = np.deg2rad(180.0 * i / orientacoes)
        dx, dy = c * np.cos(ang), c * np.sin(ang)
        k = np.zeros((comprimento, comprimento), np.uint8)
        cv2.line(k, (int(round(c - dx)), int(round(c - dy))),
                 (int(round(c + dx)), int(round(c + dy))), 1, 1)
        kernels.append(k)
    return kernels


def mascara_pelos(cinza: np.ndarray, kernels: List[np.ndarray], limiar: int,
                  alongamento_min: float, area_min: int, claros: bool) -> np.ndarray:
    base = cv2.medianBlur(cinza, 3)
    resposta = np.zeros_like(base)
    for k in kernels:
        resposta = np.maximum(resposta, cv2.morphologyEx(base, cv2.MORPH_BLACKHAT, k))
        if claros:  # pelos claros/brancos: top-hat (imagem - abertura)
            resposta = np.maximum(resposta, cv2.morphologyEx(base, cv2.MORPH_TOPHAT, k))
    _, bruta = cv2.threshold(resposta, limiar, 255, cv2.THRESH_BINARY)

    n, rot, stats, _ = cv2.connectedComponentsWithStats(bruta, connectivity=8)
    manter = np.zeros(n, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < area_min:
            continue
        x, y, ww, hh = stats[i, :4]
        ys, xs = np.nonzero(rot[y:y + hh, x:x + ww] == i)  # so dentro da caixa do componente
        (_, _), (a, b), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        longo, curto = max(a, b), max(min(a, b), 1.0)
        if longo / curto >= alongamento_min:
            manter[i] = True
    mascara = manter[rot].astype(np.uint8) * 255
    return cv2.dilate(mascara, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)


def mascara_reflexos(img: np.ndarray, v_min: int, s_max: int, area_max: float) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bruta = ((hsv[..., 2] >= v_min) & (hsv[..., 1] <= s_max)).astype(np.uint8) * 255
    n, rot, stats, _ = cv2.connectedComponentsWithStats(bruta, connectivity=8)
    lim = area_max * img.shape[0] * img.shape[1]
    manter = np.zeros(n, bool)
    manter[1:] = stats[1:, cv2.CC_STAT_AREA] <= lim  # reflexo = mancha pequena
    m = manter[rot].astype(np.uint8) * 255
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)


def processar(linha: Dict, ctx: comum.Contexto) -> Dict:
    p = ctx.params
    iid = linha["image_id"]
    img = comum.ler_imagem(linha["imagem"])
    h, w = img.shape[:2]
    fov = comum.mascara_ou_cheia(linha.get("fov"), (h, w))

    # Comprimento do elemento estruturante proporcional ao tamanho da imagem.
    comp = p["comprimento_kernel"] or max(9, int(round(0.03 * max(h, w))))
    kernels = kernels_lineares(comp, p["orientacoes"])
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m_pelos = mascara_pelos(cinza, kernels, p["limiar"], p["alongamento_min"],
                            p["area_min"], p["pelos_claros"])
    m_pelos = cv2.bitwise_and(m_pelos, fov)

    m_refl = np.zeros((h, w), np.uint8)
    if p["reflexos"]:
        m_refl = cv2.bitwise_and(mascara_reflexos(img, p["v_min"], p["s_max"], p["area_max_reflexo"]), fov)

    m_total = cv2.bitwise_or(m_pelos, m_refl)
    saida = cv2.inpaint(img, m_total, p["raio_inpaint"], cv2.INPAINT_TELEA) if m_total.any() else img.copy()

    d_img = ctx.destino("imagens", iid)
    comum.salvar_imagem(d_img, saida)
    d_pel = ctx.destino("pelos", f"{iid}_pelos", "png")
    comum.salvar_imagem(d_pel, m_total)

    area_fov = max(1, int((fov > 0).sum()))
    linha.update({
        "imagem": d_img, "pelos": d_pel,
        "kernel_pelos_px": comp,
        "fracao_pelos": round(float((m_pelos > 0).sum()) / area_fov, 5),
        "fracao_reflexos": round(float((m_refl > 0).sum()) / area_fov, 5),
    })
    if iid in ctx.exemplos:
        comum.salvar_exemplo(ctx.saida, iid, [("antes", img), ("pelos + reflexos", m_total), ("depois", saida)])
    return linha


def main() -> int:
    p = comum.criar_parser("Etapa 04 - remocao de pelos (black-hat linear + inpainting) e reflexos.")
    p.add_argument("--comprimento-kernel", type=int, default=0,
                   help="Comprimento (px) do elemento linear. 0 = automatico (3%% do maior lado, min. 9).")
    p.add_argument("--orientacoes", type=int, default=12, help="Numero de orientacoes do elemento linear.")
    p.add_argument("--limiar", type=int, default=12, help="Limiar na resposta black-hat (0-255).")
    p.add_argument("--alongamento-min", type=float, default=3.0,
                   help="Razao comprimento/largura minima para um componente ser considerado pelo.")
    p.add_argument("--area-min", type=int, default=30, help="Area minima (px) de um componente de pelo.")
    p.add_argument("--raio-inpaint", type=int, default=5, help="Raio do inpainting de Telea (px).")
    p.add_argument("--pelos-claros", action="store_true", help="Tambem detecta pelos claros (top-hat).")
    p.add_argument("--sem-reflexos", action="store_true", help="Desliga a remocao de reflexos especulares.")
    p.add_argument("--v-min", type=int, default=235, help="Reflexo: brilho minimo (V, 0-255).")
    p.add_argument("--s-max", type=int, default=40, help="Reflexo: saturacao maxima (S, 0-255).")
    p.add_argument("--area-max-reflexo", type=float, default=0.01,
                   help="Reflexo: area maxima de cada mancha (fracao da imagem).")
    args = p.parse_args()
    params = {
        "comprimento_kernel": args.comprimento_kernel, "orientacoes": args.orientacoes,
        "limiar": args.limiar, "alongamento_min": args.alongamento_min, "area_min": args.area_min,
        "raio_inpaint": args.raio_inpaint, "pelos_claros": args.pelos_claros,
        "reflexos": not args.sem_reflexos, "v_min": args.v_min, "s_max": args.s_max,
        "area_max_reflexo": args.area_max_reflexo,
    }
    return comum.executar_etapa("04_remocao_pelos", args, processar, params,
                                "Pelos: black-hat com elementos lineares + filtro de alongamento + inpainting Telea.")


if __name__ == "__main__":
    sys.exit(main())
