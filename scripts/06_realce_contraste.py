# -*- coding: utf-8 -*-
"""
ETAPA 06 - REALCE DE CONTRASTE (CLAHE NO CANAL L* DO CIELAB)    [Artigo: Secao 5.1.5]

O que faz
  1. Converte a imagem de BGR para CIELAB (L* = luminosidade; a*, b* = cromaticidade).
  2. Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization, Zuiderveld,
     1994) SOMENTE no canal L*, com limite de recorte --clip e grade --grade x --grade.
  3. Volta para BGR. Como a* e b* nao mudam, o matiz da lesao e preservado.

Justificativa (topicos: Melhoria de Imagens + Processamento em Cores)
  * Muitas lesoes tem borda difusa e baixo contraste com a pele; o realce local
    torna a transicao lesao/pele mais nitida, o que ajuda tanto a inspecao visual
    pelo medico quanto a segmentacao por limiar.
  * A equalizacao GLOBAL de histograma superexpoe regioes claras; a versao
    ADAPTATIVA trabalha por blocos, e o LIMITE de recorte impede que o ruido e a
    textura da pele sejam amplificados demais.
  * Aplicar no L* (e nao em R, G e B separadamente) evita criar cores falsas,
    o que seria grave porque cor e criterio diagnostico.

Observacao
  A imagem desta etapa e a versao "limpa" pensada para VISUALIZACAO pelo medico.
  A etapa 07 gera uma versao suavizada propria para a SEGMENTACAO.

Uso
  python scripts/06_realce_contraste.py --entrada saidas/05_constancia_cor --saida saidas/06_realce_contraste --exemplos 6
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum  # noqa: E402


def clahe_lab(img: np.ndarray, clip: float, grade: int) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L2 = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grade, grade)).apply(L)
    return cv2.cvtColor(cv2.merge([L2, a, b]), cv2.COLOR_LAB2BGR)


def contraste_rms(img: np.ndarray, valido: np.ndarray) -> float:
    """Desvio-padrao da luminancia (0-1) dentro do FOV: indicador simples de contraste."""
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float64) / 255.0
    return float(L[valido].std()) if valido.any() else float(L.std())


def processar(linha: Dict, ctx: comum.Contexto) -> Dict:
    p = ctx.params
    iid = linha["image_id"]
    img = comum.ler_imagem(linha["imagem"])
    fov = comum.mascara_ou_cheia(linha.get("fov"), img.shape[:2]) > 0
    saida = clahe_lab(img, p["clip"], p["grade"])

    d = ctx.destino("imagens", iid)
    comum.salvar_imagem(d, saida)
    linha.update({"imagem": d,
                  "contraste_rms_antes": round(contraste_rms(img, fov), 5),
                  "contraste_rms_depois": round(contraste_rms(saida, fov), 5)})
    if iid in ctx.exemplos:
        comum.salvar_exemplo(ctx.saida, iid, [("antes", img), (f"CLAHE L* clip={p['clip']:g}", saida)])
    return linha


def main() -> int:
    p = comum.criar_parser("Etapa 06 - realce de contraste com CLAHE no canal L* (CIELAB).")
    p.add_argument("--clip", type=float, default=2.0, help="Limite de recorte do CLAHE.")
    p.add_argument("--grade", type=int, default=8, help="Numero de blocos por lado da grade do CLAHE.")
    args = p.parse_args()
    return comum.executar_etapa("06_realce_contraste", args, processar,
                                {"clip": args.clip, "grade": args.grade},
                                "CLAHE aplicado apenas ao canal L* do espaco CIELAB.")


if __name__ == "__main__":
    sys.exit(main())
