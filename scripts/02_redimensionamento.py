# -*- coding: utf-8 -*-
"""
ETAPA 02 - REDIMENSIONAMENTO COM PRESERVACAO DA PROPORCAO      [Artigo: Secao 5.1.1]

O que faz
  Reduz cada imagem para que o MAIOR lado tenha --lado-maior pixels (padrao 512),
  mantendo a proporcao largura/altura. A mascara de segmentacao, quando existe,
  recebe exatamente a mesma transformacao.

Justificativa (topico: Fundamentos / Interpolacao)
  * As imagens do ISIC 2017 tem resolucoes muito diferentes entre si (a faixa
    exata sai no resumo.json da etapa 01). Trabalhar em escala unica torna os
    parametros das etapas
    seguintes (tamanho de kernel morfologico, raio de inpainting, frequencia de
    corte do filtro de Fourier) comparaveis entre imagens.
  * Reduz o custo computacional das etapas seguintes em ordens de grandeza.
  * Interpolacao INTER_AREA (media da area de pixels) na reducao evita aliasing;
    INTER_CUBIC so e usada se for preciso ampliar (raro).
  * Mascaras usam INTER_NEAREST para continuarem binarias (0/255).
  * A proporcao e preservada aqui; o tamanho quadrado final so e imposto na
    etapa 08, para nao distorcer a forma da lesao (assimetria e borda sao
    criterios clinicos da regra ABCD).

Uso
  python scripts/02_redimensionamento.py --entrada saidas/01_inventario --saida saidas/02_redimensionamento --exemplos 6
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum  # noqa: E402


def redimensionar(img, lado_maior: int, interp_mascara: bool = False):
    h, w = img.shape[:2]
    escala = lado_maior / float(max(h, w))
    if abs(escala - 1.0) < 1e-6:
        return img.copy(), 1.0
    nw, nh = max(1, round(w * escala)), max(1, round(h * escala))
    if interp_mascara:
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_AREA if escala < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (nw, nh), interpolation=interp), escala


def processar(linha: Dict, ctx: comum.Contexto) -> Dict:
    lado = ctx.params["lado_maior"]
    img = comum.ler_imagem(linha["imagem"])
    nova, escala = redimensionar(img, lado)
    destino = ctx.destino("imagens", linha["image_id"])
    comum.salvar_imagem(destino, nova)
    linha["imagem"] = destino
    linha["escala_redimensionamento"] = round(escala, 6)
    linha["largura"], linha["altura"] = nova.shape[1], nova.shape[0]

    if linha.get("mascara"):
        m = comum.ler_mascara(linha["mascara"])
        m2, _ = redimensionar(m, lado, interp_mascara=True)
        if m2.shape[:2] != nova.shape[:2]:  # garante alinhamento pixel a pixel
            m2 = cv2.resize(m2, (nova.shape[1], nova.shape[0]), interpolation=cv2.INTER_NEAREST)
        d_m = ctx.destino("mascaras", f"{linha['image_id']}_segmentation", "png")
        comum.salvar_imagem(d_m, m2)
        linha["mascara"] = d_m

    if linha["image_id"] in ctx.exemplos:
        comum.salvar_exemplo(ctx.saida, linha["image_id"], [
            (f"original {img.shape[1]}x{img.shape[0]}", img),
            (f"redimensionada {nova.shape[1]}x{nova.shape[0]}", nova),
        ])
    return linha


def main() -> int:
    p = comum.criar_parser("Etapa 02 - redimensionamento preservando a proporcao.")
    p.add_argument("--lado-maior", type=int, default=512, help="Tamanho (px) do maior lado apos o redimensionamento.")
    args = p.parse_args()
    if args.lado_maior < 64:
        p.error("--lado-maior deve ser >= 64")
    return comum.executar_etapa("02_redimensionamento", args, processar,
                                {"lado_maior": args.lado_maior},
                                "Redimensionamento do maior lado com INTER_AREA, proporcao preservada.")


if __name__ == "__main__":
    sys.exit(main())
