# -*- coding: utf-8 -*-
"""
ETAPA 08 - PADRONIZACAO GEOMETRICA E NORMALIZACAO              [Artigo: Secao 5.1.7]

O que faz
  1. Leva cada imagem a um tamanho quadrado fixo (--tamanho, padrao 256x256):
       pad      : reduz o maior lado para --tamanho e completa o outro lado com
                  a cor mediana da borda da imagem (padrao; NAO corta a lesao e
                  NAO distorce a forma);
       crop     : reduz o menor lado para --tamanho e corta o centro;
       esticar  : redimensiona direto para o quadrado (distorce a proporcao).
     Mascara da lesao e FOV recebem a mesma transformacao (area de padding = 0).
  2. Calcula media e desvio-padrao por canal (RGB, escala 0-1) usando APENAS o
     conjunto de TREINO e apenas pixels validos (FOV), e grava normalizacao.json.
     Esses valores serao usados para a normalizacao z-score  x' = (x - media)/desvio
     na entrada do classificador (inclusive a CNN opcional).
  3. Opcional (--salvar-npy): grava cada imagem ja normalizada em float32 (.npy).
  4. Grava listas separadas por conjunto: treino.csv, validacao.csv, teste.csv.

Justificativa (topicos: Transformacoes geometricas / Interpolacao + Fundamentos)
  * Classificadores (classicos sobre descritores ou CNNs) exigem entradas de
    tamanho fixo; o modo "pad" preserva a proporcao, importante para medidas de
    assimetria e de borda.
  * Usar estatisticas so do treino evita vazamento de informacao (data leakage)
    dos conjuntos de validacao e teste para o modelo.

Uso
  python scripts/08_padronizacao.py --entrada saidas/07_filtragem_fourier --saida saidas/08_padronizacao --exemplos 6
  (para a versao de visualizacao: --entrada saidas/06_realce_contraste --saida saidas/08_padronizacao_visual)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum  # noqa: E402


def padronizar(img: np.ndarray, tamanho: int, modo: str, interp: int, valor_pad) -> Tuple[np.ndarray, Tuple]:
    h, w = img.shape[:2]
    if modo == "esticar":
        return cv2.resize(img, (tamanho, tamanho), interpolation=interp), (0, 0, 0, 0)
    if modo == "crop":
        e = tamanho / float(min(h, w))
        nw, nh = max(tamanho, round(w * e)), max(tamanho, round(h * e))
        r = cv2.resize(img, (nw, nh), interpolation=interp)
        x0, y0 = (nw - tamanho) // 2, (nh - tamanho) // 2
        return r[y0:y0 + tamanho, x0:x0 + tamanho], (-y0, -x0, 0, 0)
    # pad
    e = tamanho / float(max(h, w))
    nw, nh = max(1, round(w * e)), max(1, round(h * e))
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    top, left = (tamanho - nh) // 2, (tamanho - nw) // 2
    bottom, right = tamanho - nh - top, tamanho - nw - left
    return cv2.copyMakeBorder(r, top, bottom, left, right, cv2.BORDER_CONSTANT, value=valor_pad), \
        (top, left, bottom, right)


def cor_mediana_borda(img: np.ndarray, faixa: int = 5) -> list:
    bordas = np.concatenate([img[:faixa].reshape(-1, 3), img[-faixa:].reshape(-1, 3),
                             img[:, :faixa].reshape(-1, 3), img[:, -faixa:].reshape(-1, 3)])
    return [int(v) for v in np.median(bordas, axis=0)]


def processar(linha: Dict, ctx: comum.Contexto) -> Dict:
    p = ctx.params
    iid = linha["image_id"]
    img = comum.ler_imagem(linha["imagem"])
    h, w = img.shape[:2]
    fov = comum.mascara_ou_cheia(linha.get("fov"), (h, w))
    interp_img = cv2.INTER_AREA if max(h, w) >= p["tamanho"] else cv2.INTER_CUBIC

    saida, pads = padronizar(img, p["tamanho"], p["modo"], interp_img, cor_mediana_borda(img))
    fov2, _ = padronizar(fov, p["tamanho"], p["modo"], cv2.INTER_NEAREST, 0)

    d = ctx.destino("imagens", iid)
    comum.salvar_imagem(d, saida)
    d_fov = ctx.destino("fov", f"{iid}_fov", "png")
    comum.salvar_imagem(d_fov, fov2)
    linha.update({"imagem": d, "fov": d_fov, "largura": p["tamanho"], "altura": p["tamanho"],
                  "padding_tlbr": ",".join(str(v) for v in pads)})

    if linha.get("mascara"):
        m = comum.ler_mascara(linha["mascara"])
        m2, _ = padronizar(m, p["tamanho"], p["modo"], cv2.INTER_NEAREST, 0)
        d_m = ctx.destino("mascaras", f"{iid}_segmentation", "png")
        comum.salvar_imagem(d_m, np.where(m2 > 127, 255, 0).astype(np.uint8))
        linha["mascara"] = d_m

    # Acumula estatisticas (RGB, 0-1) so do TREINO e so de pixels validos.
    if linha.get("split") == "treino":
        rgb = cv2.cvtColor(saida, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        pix = rgb[fov2 > 0]
        acc = p["_acumulador"]
        acc["n"] += pix.shape[0]
        acc["soma"] += pix.sum(axis=0)
        acc["soma2"] += (pix ** 2).sum(axis=0)

    if iid in ctx.exemplos:
        comum.salvar_exemplo(ctx.saida, iid, [("antes", img), (f"{p['modo']} {p['tamanho']}x{p['tamanho']}", saida)])
    return linha


def main() -> int:
    p = comum.criar_parser("Etapa 08 - tamanho fixo, estatisticas de normalizacao e divisao por conjunto.")
    p.add_argument("--tamanho", type=int, default=256, help="Lado (px) da imagem quadrada final.")
    p.add_argument("--modo", choices=["pad", "crop", "esticar"], default="pad")
    p.add_argument("--salvar-npy", action="store_true",
                   help="Grava tambem cada imagem normalizada (z-score, float32) em npy/. Ocupa bastante disco.")
    args = p.parse_args()

    # Chaves iniciadas por "_" sao internas: nao vao para o log nem para o resumo.json.
    acumulador = {"n": 0, "soma": np.zeros(3), "soma2": np.zeros(3)}
    params = {"tamanho": args.tamanho, "modo": args.modo, "salvar_npy": args.salvar_npy,
              "_acumulador": acumulador}
    codigo = comum.executar_etapa("08_padronizacao", args, processar, params,
                                  "Tamanho fixo (pad/crop/esticar) + media/desvio do treino.")
    return finalizar(Path(args.saida), acumulador, args, codigo)


def finalizar(saida: Path, acc: Dict, args, codigo: int) -> int:
    linhas, campos = comum.ler_manifesto(saida)
    if acc["n"] > 0:
        media = acc["soma"] / acc["n"]
        desvio = np.sqrt(np.maximum(acc["soma2"] / acc["n"] - media ** 2, 1e-12))
        norm = {"espaco": "RGB", "escala": "0-1", "calculado_em": "treino (pixels do FOV)",
                "n_pixels": int(acc["n"]), "media": [round(float(v), 6) for v in media],
                "desvio": [round(float(v), 6) for v in desvio]}
    else:
        norm = {"aviso": "nenhuma imagem do conjunto 'treino' processada; estatisticas nao calculadas"}
    comum.salvar_json(saida / "normalizacao.json", norm)
    print(f"normalizacao.json: {norm}")

    for split in sorted({l.get("split", "") for l in linhas}):
        comum.escrever_csv(saida / f"{split or 'sem_split'}.csv",
                           [l for l in linhas if l.get("split", "") == split], campos)

    if args.salvar_npy and "media" in norm:
        media, desvio = np.array(norm["media"]), np.array(norm["desvio"])
        for l in comum.tqdm(linhas, desc="08_padronizacao[npy]", unit="img"):
            try:
                img = comum.ler_imagem(comum.resolver_caminho(l["imagem"], saida))
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                z = ((rgb - media) / desvio).astype(np.float32)
                destino = saida / "npy" / f"{l['image_id']}.npy"
                destino.parent.mkdir(parents=True, exist_ok=True)
                np.save(destino, z)
            except Exception as exc:  # um arquivo com problema nao interrompe os demais
                print(f"[AVISO] npy de {l['image_id']} nao gerado: {exc}")
    return codigo


if __name__ == "__main__":
    sys.exit(main())
