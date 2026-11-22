# -*- coding: utf-8 -*-
"""
ETAPA 05 - CORRECAO DE COR / ILUMINACAO (CONSTANCIA DE COR "SHADES OF GRAY")  [Artigo: Secao 5.1.4]

O que faz
  1. Estima a cor do iluminante de cada imagem pela norma de Minkowski de ordem p
     (Finlayson e Trezzi, 2004):   e_c = ( media( I_c(x)^p ) )^(1/p),  c em {B,G,R}
     usando so pixels dentro do FOV e descartando pixels saturados (reflexos
     residuais) e quase pretos.
       p = 1  -> Gray World;   p -> infinito -> Max-RGB (White Patch);
       p = 6  -> valor recomendado por Barata et al. (2015) para dermatoscopia.
  2. Normaliza o vetor e = (e_B, e_G, e_R) e aplica o ganho diagonal de von Kries:
       I'_c = I_c / ( sqrt(3) * e_c / ||e|| )
     de modo que o iluminante estimado passa a ser branco neutro.

Justificativa (topico: Processamento de Imagens em Cores)
  As imagens do ISIC vem de varios dermatoscopios, cameras e clinicas, com
  temperaturas de cor diferentes. A mesma lesao pode parecer mais "amarela" ou
  mais "azulada" so por causa do equipamento. Como cor e um dos criterios da
  regra ABCD (e uma caracteristica usada pelo classificador), padronizar o
  iluminante reduz uma variacao que nao tem relacao com a doenca. Barata et al.
  (2015) mostraram ganho de desempenho na classificacao com essa correcao.
  A correcao e feita DEPOIS da remocao de bordas e de pelos para que a vinheta
  preta e os pelos nao enviesem a estimativa do iluminante.

Uso
  python scripts/05_constancia_cor.py --entrada saidas/04_remocao_pelos --saida saidas/05_constancia_cor --exemplos 6
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum  # noqa: E402


def shades_of_gray(img: np.ndarray, valido: np.ndarray, p: float) -> Tuple[np.ndarray, np.ndarray]:
    """Devolve (imagem corrigida uint8, iluminante normalizado [B,G,R])."""
    f = img.astype(np.float64) / 255.0
    pix = f[valido]
    if pix.shape[0] == 0:
        raise ValueError("nenhum pixel valido para estimar o iluminante")
    maximos = pix.max(axis=0)
    if np.any(maximos == 0):
        raise ValueError("canal sem intensidade valida para estimar o iluminante")
    if np.isposinf(p):
        e = maximos
    else:
        # Escala por canal para evitar underflow em expoentes grandes.
        e = maximos * np.power(np.mean(np.power(pix / maximos, p), axis=0), 1.0 / p)
    e = e / np.linalg.norm(e)
    ganho = 1.0 / (np.sqrt(3.0) * e)
    corrigida = np.clip(f * ganho[None, None, :], 0.0, 1.0)
    return (corrigida * 255.0 + 0.5).astype(np.uint8), e


def processar(linha: Dict, ctx: comum.Contexto) -> Dict:
    prm = ctx.params
    iid = linha["image_id"]
    img = comum.ler_imagem(linha["imagem"])
    fov = comum.mascara_ou_cheia(linha.get("fov"), img.shape[:2]) > 0
    maxc, minc = img.max(axis=2), img.min(axis=2)
    valido = fov & (maxc < prm["saturado"]) & (minc > prm["escuro"])
    n_fov = int(fov.sum())
    n_validos = int(valido.sum())
    saida, e = shades_of_gray(img, valido, float(prm["p"]))

    d = ctx.destino("imagens", iid)
    comum.salvar_imagem(d, saida)
    linha.update({"imagem": d, "iluminante_b": round(float(e[0]), 4),
                  "iluminante_g": round(float(e[1]), 4), "iluminante_r": round(float(e[2]), 4),
                  "pixels_cor_validos": n_validos,
                  "fracao_cor_valida_fov": round(n_validos / n_fov, 6) if n_fov else 0.0,
                  "cor_poucos_pixels": int(n_validos < 100)})
    if iid in ctx.exemplos:
        comum.salvar_exemplo(ctx.saida, iid, [("antes", img), (f"Shades of Gray p={float(prm['p']):g}", saida)])
    return linha


def main() -> int:
    p = comum.criar_parser("Etapa 05 - constancia de cor (Shades of Gray).")
    p.add_argument("--p", type=float, default=6.0,
                   help="Ordem da norma de Minkowski (1 = Gray World; 'inf' = Max-RGB).")
    p.add_argument("--saturado", type=int, default=250, help="Ignora pixels com algum canal >= este valor.")
    p.add_argument("--escuro", type=int, default=5, help="Ignora pixels com algum canal <= este valor.")
    args = p.parse_args()
    if not (args.p >= 1):
        p.error("--p deve ser >= 1")
    if not (0 <= args.escuro < args.saturado <= 255):
        p.error("exige 0 <= --escuro < --saturado <= 255")
    params = {"p": "inf" if np.isposinf(args.p) else args.p,
              "saturado": args.saturado, "escuro": args.escuro}
    return comum.executar_etapa("05_constancia_cor", args, processar, params,
                                "Shades of Gray (Minkowski p) + ganho diagonal de von Kries.")


if __name__ == "__main__":
    sys.exit(main())
