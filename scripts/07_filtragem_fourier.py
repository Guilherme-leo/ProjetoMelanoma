# -*- coding: utf-8 -*-
"""
ETAPA 07 - FILTRAGEM PASSA-BAIXA NO DOMINIO DA FREQUENCIA (FOURIER)   [Artigo: Secao 5.1.6]

O que faz (implementado "do zero" com numpy.fft, sem filtro pronto do OpenCV)
  1. Converte para CIELAB e separa o canal L* (luminosidade).
  2. Faz padding por reflexao (evita o artefato de "borda periodica" da DFT).
  3. Calcula a DFT 2D: F(u,v) = FFT2{ L(x,y) }.
  4. Multiplica por um filtro passa-baixa H(u,v), com frequencia normalizada
     D(u,v) = sqrt(u^2 + v^2) em ciclos/pixel:
        gaussiano   : H = exp( -D^2 / (2*D0^2) )            (padrao: sem "ringing")
        butterworth : H = 1 / (1 + (D/D0)^(2n))
        ideal       : H = 1 se D <= D0, senao 0            (so para comparacao no artigo:
                                                               gera oscilacoes de Gibbs)
  5. Volta ao dominio espacial pela IFFT, recorta o padding e recombina com a*, b*.

Justificativa (topico: Transformada de Fourier)
  A pele tem textura fina (poros, sulcos, ruido de sensor) que aparece como
  energia em ALTAS frequencias. Para a segmentacao por limiar interessa a
  forma "grossa" da lesao (baixas frequencias); atenuar as altas frequencias
  deixa o histograma mais bimodal e as bordas do limiar menos serrilhadas.
  O filtro gaussiano foi escolhido como padrao porque sua transformada inversa
  tambem e gaussiana, logo nao cria aneis (ringing) ao redor da lesao.

Importante
  Esta imagem e a ENTRADA DA SEGMENTACAO (proxima fase do projeto). Para
  visualizacao pelo medico, use a saida da etapa 06 (nao suavizada).
  Com --exemplos, salva tambem o espectro de magnitude antes/depois (figura do artigo).

Uso
  python scripts/07_filtragem_fourier.py --entrada saidas/06_realce_contraste --saida saidas/07_filtragem_fourier --exemplos 6
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum  # noqa: E402


def filtro_passa_baixa(forma: Tuple[int, int], d0: float, tipo: str, ordem: int) -> np.ndarray:
    """H(u,v) na MESMA ordenacao de numpy.fft.fft2 (frequencia zero no canto)."""
    h, w = forma
    u = np.fft.fftfreq(h)[:, None]   # ciclos/pixel, de -0.5 a 0.5
    v = np.fft.fftfreq(w)[None, :]
    D = np.sqrt(u ** 2 + v ** 2)
    if tipo == "gaussiano":
        return np.exp(-(D ** 2) / (2.0 * d0 ** 2))
    if tipo == "butterworth":
        return 1.0 / (1.0 + (D / d0) ** (2 * ordem))
    if tipo == "ideal":
        return (D <= d0).astype(np.float64)
    raise ValueError(f"tipo de filtro desconhecido: {tipo}")


def espectro_visual(F: np.ndarray) -> np.ndarray:
    """log(1 + |F|) centralizado e normalizado para 0-255 (apenas para figura)."""
    mag = np.log1p(np.abs(np.fft.fftshift(F)))
    mag = (mag - mag.min()) / max(mag.max() - mag.min(), 1e-12)
    return (mag * 255).astype(np.uint8)


def filtrar_canal(canal: np.ndarray, d0: float, tipo: str, ordem: int, pad_frac: float):
    h, w = canal.shape
    pad = max(8, int(round(pad_frac * max(h, w))))
    ext = cv2.copyMakeBorder(canal, pad, pad, pad, pad, cv2.BORDER_REFLECT_101).astype(np.float64)
    F = np.fft.fft2(ext)
    H = filtro_passa_baixa(ext.shape, d0, tipo, ordem)
    G = F * H
    filtrado = np.real(np.fft.ifft2(G))[pad:pad + h, pad:pad + w]
    energia_removida = 1.0 - float(np.sum(np.abs(G) ** 2) / max(np.sum(np.abs(F) ** 2), 1e-12))
    return np.clip(filtrado, 0, 255).astype(np.uint8), F, G, energia_removida


def processar(linha: Dict, ctx: comum.Contexto) -> Dict:
    p = ctx.params
    iid = linha["image_id"]
    img = comum.ler_imagem(linha["imagem"])
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L_f, F, G, energia = filtrar_canal(lab[..., 0], p["d0"], p["tipo"], p["ordem"], p["pad"])
    if p["todos_canais"]:  # opcional: suaviza tambem a cromaticidade
        a_f = filtrar_canal(lab[..., 1], p["d0"], p["tipo"], p["ordem"], p["pad"])[0]
        b_f = filtrar_canal(lab[..., 2], p["d0"], p["tipo"], p["ordem"], p["pad"])[0]
    else:
        a_f, b_f = lab[..., 1], lab[..., 2]
    saida = cv2.cvtColor(cv2.merge([L_f, a_f, b_f]), cv2.COLOR_LAB2BGR)

    d = ctx.destino("imagens", iid)
    comum.salvar_imagem(d, saida)
    linha.update({"imagem": d, "fourier_energia_removida": round(energia, 6)})

    if iid in ctx.exemplos:
        comum.salvar_exemplo(ctx.saida, iid, [("antes", img), (f"{p['tipo']} D0={p['d0']:g}", saida)])
        comum.salvar_imagem(ctx.saida / "exemplos" / f"{iid}_espectro.png", comum.montar_painel([
            ("|F| antes (log)", espectro_visual(F)),
            ("filtro H", (np.fft.fftshift(filtro_passa_baixa(F.shape, p["d0"], p["tipo"], p["ordem"])) * 255).astype(np.uint8)),
            ("|F.H| depois (log)", espectro_visual(G)),
        ]))
    return linha


def main() -> int:
    p = comum.criar_parser("Etapa 07 - filtro passa-baixa no dominio de Fourier (canal L*).")
    p.add_argument("--tipo", choices=["gaussiano", "butterworth", "ideal"], default="gaussiano")
    p.add_argument("--d0", type=float, default=0.08,
                   help="Frequencia de corte em ciclos/pixel (0 < D0 <= 0.5). Menor = mais suave.")
    p.add_argument("--ordem", type=int, default=2, help="Ordem n do filtro Butterworth.")
    p.add_argument("--pad", type=float, default=0.10, help="Padding por reflexao (fracao do maior lado).")
    p.add_argument("--todos-canais", action="store_true", help="Filtra tambem a* e b* (padrao: so L*).")
    args = p.parse_args()
    if not (0 < args.d0 <= 0.5):
        p.error("--d0 deve estar em (0, 0.5]")
    params = {"tipo": args.tipo, "d0": args.d0, "ordem": args.ordem, "pad": args.pad,
              "todos_canais": args.todos_canais}
    return comum.executar_etapa("07_filtragem_fourier", args, processar, params,
                                "Passa-baixa no dominio da frequencia (DFT 2D) aplicado ao canal L*.")


if __name__ == "__main__":
    sys.exit(main())
