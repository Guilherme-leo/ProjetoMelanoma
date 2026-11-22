# -*- coding: utf-8 -*-
"""
ETAPA 09 - SEGMENTACAO DA LESAO (OTSU + POS-PROCESSAMENTO MORFOLOGICO)   [Artigo: Secao 5.2]

O que faz
  1. Extrai um canal de intensidade (padrao: L* do CIELAB; opcoes: azul, cinza).
  2. Calcula o limiar de Otsu (1979) IMPLEMENTADO DO ZERO, usando so pixels do FOV:
        k* = argmax_k  sigma_B^2(k) = (mu_T * w0(k) - mu(k))^2 / ( w0(k) * w1(k) )
     A lesao e a classe ESCURA (intensidade <= k*).
  3. Pos-processamento morfologico:
        abertura   -> remove ruido e restos de pelos;
        fechamento -> une partes da lesao separadas por regioes mais claras;
        preenchimento de buracos (flood fill a partir da borda);
        escolha de UM componente conexo: o de maior area ponderada pela
        proximidade do centro (a lesao costuma estar centralizada na foto).
  4. Se houver mascara de referencia do ISIC, calcula IoU (Jaccard), Dice e a
     sensibilidade por pixel, e no final grava as medias por conjunto.

Justificativa (topicos: Segmentacao + Morfologia Matematica)
  Otsu e o metodo classico de limiarizacao global e nao exige rotulos. Ele
  funciona bem quando o histograma e bimodal (lesao escura x pele clara), que e
  exatamente o que as etapas 03-07 procuram garantir: sem vinheta, sem pelos,
  cor corrigida, contraste realcado e textura suavizada.

Comparacao "com x sem pre-processamento" (Tabela 6 do artigo)
  Rode esta mesma etapa sobre a saida da etapa 02 (so redimensionada):
    python scripts/09_segmentacao.py --entrada saidas/02_redimensionamento --saida saidas/09_segmentacao_sem_preproc
  e compare metricas_segmentacao.csv das duas pastas.

Uso
  python scripts/09_segmentacao.py --entrada saidas/08_padronizacao --saida saidas/09_segmentacao --exemplos 6
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum  # noqa: E402


# --------------------------------------------------------------------------- Otsu "do zero"
def otsu(valores: np.ndarray) -> Tuple[int, float]:
    """Limiar de Otsu para valores uint8. Classe 0 = [0..k], classe 1 = [k+1..255]."""
    hist = np.bincount(valores.ravel().astype(np.int64), minlength=256)[:256].astype(np.float64)
    total = hist.sum()
    if total == 0:
        raise ValueError("nenhum pixel valido para calcular o limiar de Otsu")
    p = hist / total
    w0 = np.cumsum(p)                       # probabilidade da classe 0 ate k
    mu = np.cumsum(p * np.arange(256))      # momento de 1a ordem acumulado
    mu_t = mu[-1]                           # media global
    w1 = 1.0 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_t * w0 - mu) ** 2 / (w0 * w1)
    sigma_b2 = np.nan_to_num(sigma_b2, nan=0.0, posinf=0.0, neginf=0.0)
    k = int(np.argmax(sigma_b2))
    return k, float(sigma_b2[k])


def extrair_canal(img: np.ndarray, canal: str) -> np.ndarray:
    if canal == "L":
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    if canal == "azul":  # canal B do RGB: costuma ter bom contraste lesao/pele
        return img[..., 0].copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------- pos-processamento
def _impar(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1


def preencher_buracos(m: np.ndarray) -> np.ndarray:
    """Preenche regioes de fundo totalmente cercadas pela lesao."""
    h, w = m.shape
    ext = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    inund = ext.copy()
    mascara_ff = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(inund, mascara_ff, (0, 0), 255)      # pinta o fundo ligado a borda
    buracos = cv2.bitwise_not(inund)[1:-1, 1:-1]       # o que sobrou sem pintar = buracos
    return cv2.bitwise_or(m, buracos)


def escolher_componente(m: np.ndarray, area_min_frac: float) -> np.ndarray:
    """Fica com um unico componente: maior area ponderada pela proximidade do centro."""
    n, rot, st, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(m)
    h, w = m.shape
    centro = np.array([w / 2.0, h / 2.0])
    dmax = float(np.hypot(w / 2.0, h / 2.0))
    areas = st[1:, cv2.CC_STAT_AREA]
    minimo = max(area_min_frac * h * w, 0.1 * float(areas.max()))
    candidatos = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= minimo]
    if not candidatos:
        return np.zeros_like(m)

    def pontuacao(i: int) -> float:
        d = float(np.hypot(*(cent[i] - centro))) / dmax
        return float(st[i, cv2.CC_STAT_AREA]) * (1.0 - 0.7 * d)

    melhor = max(candidatos, key=pontuacao)
    return (rot == melhor).astype(np.uint8) * 255


def segmentar(img: np.ndarray, fov: np.ndarray, p: Dict) -> Tuple[int, np.ndarray, np.ndarray]:
    canal = extrair_canal(img, p["canal"])
    valido = fov > 0
    k, _ = otsu(canal[valido])
    bruta = ((canal <= k) & valido).astype(np.uint8) * 255

    lado = max(img.shape[:2])
    ka = _impar(p["abertura"] or round(0.02 * lado))
    kf = _impar(p["fechamento"] or round(0.035 * lado))
    m = cv2.morphologyEx(bruta, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ka, ka)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kf, kf)))
    m = preencher_buracos(m)
    m = cv2.bitwise_and(escolher_componente(m, p["area_min"]), fov)
    return k, bruta, m


# --------------------------------------------------------------------------- metricas
def iou_dice(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    a, b = pred > 0, ref > 0
    inter = float(np.logical_and(a, b).sum())
    uniao = float(np.logical_or(a, b).sum())
    soma = float(a.sum() + b.sum())
    return {
        "iou": inter / uniao if uniao else 1.0,
        "dice": 2.0 * inter / soma if soma else 1.0,
        "sens_pixel": inter / float(b.sum()) if b.any() else float("nan"),
    }


def processar(linha: Dict, ctx: comum.Contexto) -> Dict:
    p = ctx.params
    iid = linha["image_id"]
    img = comum.ler_imagem(linha["imagem"])
    h, w = img.shape[:2]
    fov = comum.mascara_ou_cheia(linha.get("fov"), (h, w))
    k, bruta, pred = segmentar(img, fov, p)

    d = ctx.destino("mascaras_pred", f"{iid}_pred", "png")
    comum.salvar_imagem(d, pred)
    area_fov = max(1, int((fov > 0).sum()))
    frac = float((pred > 0).sum()) / area_fov
    linha.update({
        "mascara_pred": d, "limiar_otsu": k,
        "fracao_lesao_fov": round(frac, 5),
        "segmentacao_vazia": int(not pred.any()),
        # lesao ocupando quase todo o FOV costuma indicar classe invertida ou falha
        "segmentacao_suspeita": int(frac > p["fracao_suspeita"]),
        "iou": "", "dice": "", "sens_pixel": "",
    })
    if not pred.any():
        ctx.logger.warning(f"{iid}: segmentacao vazia (sera contada com IoU = 0).")

    ref = None
    if linha.get("mascara"):
        ref = comum.ler_mascara(linha["mascara"])
        if ref.shape != pred.shape:
            ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_NEAREST)
        met = iou_dice(pred, ref)
        linha.update({k2: round(v, 5) for k2, v in met.items()})

    if iid in ctx.exemplos:
        sobre = comum.sobrepor_contorno(img, pred, (0, 255, 0))
        if ref is not None:
            sobre = comum.sobrepor_contorno(sobre, ref, (0, 0, 255))
        comum.salvar_exemplo(ctx.saida, iid, [
            ("entrada", img), (f"Otsu k={k}", bruta), ("pos-processada", pred),
            ("verde=pred vermelho=ISIC" if ref is not None else "contorno previsto", sobre),
        ])
    return linha


def resumir_metricas(saida: Path, logger_nome: str) -> None:
    """Media, desvio e mediana de IoU/Dice por conjunto (Tabela 6 do artigo)."""
    linhas, _ = comum.ler_manifesto(saida)
    grupos = {}
    for l in linhas:
        if l.get("iou", "") == "":
            continue
        for g in (l.get("split", "") or "sem_split", "todos"):
            grupos.setdefault(g, []).append(l)
    tabela = []
    for g, itens in grupos.items():
        iou = np.array([float(l["iou"]) for l in itens])
        dice = np.array([float(l["dice"]) for l in itens])
        tabela.append({
            "conjunto": g, "n": len(itens),
            "iou_media": round(float(iou.mean()), 4), "iou_desvio": round(float(iou.std(ddof=1)) if len(iou) > 1 else 0.0, 4),
            "iou_mediana": round(float(np.median(iou)), 4),
            "dice_media": round(float(dice.mean()), 4), "dice_desvio": round(float(dice.std(ddof=1)) if len(dice) > 1 else 0.0, 4),
            "dice_mediana": round(float(np.median(dice)), 4),
            "vazias": sum(int(l.get("segmentacao_vazia", 0) or 0) for l in itens),
            "suspeitas": sum(int(l.get("segmentacao_suspeita", 0) or 0) for l in itens),
        })
    comum.escrever_csv(saida / "metricas_segmentacao.csv", tabela)
    comum.salvar_json(saida / "metricas_segmentacao.json", {"por_conjunto": tabela})
    for t in tabela:
        print(f"[{t['conjunto']}] n={t['n']}  IoU={t['iou_media']:.3f}±{t['iou_desvio']:.3f}  "
              f"Dice={t['dice_media']:.3f}±{t['dice_desvio']:.3f}  vazias={t['vazias']}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        todos = [float(l["iou"]) for l in linhas if l.get("iou", "") != ""]
        if todos:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(todos, bins=20, range=(0, 1), edgecolor="black")
            ax.set_xlabel("IoU (Jaccard)")
            ax.set_ylabel("Numero de imagens")
            ax.set_title("Distribuicao do IoU da segmentacao")
            fig.tight_layout()
            fig.savefig(saida / "histograma_iou.png", dpi=200)
            plt.close(fig)
    except ImportError:
        pass


def main() -> int:
    p = comum.criar_parser("Etapa 09 - segmentacao da lesao por Otsu + pos-processamento morfologico.")
    p.add_argument("--canal", choices=["L", "azul", "cinza"], default="L", help="Canal usado na limiarizacao.")
    p.add_argument("--abertura", type=int, default=0, help="Kernel da abertura (px). 0 = automatico (2%% do lado).")
    p.add_argument("--fechamento", type=int, default=0, help="Kernel do fechamento (px). 0 = automatico (3,5%% do lado).")
    p.add_argument("--area-min", type=float, default=0.005, help="Area minima de um componente (fracao da imagem).")
    p.add_argument("--fracao-suspeita", type=float, default=0.7,
                   help="Marca como suspeita a segmentacao que ocupa mais que esta fracao do FOV.")
    args = p.parse_args()
    params = {"canal": args.canal, "abertura": args.abertura, "fechamento": args.fechamento,
              "area_min": args.area_min, "fracao_suspeita": args.fracao_suspeita}
    codigo = comum.executar_etapa("09_segmentacao", args, processar, params,
                                  "Otsu (implementado do zero) + abertura, fechamento, preenchimento e componente central.")
    resumir_metricas(Path(args.saida), "09_segmentacao")
    return codigo


if __name__ == "__main__":
    sys.exit(main())
