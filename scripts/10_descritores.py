# -*- coding: utf-8 -*-
"""
ETAPA 10 - DESCRICAO E REPRESENTACAO: DESCRITORES DA LESAO      [Artigo: Secao 5.3]

Le a imagem e a mascara PREVISTA pela etapa 09 (nunca a mascara do ISIC: no uso
real nao existe mascara de especialista) e calcula, por imagem:

  A - ASSIMETRIA (regra ABCD)
      A mascara e centralizada e girada para alinhar o eixo principal (momentos
      de 2a ordem) com a horizontal; depois e refletida em torno de cada eixo:
        assimetria_eixo = area(M XOR reflexo(M)) / (2 * area(M))   (0 = simetrica)
  B - BORDA
      compacidade = P^2 / (4*pi*A)      (1 = circulo; maior = borda mais irregular)
      solidez     = A / area(fecho convexo)
      irregularidade_raio = desvio / media da distancia contorno-centroide
      nitidez_borda = gradiente medio de L* sobre o contorno (borda abrupta x difusa)
      excentricidade (autovalores da matriz de covariancia da forma)
  C - COR (CIELAB em unidades reais: L 0-100, a/b -128..127)
      media e desvio de L*, a*, b* dentro da lesao; contraste lesao x pele
      (Delta E 1976 entre a lesao e um anel de pele ao redor);
      n_cores_distintas: k-means (K=6) nos pixels da lesao, unindo grupos com
      Delta E < limiar e contando os que ocupam >= 5% da lesao.
  D - DIAMETRO / TAMANHO (em pixels: o ISIC nao informa a escala em mm)
      diametro equivalente, maior eixo (retangulo minimo), area relativa ao FOV.
  TEXTURA: entropia do histograma de L* e desvio-padrao local medio.
  REPRESENTACAO DA FORMA:
      7 momentos de Hu (invariantes a translacao, escala e rotacao; escala log);
      descritores de Fourier do contorno (contorno como sinal complexo
      z = x + jy, reamostrado em 64 pontos; |Z_k| normalizado por |Z_1| ->
      invariante a posicao, escala, rotacao e ponto inicial).

Saidas
  descritores.csv               uma linha por imagem (entrada da etapa 11)
  dicionario_descritores.csv    nome, criterio, descricao e unidade de cada atributo
  exemplos/                     contorno, eixos principais e fecho convexo desenhados

Justificativa (topico: Descricao e Representacao de Imagens + Fourier)
  Sao medidas objetivas e interpretaveis que traduzem os criterios que o
  dermatologista ja usa (ABCD). Elas podem ser mostradas ao medico junto da
  imagem, o que combina com o objetivo de APOIO (e nao de diagnostico).

Uso
  python scripts/10_descritores.py --entrada saidas/09_segmentacao --saida saidas/10_descritores --exemplos 6
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum  # noqa: E402

N_HU = 7
N_FOURIER = 10

# (nome, criterio, descricao, unidade) - vira o dicionario_descritores.csv
DICIONARIO: List[Tuple[str, str, str, str]] = [
    ("assimetria_eixo1", "A", "Area nao sobreposta ao refletir no eixo principal / (2*area)", "0-1"),
    ("assimetria_eixo2", "A", "Area nao sobreposta ao refletir no eixo secundario / (2*area)", "0-1"),
    ("assimetria_media", "A", "Media das duas assimetrias", "0-1"),
    ("assimetria_max", "A", "Maior das duas assimetrias", "0-1"),
    ("compacidade", "B", "P^2 / (4*pi*A); 1 = circulo", "adimensional"),
    ("solidez", "B", "Area / area do fecho convexo", "0-1"),
    ("irregularidade_raio", "B", "Desvio/media da distancia do contorno ao centroide", "adimensional"),
    ("nitidez_borda", "B", "Magnitude media do gradiente de L* sobre o contorno", "L*/px"),
    ("excentricidade", "B", "sqrt(1 - lambda2/lambda1) da covariancia da forma", "0-1"),
    ("L_media", "C", "Media de L* na lesao", "0-100"),
    ("L_desvio", "C", "Desvio-padrao de L* na lesao", "0-100"),
    ("a_media", "C", "Media de a* na lesao", "-128..127"),
    ("a_desvio", "C", "Desvio-padrao de a* na lesao", ""),
    ("b_media", "C", "Media de b* na lesao", "-128..127"),
    ("b_desvio", "C", "Desvio-padrao de b* na lesao", ""),
    ("deltaE_lesao_pele", "C", "Delta E 1976 entre a media da lesao e a media de um anel de pele", "Delta E"),
    ("n_cores_distintas", "C", "Grupos de cor (k-means, uniao por Delta E) com >= 5% da lesao", "contagem"),
    ("diametro_equivalente_px", "D", "Diametro do circulo de mesma area", "px"),
    ("maior_eixo_px", "D", "Maior lado do retangulo de area minima", "px"),
    ("area_relativa_fov", "D", "Area da lesao / area do FOV", "0-1"),
    ("entropia_L", "Textura", "Entropia (bits) do histograma de L* (32 classes) na lesao", "bits"),
    ("desvio_local_L", "Textura", "Media do desvio-padrao local (5x5) de L* na lesao", "L*"),
] + [(f"hu_{i + 1}", "Forma", f"Momento invariante de Hu {i + 1} (-sinal*log10|h|)", "")
     for i in range(N_HU)] \
  + [(f"fourier_{k}", "Forma", f"(|Z_{k}|+|Z_-{k}|)/(|Z_1|+|Z_-1|) do contorno reamostrado", "")
     for k in range(2, 2 + N_FOURIER)]
NOMES = [d[0] for d in DICIONARIO]


# --------------------------------------------------------------------------- forma
def momentos_forma(m: np.ndarray) -> Tuple[float, float, float, float, float]:
    """(cx, cy, theta, lambda1, lambda2) a partir dos momentos centrais da mascara."""
    M = cv2.moments(m, binaryImage=True)
    if M["m00"] <= 0:
        raise ValueError("mascara vazia")
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    mu20, mu02, mu11 = M["mu20"] / M["m00"], M["mu02"] / M["m00"], M["mu11"] / M["m00"]
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    cov = np.array([[mu20, mu11], [mu11, mu02]])
    l2, l1 = np.linalg.eigvalsh(cov)  # crescente
    return cx, cy, float(theta), float(max(l1, 1e-12)), float(max(l2, 0.0))


def alinhar(m: np.ndarray, cx: float, cy: float, theta: float) -> np.ndarray:
    """Centraliza a mascara em uma tela quadrada e gira para o eixo principal ficar horizontal."""
    h, w = m.shape
    lado = int(np.ceil(np.hypot(h, w))) + 4
    melhor, melhor_mu11 = None, None
    for sinal in (1.0, -1.0):  # testa os dois sentidos e fica com o que zera o mu11
        R = cv2.getRotationMatrix2D((cx, cy), sinal * np.degrees(theta), 1.0)
        R[0, 2] += lado / 2.0 - cx
        R[1, 2] += lado / 2.0 - cy
        r = cv2.warpAffine(m, R, (lado, lado), flags=cv2.INTER_NEAREST, borderValue=0)
        mm = cv2.moments(r, binaryImage=True)
        mu11 = abs(mm["mu11"]) / max(mm["m00"], 1e-12)
        if melhor is None or mu11 < melhor_mu11:
            melhor, melhor_mu11 = r, mu11
    return melhor


def assimetria(alinhada: np.ndarray) -> Tuple[float, float]:
    a = alinhada > 0
    area = float(a.sum())
    # recentraliza pelo centroide da mascara ja girada (corrige arredondamentos)
    ys, xs = np.nonzero(a)
    dy, dx = int(round(a.shape[0] / 2 - ys.mean())), int(round(a.shape[1] / 2 - xs.mean()))
    a = np.roll(np.roll(a, dy, axis=0), dx, axis=1)
    eixo1 = float(np.logical_xor(a, np.flipud(a)).sum()) / (2 * area)  # reflexo no eixo principal (horizontal)
    eixo2 = float(np.logical_xor(a, np.fliplr(a)).sum()) / (2 * area)  # reflexo no eixo secundario
    return eixo1, eixo2


def descritores_fourier(contorno: np.ndarray, n_pontos: int = 64, n_coef: int = N_FOURIER) -> List[float]:
    pts = contorno.reshape(-1, 2).astype(np.float64)
    if len(pts) >= 2:  # remove pontos repetidos consecutivos
        manter = np.any(np.diff(pts, axis=0, prepend=pts[-1:]) != 0, axis=1)
        pts = pts[manter] if manter.any() else pts[:1]
    if len(pts) < 3:
        return [0.0] * n_coef
    fechado = np.vstack([pts, pts[:1]])
    seg = np.sqrt((np.diff(fechado, axis=0) ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    alvo = np.linspace(0.0, s[-1], n_pontos, endpoint=False)
    z = np.interp(alvo, s, fechado[:, 0]) + 1j * np.interp(alvo, s, fechado[:, 1])
    mag = np.abs(np.fft.fft(z))
    base = mag[1] + mag[-1]
    if base < 1e-12:
        return [0.0] * n_coef
    return [float((mag[k] + mag[-k]) / base) for k in range(2, 2 + n_coef)]


# --------------------------------------------------------------------------- cor e textura
def lab_real(img: np.ndarray) -> np.ndarray:
    """CIELAB do OpenCV (8 bits) -> unidades reais: L 0-100, a/b centrados em 0."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def n_cores(pix_lab: np.ndarray, semente: int, limiar_de: float, frac_min: float = 0.05) -> int:
    if len(pix_lab) < 12:
        return 1
    rng = np.random.default_rng(semente)
    if len(pix_lab) > 5000:
        pix_lab = pix_lab[rng.choice(len(pix_lab), 5000, replace=False)]
    dados = pix_lab.astype(np.float32)
    cv2.setRNGSeed(semente)
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, rotulos, centros = cv2.kmeans(dados, 6, None, criterio, 3, cv2.KMEANS_PP_CENTERS)
    frac = np.bincount(rotulos.ravel(), minlength=6) / float(len(dados))
    # une centros proximos (Delta E < limiar) - union-find simples
    pai = list(range(6))

    def raiz(i):
        while pai[i] != i:
            pai[i] = pai[pai[i]]
            i = pai[i]
        return i
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(centros[i] - centros[j]) < limiar_de:
                pai[raiz(i)] = raiz(j)
    grupos: Dict[int, float] = {}
    for i in range(6):
        grupos[raiz(i)] = grupos.get(raiz(i), 0.0) + float(frac[i])
    return max(1, sum(1 for f in grupos.values() if f >= frac_min))


def entropia(valores: np.ndarray, bins: int = 32) -> float:
    hist, _ = np.histogram(valores, bins=bins, range=(0, 100))
    p = hist[hist > 0] / float(hist.sum())
    return float(-(p * np.log2(p)).sum())


# --------------------------------------------------------------------------- processamento
def calcular(img: np.ndarray, m: np.ndarray, fov: np.ndarray, p: Dict) -> Tuple[Dict[str, float], Dict]:
    contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contornos:
        raise ValueError("mascara prevista vazia (segmentacao falhou na etapa 09)")
    c = max(contornos, key=cv2.contourArea)
    area = float((m > 0).sum())
    if area < p["area_min_px"]:
        raise ValueError(f"lesao segmentada pequena demais ({area:.0f} px)")
    perimetro = float(cv2.arcLength(c, True))
    hull = cv2.convexHull(c)
    area_hull = float(cv2.contourArea(hull))
    cx, cy, theta, l1, l2 = momentos_forma(m)
    a1, a2 = assimetria(alinhar(m, cx, cy, theta))

    pts = c.reshape(-1, 2).astype(np.float64)
    raios = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    lab = lab_real(img)
    L = lab[..., 0]
    gx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    nitidez = float(grad[pts[:, 1].astype(int), pts[:, 0].astype(int)].mean())

    lesao = m > 0
    pix = lab[lesao]
    lado = max(m.shape)
    k = int(max(3, round(p["anel"] * lado))) | 1
    anel = (cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0) & ~lesao & (fov > 0)
    pele = lab[anel] if anel.sum() > 20 else lab[(fov > 0) & ~lesao]
    delta_e = float(np.linalg.norm(pix.mean(axis=0) - pele.mean(axis=0))) if len(pele) else float("nan")

    Lf = L.astype(np.float64)
    media_loc = cv2.blur(Lf, (5, 5))
    var_loc = np.maximum(cv2.blur(Lf ** 2, (5, 5)) - media_loc ** 2, 0)
    hu = cv2.HuMoments(cv2.moments(m, binaryImage=True)).flatten()
    hu_log = [float(-np.sign(h) * np.log10(abs(h))) if abs(h) > 1e-30 else 0.0 for h in hu]
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)

    v = {
        "assimetria_eixo1": a1, "assimetria_eixo2": a2,
        "assimetria_media": (a1 + a2) / 2, "assimetria_max": max(a1, a2),
        "compacidade": perimetro ** 2 / (4 * np.pi * area),
        "solidez": float(cv2.contourArea(c)) / area_hull if area_hull > 0 else float("nan"),
        "irregularidade_raio": float(raios.std() / raios.mean()) if raios.mean() > 0 else float("nan"),
        "nitidez_borda": nitidez,
        "excentricidade": float(np.sqrt(max(0.0, 1.0 - l2 / l1))),
        "L_media": float(pix[:, 0].mean()), "L_desvio": float(pix[:, 0].std()),
        "a_media": float(pix[:, 1].mean()), "a_desvio": float(pix[:, 1].std()),
        "b_media": float(pix[:, 2].mean()), "b_desvio": float(pix[:, 2].std()),
        "deltaE_lesao_pele": delta_e,
        "n_cores_distintas": n_cores(pix, p["semente"], p["limiar_delta_e"]),
        "diametro_equivalente_px": float(np.sqrt(4 * area / np.pi)),
        "maior_eixo_px": float(max(rw, rh)),
        "area_relativa_fov": area / max(1.0, float((fov > 0).sum())),
        "entropia_L": entropia(pix[:, 0]),
        "desvio_local_L": float(np.sqrt(var_loc[lesao]).mean()),
    }
    v.update({f"hu_{i + 1}": hu_log[i] for i in range(N_HU)})
    v.update({f"fourier_{k2}": f for k2, f in zip(range(2, 2 + N_FOURIER), descritores_fourier(c))})
    extras = {"contorno": c, "hull": hull, "centro": (cx, cy), "theta": theta, "l1": l1, "l2": l2}
    return v, extras


def desenhar(img: np.ndarray, ex: Dict) -> np.ndarray:
    out = img.copy()
    cv2.drawContours(out, [ex["hull"]], -1, (255, 200, 0), 1)
    cv2.drawContours(out, [ex["contorno"]], -1, (0, 255, 0), 2)
    cx, cy = ex["centro"]
    for ang, lam, cor in ((ex["theta"], ex["l1"], (0, 0, 255)), (ex["theta"] + np.pi / 2, ex["l2"], (255, 0, 255))):
        r = 2.0 * np.sqrt(max(lam, 0.0))
        p1 = (int(round(cx - r * np.cos(ang))), int(round(cy - r * np.sin(ang))))
        p2 = (int(round(cx + r * np.cos(ang))), int(round(cy + r * np.sin(ang))))
        cv2.line(out, p1, p2, cor, 2)
    return out


def processar(linha: Dict, ctx: comum.Contexto) -> Dict:
    iid = linha["image_id"]
    if not linha.get("mascara_pred"):
        raise ValueError("manifesto sem mascara_pred: rode a etapa 09 antes")
    img = comum.ler_imagem(linha["imagem"])
    h, w = img.shape[:2]
    m = comum.ler_mascara(linha["mascara_pred"])
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    fov = comum.mascara_ou_cheia(linha.get("fov"), (h, w))
    valores, extras = calcular(img, m, fov, ctx.params)
    linha.update({k: round(v, 6) if isinstance(v, float) and np.isfinite(v) else v for k, v in valores.items()})
    if iid in ctx.exemplos:
        comum.salvar_exemplo(ctx.saida, iid, [
            ("imagem", img),
            ("contorno, fecho e eixos", desenhar(img, extras)),
        ])
    return linha


def main() -> int:
    p = comum.criar_parser("Etapa 10 - descritores ABCD, textura, momentos de Hu e descritores de Fourier.")
    p.add_argument("--anel", type=float, default=0.06,
                   help="Largura do anel de pele ao redor da lesao (fracao do lado da imagem).")
    p.add_argument("--limiar-delta-e", type=float, default=10.0,
                   help="Grupos de cor com Delta E menor que isto sao considerados a mesma cor.")
    p.add_argument("--area-min-px", type=int, default=50, help="Lesoes menores que isto sao descartadas.")
    args = p.parse_args()
    params = {"anel": args.anel, "limiar_delta_e": args.limiar_delta_e,
              "area_min_px": args.area_min_px, "semente": args.semente}
    codigo = comum.executar_etapa("10_descritores", args, processar, params,
                                  "Descritores ABCD + textura + Hu + Fourier sobre a mascara prevista.")
    saida = Path(args.saida)
    linhas, _ = comum.ler_manifesto(saida)
    base = ["image_id", "split", "rotulo", "alvo_melanoma", "iou", "dice"]
    comum.escrever_csv(saida / "descritores.csv", [{k: l.get(k, "") for k in base + NOMES} for l in linhas],
                       base + NOMES)
    comum.escrever_csv(saida / "dicionario_descritores.csv",
                       [{"atributo": n, "criterio": c, "descricao": d, "unidade": u} for n, c, d, u in DICIONARIO])
    print(f"descritores.csv: {len(linhas)} imagens x {len(NOMES)} atributos")
    return codigo


if __name__ == "__main__":
    sys.exit(main())
